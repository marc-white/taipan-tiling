"""The FWTiler object exists to encapsulate taipan.tiling settings, wrap functions
associated with taipan.core (i.e. objects derived from TaipanPoint) and perform FunnelWeb
specific tiling operations. The intent of this class is separate the logic of *tiling* 
from the objects being *tiled* in a way that makes the code easier to interpret. The 
class hosts the various setup parameters associated with the various tiling algorithms,
freeing up function calls from repeated parameters.

Function heirarchy as follows:
 - FWTiler.generate_tiling()
    - FWTiler.greedy_tile_mag_range()
        - FWTiler.greedy_tile_sky()
            - FWTiler.greedy_tile_sc() --> single-core
                    ****or****
            - FWTiler.greedy_tile_mc() --> multi-core

To profile this code for speed, run as:
    kernprof -l funnelweb_generate_tiling.py
And view with:
    python -m line_profiler funnelweb_generate_tiling.py.lprof
For other usage, visit:
    https://github.com/rkern/line_profiler

Where all functions with an #@profile decorator will be profiled - uncomment beforehand.
"""
import logging
import core as tp
import tiling as tl
import time
import random
import math
import numpy as np
import copy
import line_profiler
from joblib import Parallel, delayed
import multiprocessing as mp
import pdb
from collections import Counter
from scipy.spatial import cKDTree

class FWTiler(object):
    """FunnelWeb Tiler object to encapsulate tiling settings, wrap tiling functions, and
    perform tiling operations for FunnelWeb. 
    """
    def __init__(self, completeness_target=1.0, ranking_method='priority-expsum',
                 disqualify_below_min=True, tiling_method='SH', randomise_pa=True, 
                 randomise_SH=True, tiling_file='ipack.3.8192.txt', ra_min=0.0, 
                 ra_max=360.0, dec_min=-90.0, dec_max=90.0, 
                 mag_ranges=[[5,8],[7,10],[9,12],[11,14]],
                 mag_ranges_prioritise=[[5,7],[7,8],[9,10],[11,12]], priority_normal=2, 
                 prioritise_extra=2, tile_unpick_method='combined_weighted', 
                 combined_weight=4.0, sequential_ordering=(2, 1), rank_supplements=False, 
                 repick_after_complete=True, exp_base=3.0, recompute_difficulty=True, 
                 overwrite_existing=True, check_tile_radius=True, 
                 consider_removed_targets=False, allow_standard_targets=True, 
                 assign_sky_first=True, n_cores=1, backend="multiprocessing", 
                 enforce_min_tile_score=True):
        """Constructor for FWTiler. Takes as parameters the various settings needed to 
        unpick and rank individual tiles, as well as tile the sky as a whole.
        
        Parameters
        ----------
        completeness_target: float 
            A float in the range (0, 1] denoting the science target completeness to stop 
            at. Defaults to 1.0 (full completeness).
        
        ranking_method: string
            The scheme to use for ranking the tiles. See the documentation for 
            TaipanTile.calculate_tile_score for details.
        
        disqualify_below_min: boolean
            Denotes whether to rank tiles with a number of guides below 
            GUIDES_PER_TILE_MIN or a number of standards below STANDARDS_PER_TILE_MIN a 
            score of 0. Defaults to True.
        
        tiling_method: string
            The method by which to generate a tiling set. Currently, only 'SH' 
            (Sloane-Harding tiling centres) are available.
        
        randomise_pa: boolean
            Optional Boolean, denoting whether to randomise the pa of seed tiles or not. 
            Defaults to True.
        
        randomise_SH: boolean
            Optional Boolean, denoting whether or not to randomise the RA of the 'seed' of
            the SH tiling. Defaults to True.
        
        tiling_file: string
            The SH tiling file to use for generating tiling centres. Defaults to 
            'ipack.3.8192.txt'.
        
        ra_min, ra_max, dec_min, dec_max: float
            The RA and Dec bounds of the region to be considered, in decimal degrees. To 
            have an RA range spanning across 0 deg RA, either use a negative value for 
            ra_min, or give an ra_min > ra_max.
        
        mag_ranges: list
            The magnitude ranges for each set of tiles, of the form:
            [[r1_min, r1_max], [r2_min, r2_max], ....]
        
        mag_ranges_prioritise: list
            The magnitude ranges to add extra priority to within each set of tiles. Of the 
            form: [[r1p_min, r1p_max], [r2p_min, r2p_max], ....]
        
        priority_normal: float
            The standard priority level. Completeness is assessed at this priority level
            for stars in the priority magnitude range.
        
        prioritise_extra: float
            The additional priority to add within each of the mag_range_prioritise
        
        tile_unpick_method: string
            The scheme to be used for unpicking tiles. Defaults to 'sequential'. See the 
            documentation for TaipanTile.unpick_tile for details.
        
        combined_weight: float
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details.
            
        sequential_ordering: tuple 
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details.
        
        rank_supplements: boolean
            Optional Boolean value, denoting whether to attempt to assign guides/standards 
            in priority order. Defaults to False.
        
        repick_after_complete: boolean
            Boolean value, denoting whether to repick each tile after unpicking. Defaults 
            to True.
        
        exp_base : float, optional
            For priority-expsum, this is the base for the exponent (default 3.0)        
        
        recompute_difficulty: boolean
            Boolean value, denoting whether to recompute target difficulties after a tile 
            is moved to the results list. For FunnelWeb, it also means recompute for each 
            mag range. Defaults to True.
        
        overwrite_existing: boolean
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details. Defaults to true.
        
        check_tile_radius: boolean
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details. Defaults to true.
            
        consider_removed_targets: boolean
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details. Defaults to False.
            
        allow_standard_targets: boolean
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details. Defaults to true.
             
        assign_sky_first
            Additional argument to be used in the tile unpicking process. See the 
            documentation for TaipanTile.unpick_tile for details. Defaults to true.
            
        n_cores: int
            The number of processor cores to be used for the tiling process. n_cores=0 is
            the standard serial method, n_cores=1 is serial, but using the multiprocessing
            implementation, n_cores >= 2 is done in parallel.
        backend: string
            The backend to be used for the joblib library - either "threading" or 
            "multiprocessing".
        enforce_min_tile_score: boolean
            Whether a selected tile should be greater than some minimum tile score.
        """
        self._completeness_target = None
        self._ranking_method = None
        self._disqualify_below_min = None
        self._disqualify_below_min_original = None
        self._tiling_method = None
        self._randomise_pa = None
        self._randomise_SH = None
        self._tiling_file = None
        self._ra_min = None
        self._ra_max = None
        self._dec_min = None
        self._dec_max = None
        self._mag_ranges = None
        self._mag_ranges_prioritise = None
        self._priority_normal = None
        self._prioritise_extra = None
        self._tile_unpick_method = None
        self._combined_weight = None
        self._sequential_ordering = None
        self._rank_supplements = None
        self._repick_after_complete = None
        self._exp_base = None
        self._recompute_difficulty = None
        self._overwrite_existing = None
        self._check_tile_radius = None
        self._consider_removed_targets = None
        self._allow_standard_targets = None
        self._assign_sky_first = None
        self._n_cores = None
        self._backend = None
        self._enforce_min_tile_score = None
        self._enforce_min_tile_score_original = None    
        
        # Insert the passed values. Doing it like this forces the setter functions to be 
        # called, which provides error checking
        self.completeness_target = completeness_target
        self.ranking_method = ranking_method
        self.disqualify_below_min = disqualify_below_min
        self.disqualify_below_min_original = disqualify_below_min # "Memory" param
        self.tiling_method = tiling_method
        self.randomise_pa = randomise_pa
        self.randomise_SH = randomise_SH
        self.tiling_file = tiling_file
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.mag_ranges = mag_ranges
        self.mag_ranges_prioritise = mag_ranges_prioritise
        self.priority_normal = priority_normal
        self.prioritise_extra = prioritise_extra
        self.tile_unpick_method = tile_unpick_method
        self.combined_weight = combined_weight
        self.sequential_ordering = sequential_ordering
        self.rank_supplements = rank_supplements
        self.repick_after_complete = repick_after_complete
        self.exp_base = exp_base
        self.recompute_difficulty = recompute_difficulty
        self.overwrite_existing = overwrite_existing
        self.check_tile_radius = check_tile_radius
        self.consider_removed_targets = consider_removed_targets
        self.allow_standard_targets = allow_standard_targets
        self.assign_sky_first = assign_sky_first
        self.n_cores = n_cores
        self.backend = backend
        self.enforce_min_tile_score = enforce_min_tile_score
        self.enforce_min_tile_score_original = enforce_min_tile_score # "Memory" param
        
        # Not input params (don't currently have getters or setters)
        self.standard_tree = None
        self.guide_tree = None
   
    # ------------------------------------------------------------------------------------
    # Attribute handling
    # ------------------------------------------------------------------------------------   
    @property
    def completeness_target(self):
        return self._completeness_target

    @completeness_target.setter
    def completeness_target(self, value):
        if value is None: 
            raise Exception('completeness_target may not be blank')
        elif value <= 0. or value > 1:
            raise ValueError('completeness_target must be in the range (0, 1]')
        self._completeness_target = value
        
    @property
    def ranking_method(self):
        return self._ranking_method

    @ranking_method.setter
    def ranking_method(self, value):
        if value is None: 
            raise Exception('ranking_method may not be blank')
        self._ranking_method = value
    
    @property
    def disqualify_below_min(self):
        return self._disqualify_below_min

    @disqualify_below_min.setter
    def disqualify_below_min(self, value):
        if value is None: 
            raise Exception('disqualify_below_min may not be blank')
        self._disqualify_below_min = value
        
    @property
    def disqualify_below_min_original(self):
        return self._disqualify_below_min_original

    @disqualify_below_min_original.setter
    def disqualify_below_min_original(self, value):
        if value is None: 
            raise Exception('disqualify_below_min_original may not be blank')
        self._disqualify_below_min_original = value        
        
    @property
    def tiling_method(self):
        return self._tiling_method

    @tiling_method.setter
    def tiling_method(self, value):
        # Valid tiling methods
        TILING_METHODS = ['SH',]     # Sloane-Harding
    
        if value is None: 
            raise Exception('tiling_method may not be blank')
        elif value not in TILING_METHODS:
            raise ValueError('tiling_method must be one of %s' % str(TILING_METHODS))
        self._tiling_method = value
        
    @property
    def randomise_pa(self):
        return self._randomise_pa

    @randomise_pa.setter
    def randomise_pa(self, value):
        if value is None: 
            raise Exception('randomise_pa may not be blank')
        self._randomise_pa = value
        
    @property
    def randomise_SH(self):
        return self._randomise_SH

    @randomise_SH.setter
    def randomise_SH(self, value):
        if value is None: 
            raise Exception('randomise_SH may not be blank')
        self._randomise_SH = value
        
    @property
    def tiling_file(self):
        return self._tiling_file

    @tiling_file.setter
    def tiling_file(self, value):
        if value is None: 
            raise Exception('tiling_file may not be blank')
        self._tiling_file = value
        
    @property
    def ra_min(self):
        return self._ra_min

    @ra_min.setter
    def ra_min(self, value):
        if value is None: 
            raise Exception('ra_min may not be blank')
        self._ra_min = value
        
    @property
    def ra_max(self):
        return self._ra_max

    @ra_max.setter
    def ra_max(self, value):
        if value is None: 
            raise Exception('ra_max may not be blank')
        self._ra_max = value
    
    @property
    def dec_min(self):
        return self._dec_min

    @dec_min.setter
    def dec_min(self, value):
        if value is None: 
            raise Exception('dec_min may not be blank')
        self._dec_min = value
        
    @property
    def dec_max(self):
        return self._dec_max

    @dec_max.setter
    def dec_max(self, value):
        if value is None: 
            raise Exception('dec_max may not be blank')
        self._dec_max = value
        
    @property
    def mag_ranges(self):
        return self._mag_ranges

    @mag_ranges.setter
    def mag_ranges(self, value):
        if value is None: 
            raise Exception('mag_ranges may not be blank')
        self._mag_ranges = value
        
    @property
    def mag_ranges_prioritise(self):
        return self._mag_ranges_prioritise

    @mag_ranges_prioritise.setter
    def mag_ranges_prioritise(self, value):
        if value is None: 
            raise Exception('mag_ranges_prioritise may not be blank')
        self._mag_ranges_prioritise = value

    @property
    def priority_normal(self):
        return self._priority_normal

    @priority_normal.setter
    def priority_normal(self, value):
        if value is None: 
            raise Exception('priority_normal may not be blank')
        self._priority_normal = value
        
    @property
    def prioritise_extra(self):
        return self._prioritise_extra

    @prioritise_extra.setter
    def prioritise_extra(self, value):
        if value is None: 
            raise Exception('prioritise_extra may not be blank')
        self._prioritise_extra = value
        
    @property
    def tile_unpick_method(self):
        return self._tile_unpick_method

    @tile_unpick_method.setter
    def tile_unpick_method(self, value):
        if value is None: 
            raise Exception('tile_unpick_method may not be blank')
        self._tile_unpick_method = value
        
    @property
    def combined_weight(self):
        return self._combined_weight

    @combined_weight.setter
    def combined_weight(self, value):
        if value is None: 
            raise Exception('combined_weight may not be blank')
        self._combined_weight = value
        
    @property
    def sequential_ordering(self):
        return self._sequential_ordering

    @sequential_ordering.setter
    def sequential_ordering(self, value):
        if value is None: 
            raise Exception('sequential_ordering may not be blank')
        self._sequential_ordering = value
        
    @property
    def rank_supplements(self):
        return self._rank_supplements

    @rank_supplements.setter
    def rank_supplements(self, value):
        if value is None: 
            raise Exception('rank_supplements may not be blank')
        self._rank_supplements = value
        
    @property
    def repick_after_complete(self):
        return self._repick_after_complete

    @repick_after_complete.setter
    def repick_after_complete(self, value):
        if value is None: 
            raise Exception('repick_after_complete may not be blank')
        self._repick_after_complete = value
        
    @property
    def exp_base(self):
        return self._exp_base

    @exp_base.setter
    def exp_base(self, value):
        if value is None: 
            raise Exception('exp_base may not be blank')
        self._exp_base = value
    
    @property
    def recompute_difficulty(self):
        return self._recompute_difficulty

    @recompute_difficulty.setter
    def recompute_difficulty(self, value):
        if value is None: 
            raise Exception('recompute_difficulty may not be blank')
        self._recompute_difficulty = value
    
    @property
    def overwrite_existing(self):
        return self._overwrite_existing

    @overwrite_existing.setter
    def overwrite_existing(self, value):
        if value is None: 
            raise Exception('overwrite_existing may not be blank')
        self._overwrite_existing = value
        
    @property
    def check_tile_radius(self):
        return self._check_tile_radius

    @check_tile_radius.setter
    def check_tile_radius(self, value):
        if value is None: 
            raise Exception('check_tile_radius may not be blank')
        self._check_tile_radius = value
    
    @property
    def consider_removed_targets(self):
        return self._consider_removed_targets

    @consider_removed_targets.setter
    def consider_removed_targets(self, value):
        if value is None: 
            raise Exception('consider_removed_targets may not be blank')
        self._consider_removed_targets = value
        
    @property
    def allow_standard_targets(self):
        return self._allow_standard_targets

    @allow_standard_targets.setter
    def allow_standard_targets(self, value):
        if value is None: 
            raise Exception('allow_standard_targets may not be blank')
        self._allow_standard_targets = value
    
    @property
    def assign_sky_first(self):
        return self._assign_sky_first

    @assign_sky_first.setter
    def assign_sky_first(self, value):
        if value is None: 
            raise Exception('assign_sky_first may not be blank')
        self._assign_sky_first = value
            
    @property
    def n_cores(self):
        return self._n_cores

    @n_cores.setter
    def n_cores(self, value):
        if value is None: 
            raise Exception('n_cores may not be blank')
        self._n_cores = value
    
    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        if value is None: 
            raise Exception('backend may not be blank')
        self._backend = value

    @property
    def enforce_min_tile_score(self):
        return self._enforce_min_tile_score

    @enforce_min_tile_score.setter
    def enforce_min_tile_score(self, value):
        if value is None: 
            raise Exception('enforce_min_tile_score may not be blank')
        self._enforce_min_tile_score = value

    @property
    def enforce_min_tile_score_original(self):
        return self._enforce_min_tile_score_original

    @enforce_min_tile_score_original.setter
    def enforce_min_tile_score_original(self, value):
        if value is None: 
            raise Exception('enforce_min_tile_score_original may not be blank')
        self._enforce_min_tile_score_original = value
    
    @property
    def completeness_priority(self):
        return self.priority_normal + self.prioritise_extra
    
    @property
    def min_tile_score(self):
        # Return the minimum tile score - the score a tile would have it it had a single
        # priority target. When using this, we do not consider any tiles with a lower
        # score than this.
        return self.exp_base**self.completeness_priority    
       
    @property
    def unpick_settings(self):
        # This property exists to simplify the need for a non-method function when using
        # multiprocessing (as method functions cannot be pickled to be sent to the sub-
        # processes without some difficulty). The generated dictionary can be passed to
        # the now external function without the need for a large parameter list.
        return {"overwrite_existing":self.overwrite_existing, 
                "check_tile_radius":self.check_tile_radius,
                "recompute_difficulty":self.recompute_difficulty,
                "method":self.tile_unpick_method, 
                "combined_weight":self.combined_weight,
                "sequential_ordering":self.sequential_ordering,
                "rank_supplements":self.rank_supplements, 
                "repick_after_complete":self.repick_after_complete,
                "consider_removed_targets":self.consider_removed_targets, 
                "allow_standard_targets":self.allow_standard_targets,
                "assign_sky_first":self.assign_sky_first}
    
    
    # ------------------------------------------------------------------------------------
    # taipan.tiling wrapper functions
    # ------------------------------------------------------------------------------------    
    def gen_pa(self):
        """FWTiler wrapper for tiling.gen_pa(randomise_pa)
        """
        return tl.gen_pa(self.randomise_pa)
    
        
    def compute_bounds(self):
        """FWTiler wrapper for tiling.compute_bounds(ra_min, ra_max, dec_min, dec_max)
        
        Forces the stored RA/DEC limits to their correct format. Use after FWTiler object
        creation to ensure inputs are appropriate before commencing tiling.
        """
        ra_min, ra_max, dec_min, dec_max = tl.compute_bounds(self.ra_min, self.ra_max, 
                                                             self.dec_min, self.dec_max)
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max        
     
        
    def is_within_bounds(self, tile, compute_bounds_forcoords=True):
        """FWTiler wrapper for tiling.is_within_bounds(tile, ra_min, ra_max, dec_min, 
                                    dec_max, compute_bounds_forcoords=True)
                                    
        Parameters
        ----------
        tile: TaipanTile
            The TaipanTile instance to check.
        compute_bounds_forcoords: boolean
            Boolean value, denoting whether to use the tl.convert_bounds function
            function to ensure the bounds are in standard format. Defaults to True.
            
        Returns
        -------
        within_bounds: boolean
            Boolean value denoting whether the tile centre is within the bounds (True) or 
            not (False).
        """
        return tl.is_within_bounds(tile, self.ra_min, self.ra_max, self.dec_min, 
                                   self.dec_max, compute_bounds_forcoords)
    
    
    def generate_random_tile(self):
        """FWTiler wrapper for tiling.generate_random_tile(ra_min=0.0, ra_max=360.0,
                                    dec_min=-90.0, dec_max=90.0, randomise_pa=False)
        
        Returns
        -------
        tile: TaipanTile
            The generated tile.
        """
        return tl.generate_random_tile(self.ra_min, self.ra_max, self.dec_min, 
                                       self.dec_max, self.randomise_pa)
    
    
    def generate_SH_tiling(self):
        """FWTiler wrapper for tiling.generate_SH_tiling(tiling_file, randomise_seed=True, 
                                    randomise_pa=False)
        
        Returns
        -------
        tile_list: list of TaipanTile objects
            A list of TaipanTiles that have been generated from the Sloane-Harding tiling.
        """
        return tl.generate_SH_tiling(self.tiling_file, self.randomise_SH, 
                                     self.randomise_pa)
        
        
    # ------------------------------------------------------------------------------------
    # TaipanTile wrapper functions
    # ------------------------------------------------------------------------------------   
    def unpick_tile(self, tile, candidate_targets, standard_targets, guide_targets):
        """Wrapper function for TaipanTile.unpick_tile(...) to ensure tiling settings
        are kept as internal as possible (i.e. exposing FWTiler attributes in as few
        places as possible), and to aid in readability when unpicking a tile.
        
        Parameters
        ----------
        tile: TaipanTile
            The TaipanTile object to unpick.
        candidate_targets: list of TaipanTarget objects
            The potential science targets available for allocation to this TaipanTile.
        standard_targets: list of TaipanTarget objects
            The potential standard targets available for allocation to this TaipanTile.
        guide_targets: list of TaipanTarget objects
            The potential guide targets available for allocation to this TaipanTile.            
        """
        burn = tile.unpick_tile(candidate_targets, standard_targets, guide_targets,
                                overwrite_existing=self.overwrite_existing, 
                                check_tile_radius=self.check_tile_radius,
                                recompute_difficulty=self.recompute_difficulty,
                                method=self.tile_unpick_method, 
                                combined_weight=self.combined_weight,
                                sequential_ordering=self.sequential_ordering,
                                rank_supplements=self.rank_supplements, 
                                repick_after_complete=self.repick_after_complete,
                                consider_removed_targets=self.consider_removed_targets, 
                                allow_standard_targets=self.allow_standard_targets,
                                assign_sky_first=self.assign_sky_first)
        
        
    def calculate_tile_score(self, tile):
        """Wrapper function for TaipanTile.calculate_tile_score(...) to ensure tiling 
        settings are kept as internal as possible (i.e. exposing FWTiler attributes in as
        few places as possible), and to aid in readability when calculating tile scores.
        
        Parameters
        ----------
        tile: TaipanTile
            The TaipanTile to calculate the score of.
            
        Returns
        -------
        score: float
            The calculated score of the input TaipanTile.
        """
        return tile.calculate_tile_score(method=self.ranking_method,
                                         disqualify_below_min=self.disqualify_below_min, 
                                         combined_weight=self.combined_weight, 
                                         exp_base=self.exp_base)

        
    # ------------------------------------------------------------------------------------
    # FunnelWeb Tiling Helper Functions
    # ------------------------------------------------------------------------------------
    def get_targets_mag_range(self, candidate_targets, mag_range,  
                              mag_range_prioritise=None, last_range=False):
        """Function to determine the candidate targets for a given magnitude range.
    
        When using mag_range_prioritise to prioritise a subsection of the total magnitude 
        bin, if a target is excluded (but would be included in a subsequent bin), it will 
        only be observed if *low-priority*. This is to allow high priority (likely 
        fainter) targets to be observed with a longer exposure time in the next magnitude 
        bin (assuming bright-faint bin ordering). If the target is high-priority, and  
        outside the priority mag range, but there are no fainter bins, it will be 
        considered.
    
        Parameters
        ----------
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        mag_range: list of floats
            Upper and lower bounds of the magnitude bin under consideration. 
            Of form [L, U].
    
        mag_range_prioritise: list of floats, optional
            Upper and lower bounds of the priority magnitude range within the mag bin
            under consideration. If used, of form [L, U], else defaults to None. 
    
        last_range: boolean
            Indicates whether the magnitude range is the last bin under consideration, and
            thus whether high priority targets should be considered.
        
        Returns
        -------
        candidate_targets_range: list of :class:`TaipanTarget`
            The candidate targets that satisfy magnitude range and priority requirements.
        """ 
        print "Determining candidate targets for magnitude range...",
        #Find the candidates in the correct magnitude range. If we are not in the faintest
        #magnitude range, then we have to ignore high priority targets for now, as we'd
        #rather them be observed with a long exposure time
        if mag_range_prioritise:
            if not last_range:
                candidate_targets_range = [t for t in candidate_targets 
                    if ( (mag_range_prioritise[1] <= t.mag < mag_range[1]) and #faint
                    (t.priority <= self.priority_normal) ) or
                    ( (mag_range[0] <= t.mag < mag_range_prioritise[1]) )] #bright
            
                # Increase the priority for targets in the priority magnitude range
                for t in candidate_targets_range:
                    if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                        t.priority >= self.priority_normal):
                        t.priority += self.prioritise_extra
        
            # In faintest bin, consider all targets
            else:
                candidate_targets_range = [t for t in candidate_targets
                    if (mag_range[0] <= t.mag < mag_range[1])]
                for t in candidate_targets_range:
                    if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                        t.priority == self.priority_normal):
                        t.priority += self.prioritise_extra
            
            print "done, # Targets=%i" % len(candidate_targets_range)      
            return set(candidate_targets_range)


    def get_standards_mag_range(self, standard_targets, mag_range):
        """Function to select standard stars from within a given magnitude bin.
    
        Parameters
        ----------
        standard_targets: list of :class:`TaipanTarget`
            List of all available standard stars.
    
        mag_range: list of floats
            Upper and lower bounds of the magnitude bin under consideration. 
            Of form [L, U].
        
        Returns
        -------
        standard_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
        """
        print "Determining standard targets for magnitude range...",
        standard_targets_range = [t for t in standard_targets 
                                  if mag_range[0] <= t.mag < mag_range[1]]
        
        print "done, # Standards=%i" % len(standard_targets_range)      
        return standard_targets_range


    def get_guides_mag_range(self, guide_targets, candidate_targets_range):
        """Function to select guide stars from within a given magnitude bin.
    
        At present the only criteria for being considered a guide is to *not* be a 
        candidate target for the range.
    
        TODO: Consider magnitude of guide stars.
    
        Parameters
        ----------
        guide_targets: list of :class:`TaipanTarget`
            List of all available guide stars.
    
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
        
        Returns
        -------
        non_candidate_guide_targets: list of :class:`TaipanTarget`
            The guide targets that requirements.
        """
        print "Determining guide targets for magnitude range...",
        # Find the guides that are not candidate targets only. These have to be copied, 
        # because the same target will be a guide for one field and not a guide for 
        # another field.
        #non_candidate_guide_targets = [guide for guide in guide_targets 
        #                               if guide not in candidate_targets_range]
        
        # New set with elements in guide_targets but not candidate_targets_range
        non_candidate_guide_targets = guide_targets.difference(candidate_targets_range)
        
        non_candidate_guide_targets = copy.deepcopy(non_candidate_guide_targets)
        
        # Set the flags appropriately
        for target in non_candidate_guide_targets:
            target.guide = True 
            target.standard = False
            target.science = False      
        
        print "done, # Guides=%i" % len(non_candidate_guide_targets)      
        return list(non_candidate_guide_targets)
 
 
    def calc_priority_targets(self, candidate_targets):
        """Calculates the number of priority targets given the candidate targets and the 
        completeness priority.
    
        Parameters
        ----------
        candidate_targets: list of :class:`TaipanTarget`
            The list of candidate targets to be evaluated based on priority.
        
        Returns
        -------
        n_priority_targets: int
            Number of targets considered a priority for the given completeness_priority.
        """
        # The number of priority targets is the number of targets above 
        # completeness_priority. This includes some targets that are also standards.
        n_priority_targets = 0
        for target in candidate_targets:
            if target.priority >= self.completeness_priority:
                n_priority_targets += 1
        #if n_priority_targets == 0:
            #raise ValueError('Require some priority targets in each mag range!')
        
        return n_priority_targets
        
        
    def log_tiling_progress(self, local_candidate_targets, n_priority_targets, 
                            remaining_priority_targets, num_candidate_targets_range, 
                            process_i, ra, dec, best_rank, num_assigned_priority, 
                            num_assigned_candidates):
        """Prints a summary of the current tiling progress and details of the most 
        recently assigned tile.
        
        Parameters
        ----------
        local_candidate_targets: list of :class: `TaipanTarget`
            The candidate targets within the bounds of the assigned tile (i.e. within a
            single tile radii of the given ra and dec).
        
        n_priority_target: int
            The total number of priority targets to allocate.
        
        remaining_priority_targets: int
            The number of priority targets remaining to be allocated.
        
        num_candidate_targets_range: int
            The number of candidate targets remaining in the magnitude range.
        
        process_i: int
            The process ID (0, 1, 2, 3, ...) of the the process that generated the tile.
        
        ra: float
            The right-ascension coordinate of the most recently assigned tile.
        
        dec: float
            The declination coordinate of the most recently assigned tile.
        
        best_rank: int
            The tile score of the highest ranked (and assigned) tile.
            
        num_assigned_priority: int
            The number of priority targets assigned to the best tile.
            
        num_assigned_candidates: int
            The number of candidates assigned to the tile (including priority targets).
        """
        nearby_priority = self.calc_priority_targets(local_candidate_targets)
        nearby_candidate = len(local_candidate_targets)
        
        num_non_priority_candidates = num_assigned_candidates - num_assigned_priority
        
        # Calculate the current completeness (cc)
        cc = (float(n_priority_targets - remaining_priority_targets)
              / float(n_priority_targets))
        print time.strftime("%H:%M:%S %d/%m/%y") + ";",
        print "%5i / %5i P" % (remaining_priority_targets, n_priority_targets),                                          
        print "[%5i C ] --> %5.2f%%;" % (num_candidate_targets_range, 100*cc),
        print "assigned: P =%3i, C =%3i;" % (num_assigned_priority,
                                           num_non_priority_candidates),
        print "local: P =%4i, C =%4i;" % (nearby_priority, nearby_candidate),
        print "PID #%i, RA=%6.2f, DEC=%6.2f," % (process_i, ra, dec),
        print "rank = %5i %s" % (best_rank, "T" if self.disqualify_below_min else "F")  
        
              
    # ------------------------------------------------------------------------------------
    # FunnelWeb Tiling Functions
    # ------------------------------------------------------------------------------------
    #@profile
    def get_best_distant_tile(self, candidate_tiles, ranking_list, best_tiles=None, 
                              n_radii=6):
        """Selects the highest ranked tile from tile_list that is considered distant from
        all tiles in best_tiles.
        
        Todo: check to avoid indexing errors.
        
        Parameters
        ----------
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
        
        ranking_list: list of floats
            List of tile scores corresponding to candidate_tiles.
            
        best_tiles: list of :class: `TaipanTile`
            Already selected list of best tiles that any new tile must be considered
            distant from.
            
        n_radii: int
            The minimum separation between any selected tile and those in best_tiles.    
            
        Returns
        -------
        best_tile: TaipanTile object or None
            Either the highest ranked distant tile, or None if there are none to be found.
            
        best_rank: float
            Ranking of the selected best tile.
        """
        if best_tiles and len(best_tiles) > 0:
            # While loop setup: sort indices of ranking list, initialise bool check/count
            # ranking_list_i_sorted is sorted min-max so we index from the end (hence -1)
            ranking_list_i_sorted = np.argsort(ranking_list)
            found_best_distant_tile = False
            nth_max = -1
            
            # Consider from max-min the best tiles until we find a distant one
            while not found_best_distant_tile:
                best_tile_i = ranking_list_i_sorted[nth_max]
                candidate_ra = candidate_tiles[best_tile_i].ra
                candidate_dec =  candidate_tiles[best_tile_i].dec
                
                nearby_best_tiles = tp.targets_in_range(candidate_ra, candidate_dec, 
                                                        best_tiles, 
                                                        n_radii*tp.TILE_RADIUS)
                                                        
                if len(nearby_best_tiles) == 0 and ranking_list[best_tile_i] > 0:
                    # Previous best tiles can be considered distant *and* the newly 
                    # selected tile is not disqualified for not having the minimum number
                    # of standards and guides --> select this tile
                    found_best_distant_tile = True
                    
                else:
                    # The list is not empty, thus at least one of the other best tiles
                    # can be considered close --> move on to the next highest ranked tile
                    # Note: as long as we are never using an unreasonable number of 
                    # processors (e.g. equal to a significant fraction of the total 
                    # number of tiles on the sky) we should never reach the end of the 
                    # ranking list for and thus do not need checks against it. It is 
                    # however possible for us to reach the completion target without using
                    # the best available remaining targets.
                    nth_max -= 1
                    
                    # We have no exhausted ranking list and cannot find a suitable tile
                    if np.abs(nth_max) > len(ranking_list_i_sorted):
                        return None, None
        
        # Either we are just selecting the first best tile, or will only be selecting one
        else:
            # Find the highest-ranked tile in the candidates_list
            best_tile_i = np.argmax(ranking_list)
        
        # We now have the index of the best tile --> select it    
        best_tile = candidate_tiles.pop(best_tile_i) 
        best_rank = ranking_list.pop(best_tile_i)
        
        logging.info('Best tile has ranking score %3.1f' % (best_rank, ))
        
        return best_tile, best_rank
   
    
    #@profile
    def replace_best_tile(self, best_tile, candidate_tiles, candidate_targets, 
                          candidate_targets_range):
        """Function to select the highest ranked tile for the final tiling, and 
        re-generate a replacement. 
    
        Any science targets allocated to this tile are then removed from consideration
        for future tiles, and any neighbouring tiles with now allocated targets are
        repicked (reassigned).
    
        Parameters
        ----------
        best_tile: object of :class: `TaipanTile`
            The highest ranked tile to be replaced and have its allocated targets removed
            from further consideration.
        
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
        
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
        
        Returns
        -------
        assigned_priority: int
            The number of priority targets allocated to best_tile.
            
        num_assigned_candidates: int
            The number of candidates assigned to the tile (including priority targets).
        """
        # Record the ra and dec of the candidate for tile re-creation
        best_ra = best_tile.ra
        best_dec = best_tile.dec

        # Strip the now-assigned targets out of the candidate_targets and 
        # candidate_targets_range lists, thus removing them from further consideration,
        # then recalculate difficulties for affected remaning targets
        logging.info('Re-computing target list...')
        assigned_targets = best_tile.get_assigned_targets_science()
        
        num_assigned_priority = 0
        num_assigned_candidates = len(assigned_targets)

        init_targets_len = len(candidate_targets_range)
        reobserved_standards = []
        
        for target in assigned_targets:
            # Note: when candidate_targets contains stars with higher than normal 
            # *initial* priorities, it is possible for a star to be overlooked for
            # consideration as a science target (observations being preferred in a 
            # fainter magnitude bin for "high-priority" targets), but still 
            # considered for selection as a standard target. As such is important
            # to use candidate_targets_range, rather than simply candidate_targets 
            # when dealing with assigned targets.
            if target in candidate_targets_range:
                candidate_targets.remove(target)
                candidate_targets_range.remove(target)

                # Count the priority targets we've just assigned in the same way
                # as they were originally counted
                if target.priority >= self.completeness_priority:
                    num_assigned_priority += 1
        
                # Change priorities back to normal for targets in our priority 
                # magnitude range
                target.priority = target.priority_original
                  
            elif target.standard:
                reobserved_standards.append(target)
                logging.info('Re-allocating standard ' + str(target.idn) + 
                             ' that is also a science target.')
            else:
                logging.warning('### WARNING: Assigned a target that is neither a ' + 
                                'candidate target nor a standard!')
         
        if len(set(assigned_targets)) != len(assigned_targets):
            logging.warning('### WARNING: target duplication detected')
        if len(candidate_targets_range) != (init_targets_len - len(assigned_targets) 
                                            + len(reobserved_standards)):
            logging.warning('### WARNING: Discrepancy found in target list reduction')
            logging.warning('Best tile had %d targets; only %d removed from list' %
                            (len(assigned_targets),
                             init_targets_len - len(candidate_targets)))
        if self.recompute_difficulty:
            # Create a temporary list version of our candidate targets range set
            candidate_targets_range_list = list(candidate_targets_range)
            
            logging.info('Re-computing target difficulties...')
            tp.compute_target_difficulties(tp.targets_in_range(best_ra, best_dec, 
                                           candidate_targets_range_list,
                                           tp.TILE_RADIUS + tp.FIBRE_EXCLUSION_DIAMETER))

        # Replace the removed tile in candidate_tiles
        candidate_tiles.append(tp.TaipanTile(best_ra, best_dec, pa=self.gen_pa()))
    
        logging.info('Assigned tile at %3.1f, %2.1f' % (best_ra, best_dec))
    
        return num_assigned_priority, num_assigned_candidates

    
    #@profile
    def get_tile_neighbourhood(self, ra, dec, candidate_tiles, candidate_targets_range, 
                               standard_targets_range, guide_targets_range, 
                               n_tile_radii=2, n_target_radii=3, 
                               remove_tiles_from_master_list=True):
        """Pop all tiles, targets, standards, and guides from the RA and DEC neighbourhood
        into new lists (in the process removing them from the master lists). The master 
        lists will be updated by reference. This function is required (presently) for the
        multiprocessing code as Taipan object equivalence is not done at the object level.        
    
        Parameters
        ----------
        ra: float
            The RA coordinate at the centre of the neighbourhood.
        
        dec: float
            The DEC coordinate at the centre of the neighbourhood.
    
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
        
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
            
        standard_targets_range: list of :class:`TaipanTarget`
            Subset of standard_targets consisting of only the standard targets that are 
            within the current magnitude range.         
            
        guide_targets_range: list of :class:`TaipanTarget`
            Subset of guide_targets consisting of only the guide targets that are within 
            the current magnitude range. 
    
        n_tile_radii: float
            The number of tile radii out to consider for the neighbourhood.
            
        n_target_radii: float
            The number of target radii out to consider for the neighbourhood.    
        
        remove_tiles_from_master_list: boolean
            Whether to actually remove the neighbourhood from the master candidate_tile 
            list. This should be set to true whenever running a multiprocessing 
            implementation, as the *copies* of the tiles will be modified and returned, 
            but can be set to false for single core implementations. Defaults to true.
    
        Returns
        -------
        nearby_tiles: list of :class: `TaipanTile`
            The list of neighbouring tiles.
        
        nearby_targets: list of :class:`TaipanTarget`
            The list of nearby targets.
            
        nearby_standards: list of :class:`TaipanTarget`
            The list of nearby standards.        
            
        nearby_guides: list of :class:`TaipanTarget`
            The list of nearby guides.
        """    
        # Create lists of all neighbouring tiles, and candidate science/standard/guides
        nearby_tiles = tp.targets_in_range(ra, dec, candidate_tiles, 
                                           n_tile_radii*tp.TILE_RADIUS) 
   
        nearby_targets = tp.targets_in_range(ra, dec, candidate_targets_range, 
                                             n_target_radii*tp.TILE_RADIUS) 
    
        nearby_standards = tp.targets_in_range(ra, dec, standard_targets_range, 
                                               n_target_radii*tp.TILE_RADIUS,
                                               tree=self.standard_tree) 
                                                
        nearby_guides = tp.targets_in_range(ra, dec, guide_targets_range, 
                                            n_target_radii*tp.TILE_RADIUS,
                                            tree=self.guide_tree) 
    
        # Remove candidate tiles from the master list - these need to be updated with the
        # copies that come back from the other processes. No need to do the same for
        # targets, standards, or guides.
        if remove_tiles_from_master_list:
            for tile in nearby_tiles:
                candidate_tiles.remove(tile) 

        # All done, return lists of nearby tiles/targets/standards/guides
        return nearby_tiles, nearby_targets, nearby_standards, nearby_guides
    
    
    #@profile
    def greedy_tile_sc(self, candidate_tiles, candidate_targets, candidate_targets_range, 
                       standard_targets_range, guide_targets_range, ranking_list, 
                       n_priority_targets, remaining_priority_targets):
        """
        Parameters
        ----------
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
            
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
    
        standard_targets_range: list of :class:`TaipanTarget`
            Subset of standard_targets consisting of only the standard targets that are 
            within the current magnitude range. 
        
        guide_targets_range: list of :class:`TaipanTarget`
            Subset of guide_targets consisting of only the guide targets that are within 
            the current magnitude range. 
        
        ranking_list: list of int
            List of tile scores with indices corresponding to candidate_tiles.
            
        n_priority_targets: int
            The total number of priority targets to allocate.
        
        remaining_priority_targets: int
            The number of priority targets remaining to be allocated.
            
        Returns
        -------
        best_tile: TaipanTile
            The selected best candidate tile, with targets now considered allocated.
            
        remaining_priority_targets: int
            The updated number of priority targets remaining to be allocated.
        """
        # Find best tile
        best_tile, best_rank = self.get_best_distant_tile(candidate_tiles, ranking_list)                                                 
        
        # Replace the best tile and remove its assigned targets from further consideration
        num_assigned_priority, num_assigned_candidates = self.replace_best_tile(
                                                           best_tile, candidate_tiles, 
                                                           candidate_targets, 
                                                           candidate_targets_range)
        
        # Create a temporary list version of our candidate targets range set
        candidate_targets_range_list = list(candidate_targets_range)
        
        # Update the count for remaining priority targets                                            
        remaining_priority_targets -= num_assigned_priority
        
        # Implement tiered approach and get neighbourhood (tiles/targets/standards/guides)
        # of the central tile. By doing this, further calls to tp.targets_in_range from
        # within repick_within_range do not have to use the full lists (which might be 
        # millions of targets long), but can instead search the neighbourhood which will 
        # by definition contain fewer targets. Make sure not to remove the tiles from the
        # master candidate_tiles list.
        nearby_tiles, nearby_targets, nearby_standards, nearby_guides = \
                self.get_tile_neighbourhood(best_tile.ra, best_tile.dec, 
                                            candidate_tiles, candidate_targets_range_list,  
                                            standard_targets_range, guide_targets_range,  
                                            n_tile_radii=2, n_target_radii=3,
                                            remove_tiles_from_master_list=False)  
        
        # Create an update on the tiling progress
        local_candidate_targets = tp.targets_in_range(best_tile.ra, best_tile.dec, 
                                                      nearby_targets, 1*tp.TILE_RADIUS)

        self.log_tiling_progress(local_candidate_targets, n_priority_targets, 
                                 remaining_priority_targets, len(candidate_targets_range), 
                                 0, best_tile.ra, best_tile.dec, best_rank, 
                                 num_assigned_priority, num_assigned_candidates)
        
        # Repick all tiles within a given radius of the selected tile
        repick_within_radius(best_tile, nearby_tiles, nearby_targets, nearby_standards, 
                             nearby_guides, self.unpick_settings)
                         
        return best_tile, remaining_priority_targets
        
    #@profile    
    def greedy_tile_mc(self, candidate_tiles, candidate_targets, candidate_targets_range, 
                       standard_targets_range, guide_targets_range, ranking_list, 
                       n_priority_targets, remaining_priority_targets):
        """
        Parameters
        ----------
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
            
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
    
        standard_targets_range: list of :class:`TaipanTarget`
            Subset of standard_targets consisting of only the standard targets that are 
            within the current magnitude range. 
        
        guide_targets_range: list of :class:`TaipanTarget`
            Subset of guide_targets consisting of only the guide targets that are within 
            the current magnitude range. 
        
        ranking_list: list of int
            List of tile scores with indices corresponding to candidate_tiles.
            
        n_priority_targets: int
            The total number of priority targets to allocate.
        
        remaining_priority_targets: int
            The number of priority targets remaining to be allocated.
            
        Returns
        -------
        best_tiles: list of :class: `TaipanTile`
            List of the n selected best candidate tiles, with targets now considered
            allocated.
            
        remaining_priority_targets: int
            The updated number of priority targets remaining to be allocated.
        """
        # Dictionary of form:
        # {TaipanTile:([Tiles],[Targets],[Standards],[Guides]),}
        best_tiles = {}
        
        # Shared resources for multiprocessing.pool implementation
        if self.backend == "pool":
            manager = mp.Manager()
            tile_neighbourhood = manager.list()
            repicked_tiles = manager.list()
        
        num_candidate_targets_range = len(candidate_targets_range)
    
        # This loop builds up the dictionary consisting of the best tile and its
        # neighbouring tiles/targets/standards/guides until it has length equal to
        # the # of cores available. Once built, each entry in the dictionary is
        # sent off to a different process to have the neighbourhood repicked.
        # TODO: Have contingency if for some reason we cannot select enough tiles
        for process_i in xrange(0, self.n_cores):
            # Find best tile that is >=6 tiles away from other best tiles
            # If best_tiles.keys() is empty, the best tile will be returned
            nth_best_tile, best_rank = self.get_best_distant_tile(candidate_tiles,
                                                                  ranking_list,
                                                                  best_tiles.keys(), 
                                                                  n_radii=6)
            
            # We were unsuccessful in finding tiles up to n_cores, proceed with
            # what we have and break out of the for loop
            # Note: assumption is that we will never reach this point on the first
            # iteration of this loop - a situation that could only arise if *all* tile
            # scores are zero.
            if not nth_best_tile:
                logging.info("nth_best_tile is None, aborting filling to n_cores")
                break
                
                # Should never get here, but just in case
                if process_i == 0:
                    raise Exception("Greedy MC failed without selecting a tile!")
            
            # Replace the best tile and remove its assigned targets from further
            # consideration
            num_assigned_priority, num_assigned_candidates = self.replace_best_tile(
                                                               nth_best_tile, 
                                                               candidate_tiles, 
                                                               candidate_targets, 
                                                               candidate_targets_range)
                                                    
            # Update the count for remaining priority targets                                            
            remaining_priority_targets -= num_assigned_priority                                        
            
            # Create a temporary list version of our candidate targets range set
            candidate_targets_range_list = list(candidate_targets_range)
            
            # Create lists of all neighbouring affected tiles/targets/standards/guides
            nearby_tiles, nearby_targets, nearby_standards, nearby_guides = \
                self.get_tile_neighbourhood(nth_best_tile.ra, nth_best_tile.dec, 
                                            candidate_tiles, candidate_targets_range_list,  
                                            standard_targets_range, guide_targets_range,  
                                            n_tile_radii=2, n_target_radii=3)  
            
            # Create an update on the tiling progress
            local_candidate_targets = tp.targets_in_range(nth_best_tile.ra, 
                                                          nth_best_tile.dec, 
                                                          nearby_targets, 
                                                          1*tp.TILE_RADIUS)

            self.log_tiling_progress(local_candidate_targets, n_priority_targets, 
                                     remaining_priority_targets, 
                                     len(candidate_targets_range), process_i, 
                                     nth_best_tile.ra, nth_best_tile.dec, best_rank,
                                     num_assigned_priority, num_assigned_candidates)
                                                                  
            # Add an entry to the dictionary to be sent to the subprocess
            if self.backend == "multiprocessing" or self.backend == "threading":
                best_tiles[nth_best_tile] = (nearby_tiles[:], nearby_targets[:],
                                             nearby_standards[:], nearby_guides[:])
            elif self.backend == "pool":
                # Instead add the neighbourhood information to the manager
                tile_neighbourhood.append((nth_best_tile, nearby_targets, 
                                           nearby_standards, nearby_guides, nearby_tiles))
                                           
                # But still add the best tiles to the dictionary
                best_tiles[nth_best_tile] = process_i
            
            else:
                raise Exception("Warning: Unidentified backend: " + self.backend) 
                
            # Recalculate the ranking list to account for the now missing items
            ranking_list = [self.calculate_tile_score(tile) for tile in candidate_tiles] 
            
            # If max of the ranking_list is now 0, abort filling up to n_cores
            # Note: by aborting here, any already created processes (at least one) will be
            # run, but we won't create any more. Should not matter what the status of 
            # self.disqualify_below_min is - don't want any targets with a score of 0.
            if max(ranking_list) < 0.05:
                logging.info("max(ranking_list) < 0.05, abort filling to n_cores")
                break
        
        if self.backend == "multiprocessing" or self.backend == "threading":
            # Dictionary is constructed, now perform parallel repick
            n_processes = len(best_tiles.keys())
            results = Parallel(n_jobs=n_processes, backend=self.backend)(
                         delayed(repick_within_radius)(tile, best_tiles[tile][0],
                                                       best_tiles[tile][1],
                                                       best_tiles[tile][2],
                                                       best_tiles[tile][3],
                                                       self.unpick_settings)
                                                 for tile in best_tiles.keys())  
    
            # Done, now add back in the nearby candidate tiles
            for updated_nearby_candidate_tiles in results:
                candidate_tiles.extend(updated_nearby_candidate_tiles)
            
        elif self.backend == "pool":
            # Try using multiprocessing module, rather than joblib
            #raise Exception("Error: no other MP method yet implemented")
            n_processes = len(tile_neighbourhood)
            pool = mp.Pool(processes=n_processes)

            pool.map(repick_within_radius_pool, [(process_i, tile_neighbourhood,  
                                             repicked_tiles, self.unpick_settings)
                                           for process_i in xrange(0, n_processes)])
            pool.close()
            pool.join()
            
            # Done, now add back in the nearby candidate tiles
            for neighbourhood in repicked_tiles:
                candidate_tiles.extend(neighbourhood)
           
        return best_tiles.keys(), remaining_priority_targets
        

    #@profile
    def greedy_tile_sky(self, candidate_targets, candidate_targets_range,  
                        standard_targets_range, guide_targets_range, candidate_tiles, 
                        mag_range):
        """Function to perform the greedy tiling algorithm given targets, standards, 
        guides, and a selection of tiles.
        
        Depending on the value FWTiler.n_cores, this function will either employ a non-
        multiprocessing, or multiprocessing variant of the greedy tiling algorithm. The
        algorithm is described as follows:
            A) While not at the completeness target, and there are still valid tiles:
                1) Do the following up to the number of available cores:
                    - Select the Nth best tile, separated by more than 6 tile radii on-
                      sky from any/all other best tiles, and add to final tiling. 
                    - Replace the best tile and update the remaining number of priority
                      targets.
                    - Collate all affected tiles within a 2 tile radii of the best tile,
                      as well as those targets within 3 tile radii (popped into
                      separate listd as to only consider relevant targets). 
                2) Repick the affected tiles in a parallel environment, one process for 
                   each best tile and set of associated tiles/targets.
                3) Now back in the standard environment, reassemble the separate lists 
                   passed to each sub-process, and update the master candidate lists 
                   accordingly.
            B) Tiling complete, return the resulting set of tiles.                   
    
        Parameters
        ----------
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        candidate_targets_range: list of :class:`TaipanTarget`
            Subset of candidate_targets consisting of only the targets that satisfy the 
            current magnitude range and priority requirements
    
        standard_targets_range: list of :class:`TaipanTarget`
            Subset of standard_targets consisting of only the standard targets that are 
            within the current magnitude range. 
        
        guide_targets_range: list of :class:`TaipanTarget`
            Subset of guide_targets consisting of only the guide targets that are within 
            the current magnitude range. 
            
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
            
        mag_range: [min_mag, max_mag]
            List of the bounds of the current magnitude range.
    
        Returns
        -------
        tile_list: list of :class: `TaipanTile`
            The set of tiles meeting the completeness requirement for the provided set of
            targets.
        """
        # Create a temporary list version of our candidate targets range set
        candidate_targets_range_list = list(candidate_targets_range)
                   
        # Compute the initial difficulties of all candidate tiles
        if self.recompute_difficulty:
            start = time.time()
            logging.info("Computing difficulties...")
            print ("Computing target difficulties..."),
            tp.compute_target_difficulties(candidate_targets_range_list)
            finish = time.time()
            delta = finish - start
            print ("done in %d:%02.1f") % (delta/60, delta % 60.)
        
        # Create the initial unpick of every tile, making sure to only pass it those
        # targets that it can actually tile. Duplicates are not a concern at this stage,
        # nor are updating the target difficulties - both will be dealt with later.
        logging.info('Creating initial tile unpicks...')
        print "Performing initial global tile unpicks...",
        start = time.time()
        
        # Create temporary version of candidate_tiles
        temp_candidate_tiles = candidate_tiles[:]
        
        # Pre-generate the KD tree for the standards and guide lists (which do not change)
        self.standard_tree = precompute_kd_tree(standard_targets_range)
        self.guide_tree = precompute_kd_tree(guide_targets_range)
        
        while len(temp_candidate_tiles) > 0:  
            # Get the neighbourhood of an arbitrary tile, unpick the neighbours, then
            # remove all from the set. Doing the initial unpicking this way has the 
            # potential to be 7x faster as we make fewer calls to tp.targets_in_range with
            # the full length target/standard/guide lists.
            central_tile = temp_candidate_tiles[-1]
            
            # Get the neighbourhood tiles, targets, standards, and guides. The selected
            # tiles are removed from the list
            nearby_tiles, nearby_targets, nearby_standards, nearby_guides = \
                self.get_tile_neighbourhood(central_tile.ra, central_tile.dec, 
                                            temp_candidate_tiles, 
                                            candidate_targets_range_list,  
                                            standard_targets_range, guide_targets_range,  
                                            n_tile_radii=2, n_target_radii=3) 
            
            for tile in nearby_tiles:
                # Pre-generate the KD tree for the nearby standards and guide lists. These
                # will not change within each iteration of the loop
                nearby_standard_tree = precompute_kd_tree(nearby_standards)
                nearby_guide_tree = precompute_kd_tree(nearby_guides)
                
                # Unpick each tile in the neighbourhood
                self.unpick_tile(tile, 
                                 tp.targets_in_range(tile.ra, tile.dec, nearby_targets, 
                                                     1*tp.TILE_RADIUS), 
                                 tp.targets_in_range(tile.ra, tile.dec, nearby_standards, 
                                                     1*tp.TILE_RADIUS,
                                                     tree=nearby_standard_tree), 
                                 tp.targets_in_range(tile.ra, tile.dec, nearby_guides, 
                                                     1*tp.TILE_RADIUS,
                                                     tree=nearby_guide_tree)) 
        
        finish = time.time()
        delta = finish - start
        print ("done in %d:%02.1f") % (delta/60, delta % 60.)
        
        # Compute initial rankings for all of the tiles
        ranking_list = [self.calculate_tile_score(tile) for tile in candidate_tiles]      
                    
        # Calculate priority targets
        n_priority_targets = self.calc_priority_targets(candidate_targets_range_list)
        
        remaining_priority_targets = n_priority_targets

        # While we are below our completeness criteria AND the highest-ranked tile is not
        # empty, perform the greedy algorithm
        logging.info('Starting greedy/Funnelweb tiling allocation...')        
        tile_list = []
        tile_i = 0
        
        # Note: 0.05 is a simple proxy for max > 0
        while ((float(n_priority_targets - remaining_priority_targets) 
               / float(n_priority_targets)) < self.completeness_target) and \
               (max(ranking_list) > 0.05): 
            # Single core
            if self.n_cores == 0:
                best_tile, remaining_priority_targets = self.greedy_tile_sc(
                                                            candidate_tiles, 
                                                            candidate_targets, 
                                                            candidate_targets_range, 
                                                            standard_targets_range, 
                                                            guide_targets_range, 
                                                            ranking_list, 
                                                            n_priority_targets, 
                                                            remaining_priority_targets)                                            
            
                # Add the magnitude range information
                best_tile.mag_min = mag_range[0]
                best_tile.mag_max = mag_range[1]
                
                # Lastly add the best tile to the set of selected tiles
                tile_list.append(best_tile)   
                
            # Multicore    
            else:
                best_tiles, remaining_priority_targets = self.greedy_tile_mc(
                                                            candidate_tiles, 
                                                            candidate_targets, 
                                                            candidate_targets_range, 
                                                            standard_targets_range, 
                                                            guide_targets_range, 
                                                            ranking_list, 
                                                            n_priority_targets, 
                                                            remaining_priority_targets) 
                
                # Add the n best tiles to the set of selected tiles, and assign mag info
                for nth_best_tile in best_tiles:
                    nth_best_tile.mag_min = mag_range[0]
                    nth_best_tile.mag_max = mag_range[1] 
                    
                    tile_list.append(nth_best_tile) 
                
            # Recalculate the ranking list
            ranking_list = [self.calculate_tile_score(tile) for tile in candidate_tiles]

            # Logging
            logging.info('%d targets, %d standards, %d guides' %
                         (tile_list[-1].count_assigned_targets_science(),
                          tile_list[-1].count_assigned_targets_standard(),
                          tile_list[-1].count_assigned_targets_guide(), ))
            logging.info('Now assigned %d tiles' % (len(tile_list), ))
            logging.info('Priority completeness achieved: {0:1.4f}'.format(
                            (float(n_priority_targets - remaining_priority_targets) \
                            / float(n_priority_targets))) )
            logging.info('Remaining priority targets: {0:d} / {1:d}'.format(
                         remaining_priority_targets, n_priority_targets))
            logging.info('Remaining guides & standards (this mag range): %d, %d' %
                         (len(guide_targets_range), len(standard_targets_range)))
        
            # Two checks to be run here:
            # 1 - Reduce the selection of non-priority targets by setting
            #     disqualify_below_min to False if the max of the ranking list falls below
            #     some threshold (e.g. the score of a single priority target). This moves
            #     the tiling more rapidly towards completion and allows priority targets
            #     not in the priority magnitude range to observed in the next mag bin.
            # 2 - If there are no more legal targets (because of disqualify_below_min 
            #     setting the score to zero when below the minimum number of standards/
            #     guides) cease disqualifying such tiles to head towards completion.
            # Note: by having the two booleans first we can short-circuit the expression
            # in the event they are false, removing the need to sort the ranking list.
            # This structure ensures we only enter either of the two if statements only
            # once, as we flip the booleans after successful use.
            if self.enforce_min_tile_score and max(ranking_list) < self.min_tile_score:
                logging.info('Detected no tiles above min score - relaxing requirements')
                self.enforce_min_tile_score = False
                self.disqualify_below_min = False
                ranking_list = [self.calculate_tile_score(tile) 
                                for tile in candidate_tiles]  
                                
            elif self.disqualify_below_min and max(ranking_list) < 0.05:
                logging.info('Detected no remaining legal tiles - relaxing requirements')
                self.disqualify_below_min = False
                ranking_list = [self.calculate_tile_score(tile) 
                                for tile in candidate_tiles]        
        
        # Print summary of the magnitude range
        cc = (float(n_priority_targets - remaining_priority_targets)
                          / float(n_priority_targets))
        print "Mag range complete with %i tiles, %5.2f %% complete" % (len(tile_list),
                                                                          100*cc)
        if cc < self.completeness_target:
            logging.warning("### WARNING: mag range tiling aborted prior to completeness")
            
        # Reset booleans
        self.disqualify_below_min = self.disqualify_below_min_original                                                             
        self.enforce_min_tile_score = self.enforce_min_tile_score_original  
                                                                          
        return tile_list

    
    #@profile
    def greedy_tile_mag_range(self, candidate_targets, standard_targets, guide_targets, 
                              candidate_tiles, range_ix):
        """Function to perform a greedy sky tiling for a given magnitude range.
    
        Parameters
        ----------
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        standard_targets: list of :class:`TaipanTarget`
            The entire list of standard targets to consider.
        
        guide_targets: list of :class:`TaipanTarget`
            The entire list of guide targets to consider. 
    
        candidate_tiles: list of :class: `TaipanTile`
            The list of filled candidate tiles covering the section of sky observed.
            
        range_ix: int
            The index of the magnitude range to tile.
            
        Returns
        -------
        tile_list: list of :class: `TaipanTile`
            The set of tiles meeting the completeness requirement for the provided set of
            targets.
        """
        # Initialise the tile list for this magnitude range
        tile_list = []
    
        mag_range = self.mag_ranges[range_ix]
    
        print "Tiling mag range %s" % mag_range
        
        # Perform check to see if using priority magnitude ranges
        try:
            mag_range_prioritise = self.mag_ranges_prioritise[range_ix]
        except:
            mag_range_prioritise = None
    
        # Check to see if this is the final magnitude range to be considered
        last_range = not (range_ix < (len(self.mag_ranges) - 1))  
    
        # Determine targets, standards, and guides                                                 
        candidate_targets_range = self.get_targets_mag_range(candidate_targets, mag_range, 
                                                             mag_range_prioritise, 
                                                             last_range)  
        
        standard_targets_range = self.get_standards_mag_range(standard_targets, mag_range)
        
        non_candidate_guide_targets = self.get_guides_mag_range(guide_targets, 
                                                                candidate_targets_range)
    
        logging.info("Mag range: {0:5.1f} {1:5.1f}".format(mag_range[0], mag_range[1]))
        logging.info("Mag range to prioritize: {0:5.1f} {1:5.1f}".format(
                 mag_range_prioritise[0],mag_range_prioritise[1]))
        logging.info("Number of targets in this range: {0:d}".format(
                 len(candidate_targets_range)))
        
        # Generate tiling for the magnitude range
        tile_list = self.greedy_tile_sky(candidate_targets, candidate_targets_range, 
                                         standard_targets_range, 
                                         non_candidate_guide_targets, 
                                         candidate_tiles, mag_range)

        # Now return the priorities to as they were for the remaining targets.
        for target in candidate_targets_range:
            target.priority = target.priority_original
    
        # Consolidate the tiling
        print "Consolidating %i tiles" % len(tile_list),
        start = time.time()
        tile_list = tl.tiling_consolidate(tile_list)
        finish = time.time()
        delta = finish - start
        print ("--> %i tiles, done in %d:%02.1f \n") % (len(tile_list), delta/60, 
                                                        delta % 60.)
      
        logging.info('Mag range: {0:3.1f} to {1:3.1f}, '.format(mag_range_prioritise[0], 
                    mag_range_prioritise[1]))
        logging.info('Total Tiles so far = {0:d}'.format(len(tile_list))) 
    
        return tile_list
    
    
    #@profile
    def generate_tiling(self, candidate_targets, standard_targets, guide_targets):
        """
        Generate a tiling based on the greedy algorithm operating on a set of magnitude 
        ranges sequentially. Within each magnitude range, a complete set of tiles are 
        selected that enables completeness higher than the minimum priority only.

        For each magnitude range, the greedy algorithm works as follows:
    
        - Generate a set of tiles covering the area of interest.
        - Unpick each tile (meaning allocate fibers), allowing targets to be duplicated '
          between tiles.
        - Select the 'best' tile from this set, and add it to the resultant
          tiling.
        - Replace the removed tile in the list (i.e. as you probably haven't yet observed
          all targets in that part of the sky), then re-unpick the tiles in the set which 
          are affected by the removal of the tile - i.e. the tile just replaced and 
          neighbouring tiles.
        - Repeat until no useful tiles remain, or the completeness target is
          reached.
        - Then go on to the next magnitude range until there are no magnitude ranges left.

        Parameters
        ----------
        candidate_targets: list of :class:`TaipanTarget`
            The entire list of candidate targets to consider.
        
        standard_targets: list of :class:`TaipanTarget`
            The entire list of standard targets to consider.
        
        guide_targets: list of :class:`TaipanTarget`
            The entire list of guide targets to consider. 
        
        Returns
        -------
        tile_list: list of :class: `TaipanTile`
            The set of tiles meeting the completeness requirement for the provided set of
            targets.
        
        final_completeness: float
            The target completeness achieved.
        
        candidate_targets: list of :class:`TaipanTarget`
            Any targets from candidate_targets that do not appear in the final tiling_list 
            (i.e. were not assigned to a successful tile).
        """
        # Initialise list to hold all tiling results
        tile_lists = []

        # Push the coordinate limits into standard format
        self.compute_bounds()

        # Generate the SH tiling to cover the region of interest
        candidate_tiles = self.generate_SH_tiling()
        candidate_tiles = [tile for tile in candidate_tiles 
                           if self.is_within_bounds(tile)]

        #candidate_targets_master = candidate_targets[:]
    
        # Initialise some of our counter variables
        no_submitted_targets = len(candidate_targets)
        
        if no_submitted_targets == 0:
            raise ValueError('Attempting to generate a tiling with no targets!')
                
        # Generate a greedy style tiling for each magnitude range 
        for range_ix in xrange(len(self.mag_ranges)):
            tile_list =  self.greedy_tile_mag_range(candidate_targets, standard_targets, 
                                                    guide_targets, candidate_tiles, 
                                                    range_ix)
            tile_lists.append(tile_list)
        
        # Concatenate tiling lists from each range
        tile_list = []
        for mag_range_tiling in tile_lists:
            tile_list.extend(mag_range_tiling)
        
        # Return the tiling, the completeness factor and the remaining targets
        final_completeness = float(no_submitted_targets 
            - len(candidate_targets)) / float(no_submitted_targets)
    
        # Perform a global re-pick if not done during tiling
        if not self.repick_after_complete:
            start = time.time()
            print "Performing global repick...",
        
            for tile in tile_list:
                tile.repick_tile()
            
            finish = time.time()
            delta = finish - start
            print ("done in %d:%02.1f") % (delta/60, delta % 60.)
    
        print "Tiling complete! \n"
    
        return tile_list, final_completeness, candidate_targets 


# ----------------------------------------------------------------------------------------
# External tiling code for multiprocessing
# ----------------------------------------------------------------------------------------
#@profile
def repick_within_radius(best_tile, candidate_tiles, candidate_targets, 
                         candidate_standards, candidate_guides, unpick_settings, 
                         n_radii=2):
    """Function to repick neighbouring tiles after selecting a tile for the final 
    tiling to account for target duplication between tiles.

    Parameters
    ----------
    best_tile: TaipanTile
        The best (highest-ranked distant) tile selected as part of the greedy tiling 
        algorithm.
    
    candidate_tiles: list of :class: `TaipanTile`
        The list of *nearby* filled candidate tiles covering the section of sky observed.   
         
    candidate_targets: list of :class:`TaipanTarget`
        The list of *nearby* candidate targets to consider for repicking.
    
    standard_targets: list of :class:`TaipanTarget`
        The list *nearby* standard targets to consider for repicking.
    
    guide_targets: list of :class:`TaipanTarget`
        The list of *nearby* guide targets to consider for repicking.
        
    unpick_settings: Dict
        Dictionary to store the tile unpick settings of the FWTiler object when passing to
        this function. The multiprocessing environment that this function is used in 
        prevents the use of instance methods, and the dictionary is a convenient wrapper
        to avoid the unnecessary exposure of parameters. Consists of:
        
            overwrite_existing, check_tile_radius, recompute_difficulty, 
            tile_unpick_method, combined_weight, sequential_ordering,rank_supplements,
            repick_after_complete, consider_removed_targets, allow_standard_targets, and
            assign_sky_first
    
    n_radii: int
        The radius out to which neighbouring tiles should be repicked.
        
    Returns
    -------
    candidate_tiles: list of :class: `TaipanTile`
        The now re-unpicked list of *nearby* filled candidate tiles covering the section 
        of sky observed, with duplicates removed from the recently selected best tile.
    """
    # Repick any tiles within n_radii*TILE_RADIUS
    assigned_targets = best_tile.get_assigned_targets_science()

    nearby_candidate_tiles = tp.targets_in_range(best_tile.ra, best_tile.dec, 
                                            candidate_tiles, n_radii*tp.TILE_RADIUS) 

    affected_tiles = list({atile for atile in nearby_candidate_tiles 
                          for t in assigned_targets \
                          if t in atile.get_assigned_targets_science()})
                          
    # Tile order is scrambled for MC runs, so rather than adding the empty replacement 
    # tile by [-1] index as works for the n_cores=0 case, explicitly add any tiles that 
    # have no allocated fibres 
    for tile in candidate_tiles:
        if tile.count_assigned_fibres() == 0:
            affected_tiles.append(tile)
    
    # Pre-generate the KD tree for the nearby standards and guide lists. These lists will
    # remain the same, with consistent ordered, per visit to this function
    standard_tree = precompute_kd_tree(candidate_standards)
    guide_tree = precompute_kd_tree(candidate_guides)
    
    for tile_i, tile in enumerate(affected_tiles):
        # Repick the affected tiles, but making sure to only supply the targets that 
        # actually fall within the tile boundaries - unnecessary processing otherwise
        tile.unpick_tile(tp.targets_in_range(tile.ra, tile.dec, candidate_targets, 
                                             1*tp.TILE_RADIUS), 
                         tp.targets_in_range(tile.ra, tile.dec, candidate_standards, 
                                            1*tp.TILE_RADIUS, tree=standard_tree), 
                         tp.targets_in_range(tile.ra, tile.dec, candidate_guides, 
                                            1*tp.TILE_RADIUS, tree=guide_tree),
                         **unpick_settings)

    # It is possible for a candidate tile to have no assigned fibres at this point, but
    # that should occur only towards the end of a tiling run, or for a sparsely populated
    # field (i.e. the bright magnitude bin). So long as the replacement tiles are properly
    # unpicked, there should be no issue reaching the completion target.
    
    return candidate_tiles


#@profile
def repick_within_radius_pool(input_params):
    """multiprocessing.pool implementation of repick_within_radius.
    
    input_params is tuple of form: 
    (process_i, neighbourhood_tiles, repicked_tiles, fwtiler.unpick_settings)
    
    Where process_i is used to index neighbourhood_tiles, which is of type manager.list()
    and holds the targets, standards, guides, and tiles from the neighbourhood. 
    repicked_tiles is also of type manager.list(), and is used to store the repicked 
    tiles from the neighbourhood.
    
    Parameters
    ----------
    input_params: list
        List of input parameters, described above.
    """
    # Unpack input_params
    process_i = input_params[0]
    neighbourhood_tiles = input_params[1]
    repicked_tiles = input_params[2]
    unpick_settings = input_params[3]
    
    # Unpack neighbourhood_tiles
    best_tile = neighbourhood_tiles[process_i][0]
    candidate_targets = neighbourhood_tiles[process_i][1]
    candidate_standards = neighbourhood_tiles[process_i][2]
    candidate_guides = neighbourhood_tiles[process_i][3]
    candidate_tiles = neighbourhood_tiles[process_i][4]
    
    n_radii=2

    # Repick any tiles within n_radii*TILE_RADIUS
    assigned_targets = best_tile.get_assigned_targets_science()

    nearby_candidate_tiles = tp.targets_in_range(best_tile.ra, best_tile.dec, 
                                            candidate_tiles, n_radii*tp.TILE_RADIUS) 

    affected_tiles = list({atile for atile in nearby_candidate_tiles 
                          for t in assigned_targets \
                          if t in atile.get_assigned_targets_science()})
                      
    # Tile order is scrambled for MC runs, so rather than adding the empty replacement 
    # tile by [-1] index as works for the n_cores=0 case, explicitly add any tiles that 
    # have no allocated fibres 
    for tile in candidate_tiles:
        if tile.count_assigned_fibres() == 0:
            affected_tiles.append(tile)

    # Pre-generate the KD tree for the nearby standards and guide lists. These lists will
    # remain the same, with consistent ordered, per visit to this function
    standard_tree = precompute_kd_tree(candidate_standards)
    guide_tree = precompute_kd_tree(candidate_guides)

    for tile_i, tile in enumerate(affected_tiles):
        # Repick the affected tiles, but making sure to only supply the targets that 
        # actually fall within the tile boundaries - unnecessary processing otherwise
        tile.unpick_tile(tp.targets_in_range(tile.ra, tile.dec, candidate_targets, 
                                             1*tp.TILE_RADIUS), 
                         tp.targets_in_range(tile.ra, tile.dec, candidate_standards, 
                                            1*tp.TILE_RADIUS, tree=standard_tree), 
                         tp.targets_in_range(tile.ra, tile.dec, candidate_guides, 
                                            1*tp.TILE_RADIUS, tree=guide_tree),
                         **unpick_settings)

    # It is possible for a candidate tile to have no assigned fibres at this point, but
    # that should occur only towards the end of a tiling run, or for a sparsely populated
    # field (i.e. the bright magnitude bin). So long as the replacement tiles are properly
    # unpicked, there should be no issue reaching the completion target.
    
    # Add the now repicked tiles to the "results" manager to finish
    repicked_tiles.append(candidate_tiles)


# ----------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------
def is_same_taipan_object(obj_1, obj_2):
    """Function to check the equivalence of two TaipanPoint derived objects. Stand-in for
    the currently not implemented object level equivalence tests.
    
    Parameters
    ----------
    obj_1: TaipanPoint derived object
        The first Taipan object to compare.
    obj_2: TaipanPoint derived object
        The second Taipan object ot compare
        
    Returns
    -------
    equivalent: boolean
        Boolean value indicating whether the two objects are in fact the same.
    """
    equivalent = False
    
    if type(obj_1) == type(obj_2):
        # For TaipanTargets to be considered the same, they must have the same ID, RA, 
        # DEC, and science/standard/guide status
        if (type(obj_1) is tp.TaipanTarget) and (obj_1.idn == obj_2.idn) and \
            (obj_1.science == obj_2.science) and (obj_1.standard == obj_2.standard) and \
            (obj_1.guide == obj_2.guide):
            equivalent = True

    return equivalent


def count_unique_science_targets(tiling):
    """Function to run at the conclusion of the tiling run to detect any duplication in 
    the assigned targets (duplication likely caused by unintended consequences of running
    tiling in parallel. This function does not consider reused standards duplicates.
    
    Parameters
    ----------
    tiling: list of TaipanTile objects
        A list of TaipanTiles (e.g. the results of a tiling run)
        
    Returns
    -------
    unique: int
        The total number of *unique* targets observed within the tile set.
    total: int
        The total number of targets observed within the tile set.
    duplicates: int
        The total number of *duplicate* targets observed within the tile set.                
    """
    uniques = 0
    duplicates = 0
    
    all_targets = []
    
    # Look for duplicates by assuming RA and DEC will be unique
    for tile in tiling:
        for target in tile.get_assigned_targets_science(include_science_standards=False):
            all_targets.append(str(target.ra) + "-" + str(target.dec))           
    
    # Using a set in this fashion is quicker than a for loop
    unique = len(set(all_targets))
    total = len(all_targets)
    duplicates = total-unique
    
    return unique, total, duplicates

    
def precompute_kd_tree(target_list):
    """Precomputes the KD tree for a list of targets.
    
    Parameters
    ----------
    target_list: list of :class:`TaipanTarget`
        The list to compute the tree for.
    
    Returns
    -------
    tree: cKDTree
        The pre-computed KD-tree.
    """
    cart_targets = np.asarray([t.usposn for t in target_list])
    tree = cKDTree(cart_targets, leafsize=tp.BREAKEVEN_KDTREE)
    
    return tree