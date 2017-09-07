"""
"""
import logging
import core as tp
import tiling as tl
import time
import random
import math
import numpy as np
import copy
# import line_profiler
from threading import Thread, Lock
from joblib import Parallel, delayed
import multiprocessing as mp
import functools
from matplotlib.cbook import flatten

class FWTiler(object):
    """FunnelWeb Tiler object encapsulate tiling settings, wrap tiling functions, and
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
                 assign_sky_first=True, n_cores=1):
        """Constructor for FWTiler. Takes as parameters the various settings needed to 
        unpick and rank individual tiles, as well as tile the sky as a whole.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        self._completeness_target = None
        self._ranking_method = None
        self._disqualify_below_min = None
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
        
        # Insert the passed values
        # Doing it like this forces the setter functions to be
        # called, which provides error checking
        self.completeness_target = completeness_target
        self.ranking_method = ranking_method
        self.disqualify_below_min = disqualify_below_min
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
    def tiling_method(self):
        return self._tiling_method

    @tiling_method.setter
    def tiling_method(self, value):
        if value is None: 
            raise Exception('tiling_method may not be blank')
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
    
    # ------------------------------------------------------------------------------------
    # tiling wrapper functions
    # ------------------------------------------------------------------------------------    
    def gen_pa():
        """FWTiler wrapper for tiling.gen_pa(randomise_pa)
        """
        return tl.gen_pa(self.randomise_pa)
    
        
    def compute_bounds():
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
     
        
    def is_within_bounds(tile, compute_bounds_forcoords=True):
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
    
    
    def generate_random_tile():
        """FWTiler wrapper for tiling.generate_random_tile(ra_min=0.0, ra_max=360.0,
                                    dec_min=-90.0, dec_max=90.0, randomise_pa=False)
        
        Returns
        -------
        tile: TaipanTile
            The generated tile.
        """
        return tl.generate_random_tile(self.ra_min, self.ra_max, self.dec_min, 
                                       self.dec_max, self.randomise_pa)
    
    
    def generate_SH_tiling():
        """FWTiler wrapper for tiling.generate_SH_tiling(tiling_file, randomise_seed=True, 
                                    randomise_pa=False)
        
        Returns
        -------
        tile_list: list of TaipanTile objects
            A list of TaipanTiles that have been generated from the Sloane-Harding tiling.
        """
        return tl.generate_SH_tiling(self.tiling_file, self.randomise_SH, 
                                     self.randomise_pa)
        
        
    def tiling_consolidate(tile_list):
        """FWTiler wrapper for tiling.tiling_consolidate(tile_list)
        
        Parameters
        ----------
        tile_list : list of TaipanTile objects
            The list of TaipanTile objects that constitute the tiling.

        Returns
        -------
        consolidated_list: list of TaipanTile objects
            The list of TaipanTile objects representing the consolidation of tile_list. 
            consolidated_list will NOT preserve the ordering in tile_list.
        """
        return tl.tiling_consolidate(tile_list)
        
        
    # ------------------------------------------------------------------------------------
    # TaipanTile wrapper functions
    # ------------------------------------------------------------------------------------   
    def unpick_tile(tile, candidate_targets, standard_targets, guide_targets):
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
        
        
    def calculate_tile_score(tile):
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
    # Attribute handling
    # ------------------------------------------------------------------------------------     
        
# ----------------------------------------------------------------------------------------
# FunnelWeb multicore
# ----------------------------------------------------------------------------------------
def get_targets_mag_range(candidate_targets, mag_range, priorities,  
                          mag_range_prioritise=None, last_range=False):
    """Function to determine the candidate targets for a given magnitude range.
    
    When using mag_range_prioritise to prioritise a subsection of the total magnitude bin,
    if a target is excluded (but would be included in a subsequent bin), it will only be 
    observed if *low-priority*. This is to allow high priority (likely fainter) targets
    to be observed with a longer exposure time in the next magnitude bin (assuming bright-
    faint bin ordering). If the target is high-priority, and outside the priority mag 
    range, but there are no fainter bins, it will be considered.
    
    Parameters
    ----------
    candidate_targets: list of :class:`TaipanTarget`
        The entire list of candidate targets to consider.
        
    mag_range: list of floats
        Upper and lower bounds of the magnitude bin under consideration. Of form [L, U].
    
    mag_range_prioritise: list of floats, optional
        Upper and lower bounds of the priority magnitude range within the magnitude bin
        under consideration. If used, of form [L, U], else defaults to None. 
    
    priorities: list of ints
        The priorities for evaluating high- and low-priority targets. Of form: 
        [priority_normal, prioritise_extra].
    
    last_range: boolean
        Indicates whether the magnitude range is the last bin under consideration, and
        thus whether high priority targets should be considered.
        
    Returns
    -------
    candidate_targets_range: list of :class:`TaipanTarget`
        The candidate targets that satisfy magnitude range and priority requirements.
    """ 
    #Find the candidates in the correct magnitude range. If we are not in the faintest
    #magnitude range, then we have to ignore high priority targets for now, as we'd
    #rather them be observed with a long exposure time
    if mag_range_prioritise:
        if not last_range:
            candidate_targets_range = [t for t in candidate_targets 
                if ( (mag_range_prioritise[1] <= t.mag < mag_range[1]) and #faint
                (t.priority <= priorities[0]) ) or
                ( (mag_range[0] <= t.mag < mag_range_prioritise[1]) )] #bright
            
            # Increase the priority for targets in the priority magnitude range
            for t in candidate_targets_range:
                if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                    t.priority >= priorities[0]):
                    t.priority += priorities[1]
        
        # In faintest bin, consider all targets
        else:
            candidate_targets_range = [t for t in candidate_targets
                if (mag_range[0] <= t.mag < mag_range[1])]
            for t in candidate_targets_range:
                if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                    t.priority == priorities[0]):
                    t.priority += priorities[1] 
                    
        return candidate_targets_range

def get_standards_mag_range(standard_targets, mag_range):
    """Function to select standard stars from within a given magnitude bin.
    
    Parameters
    ----------
    standard_targets: list of :class:`TaipanTarget`
        List of all available standard stars.
    
    mag_range: list of floats
        Upper and lower bounds of the magnitude bin under consideration. Of form [L, U].
        
    Returns
    -------
    standard_targets_range: list of :class:`TaipanTarget`
        The standard targets that satisfy the magnitude range requirement.
    """
    standard_targets_range = [t for t in standard_targets 
            if mag_range[0] <= t.mag < mag_range[1]]
            
    return standard_targets_range


def get_guides_mag_range(guide_targets, candidate_targets_range):
    """Function to select guide stars from within a given magnitude bin.
    
    At present the only criteria for being considered a guide is to *not* be a candidate
    target for the range.
    
    TODO: Consider magnitude of guide stars.
    
    Parameters
    ----------
    guide_targets: list of :class:`TaipanTarget`
        List of all available guide stars.
    
    candidate_targets_range: list of :class:`TaipanTarget`
        The candidate targets that satisfy magnitude range and priority requirements.
        
    Returns
    -------
    non_candidate_guide_targets: list of :class:`TaipanTarget`
        The guide targets that requirements.
    """
    # Find the guides that are not candidate targets only. These have to be copied, 
    # because the same target will be a guide for one field and not a guide for 
    # another field.
    non_candidate_guide_targets = []
    for potential_guide in guide_targets:
        if potential_guide not in candidate_targets_range:
            aguide = copy.copy(potential_guide)
            aguide.guide=True
            #WARNING: We have to set the standard and science flags as well, as this error
            #checking isn't done in core.py
            aguide.standard=False
            aguide.science=False
            non_candidate_guide_targets.append(aguide)
            
    return non_candidate_guide_targets
 
 
def calc_priority_targets(candidate_targets, completeness_priority):
    """Calculates the number of priority targets given the candidate targets and the 
    completeness priority.
    
    Parameters
    ----------
    candidate_targets: list of :class:`TaipanTarget`
        The list of candidate targets to be evaluated based on priority.
    
    completeness_priority: int
        The target priority level to evaluate completion at.
        
    Returns
    -------
    n_priority_targets: int
        The number of targets considered a priority for the given completeness_priority.
    """
    # The number of priority targets is the number of targets above completeness_priority.
    # This includes some targets that are also standards.
    n_priority_targets = 0
    for target in candidate_targets:
        if target.priority >= completeness_priority:
            n_priority_targets += 1
    if n_priority_targets == 0:
        raise ValueError('Require some priority targets in each mag range!')
        
    return n_priority_targets


def select_and_replace_best_tile(tile_list, ranking_list, candidate_tiles, 
                                 candidate_targets, candidate_targets_range, 
                                 completeness_priority, recompute_difficulty, 
                                 randomise_pa, remaining_priority_targets):
    """Function to select the highest ranked tile for the final tiling, and re-generated
    a replacement. 
    
    No return values as only candidate_tiles, candidate_targets, and
    candidate_targets_range are modified, but all are lists so changes will be reflected
    out of scope.
    
    Parameters
    ----------
    
    
    """
    # Find the highest-ranked tile in the candidates_list, and remove it
    tile_i = np.argmax(ranking_list)
    tile_list.append(candidate_tiles.pop(tile_i))
    best_ranking = ranking_list.pop(tile_i)
    logging.info('Tile selected!')
                
    # Record the ra and dec of the candidate for tile re-creation
    best_ra = tile_list[-1].ra
    best_dec = tile_list[-1].dec

    # Strip the now-assigned targets out of the candidate_targets list,
    # then recalculate difficulties for affected remaning targets
    logging.info('Re-computing target list...')
    assigned_targets = tile_list[-1].get_assigned_targets_science()

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
            candidate_targets.pop(candidate_targets.index(target))
            candidate_targets_range.pop(candidate_targets_range.index(target))

            #Count the priority targets we've just assigned in the same way
            #as they were originally counted
            if target.priority >= completeness_priority:
                remaining_priority_targets -= 1
            
            #Change priorities back to normal for targets in our priority magnitude
            #range
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
    if recompute_difficulty:
        logging.info('Re-computing target difficulties...')
        tp.compute_target_difficulties(tp.targets_in_range(
            best_ra, best_dec, candidate_targets_range,
            tp.TILE_RADIUS+tp.FIBRE_EXCLUSION_DIAMETER))

    # Replace the removed tile in candidate_tiles
    candidate_tiles.append(tp.TaipanTile(best_ra, best_dec, pa=gen_pa(randomise_pa)))
    
    logging.info('Assigned tile at %3.1f, %2.1f' % (best_ra, best_dec))
    logging.info('Tile has ranking score %3.1f' % (best_ranking, ))
    
    return remaining_priority_targets


def repick_within_radius(tile_list, candidate_tiles, candidate_targets_range, 
                         standard_targets_range, guide_targets_range, 
                         tile_unpick_settings, n_radii=2):
    """Function to repick neighbouring tiles after selecting a tile for the final tiling
    to account for target duplication between tiles.
    
    Parameters
    ----------
    """
    # Repick any tiles within n_radii*TILE_RADIUS of it, and then add to the ranking_list
    logging.info('Re-picking affected tiles...')
    
    assigned_targets = tile_list[-1].get_assigned_targets_science()

    # This is  a big n_tiles x n_assigned operation - lets make it faster by 
    # considering only the nearby candidate tiles (within 2 * TILE_RADIUS)
    nearby_candidate_tiles = tp.targets_in_range(tile_list[-1].ra, tile_list[-1].dec, 
                                                 candidate_tiles, n_radii*tp.TILE_RADIUS)         
    affected_tiles = list({atile for atile in nearby_candidate_tiles 
                          for t in assigned_targets \
                          if t in atile.get_assigned_targets_science()})
   
    # This won't cause the new tile to be re-picked, so manually add that
    affected_tiles.append(candidate_tiles[-1])
    
    for tile_i, tile in enumerate(affected_tiles):
        burn = tile.unpick_tile(candidate_targets_range, standard_targets_range, 
                                guide_targets_range, **tile_unpick_settings)
                                
        logging.info('Completed %d / %d' % (tile_i, len(affected_tiles)))




def perform_greedy_tiling(candidate_targets, candidate_targets_range,  
                          standard_targets_range, guide_targets_range, candidate_tiles,   
                          ranking_method, disqualify_below_min_range, exp_base, 
                          mag_range, completeness_target, completeness_priority, 
                          recompute_difficulty, randomise_pa, tile_unpick_settings):
    """Function to perform the greedy tiling algorithm given targets, standards, guides,
    and a selection of tiles.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    # Create initial tile unpicks
    # Note that we are *not* updating candidate_targets during this process,
    # as overlap is allowed - instead, we will need to manually update
    # candidate_tiles once we pick the highest-ranked tile
    # Likewise, we don't want the target difficulties to change
    # Therefore, we'll assign the output of the function to a dummy variable                   
    if recompute_difficulty:
        logging.info("Computing difficulties...")
        tp.compute_target_difficulties(candidate_targets_range)
    
    print "Tiling mag range %s; # Targets=%i" % (mag_range, len(candidate_targets_range)),
    
    logging.info('Creating initial tile unpicks...')
    for tile_i, tile in enumerate(candidate_tiles):   
        burn = tile.unpick_tile(candidate_targets_range, standard_targets_range, 
                                guide_targets_range, **tile_unpick_settings)
        logging.info('Created %d / %d tiles' % (tile_i, len(candidate_tiles)))
    
    # Compute initial rankings for all of the tiles
    ranking_list = [tile.calculate_tile_score(method=ranking_method,
                    disqualify_below_min=disqualify_below_min_range, 
                    combined_weight=tile_unpick_settings["combined_weight"], 
                    exp_base=exp_base) for tile in candidate_tiles]
                    
    # Calculate priority targets
    n_priority_targets = calc_priority_targets(candidate_targets_range, 
                                               completeness_priority)
    remaining_priority_targets = n_priority_targets

    # While we are below our completeness criteria AND the highest-ranked tile is not
    # empty, perform the greedy algorithm
    logging.info('Starting greedy/Funnelweb tiling allocation...')        
    tile_list = []
    tile_i = 0
    while ((float(n_priority_targets - remaining_priority_targets) 
           / float(n_priority_targets)) < completeness_target) and \
           (max(ranking_list) > 0.05): 
        # Note: 0.05 is a simple proxy for max > 0
    
        #FIXME: We really want to select the best N tiles here, where N is as large as
        #possible, and each tile is more than 6 tile radii apart. e.g. around the
        #equator, there are 20 such tiles. Each of these N>20 high-ish priority tiles
        #can then be repicked separately, AND the affected tiles within their affected
        #radii can be repicked. 
        #
        #Even better, we pass to this new routine a subset only of the candidate targets 
        #and tiles that may fit within this range. Kind-of like a tree algorithm, we cut
        #the sky down to the relevant part only, and deal with just this part.
        #
        #Pseudocode:
        #
        #Create an empty list of best_tiles
        #Loop over at most n_processors:
        # - Find the best tile that is more than 6 tile radii from all other best tiles.
        # - pop into a new list all tiles within 2 tile radii of this best tile, and
        #   all candidate_targets and candidate_targets_range within 3 tile radii of this
        #   best tile.
        #Loop in e.g. multi-process environment over all n_best lists of
        # [best_tile, nearby_tiles, nearby_candidates, nearby_other_stuff]
        # - Pick the tile and repick all within 2 tile radii
        # - Add to a local (?) list of assigned_targets and assigned_tiles. This is the 
        #   bit that needs testing in multiprocessing (does it have to be "local"?)
        #Loop in the standard environment to put Humpty back together again.
    
        # Select the best ranked tile and replace
        remaining_priority_targets = select_and_replace_best_tile(tile_list, ranking_list, 
                                            candidate_tiles, candidate_targets, 
                                            candidate_targets_range, 
                                            completeness_priority, recompute_difficulty, 
                                            randomise_pa, remaining_priority_targets)
    
        # Add the magnitude range information
        tile_list[-1].mag_min = mag_range[0]
        tile_list[-1].mag_max = mag_range[1]
        
        # Repick all tiles within a given radius of the selected tile
        repick_within_radius(tile_list, candidate_tiles, candidate_targets_range, 
                             standard_targets_range, guide_targets_range, 
                             tile_unpick_settings, n_radii=2)
        
        # Recalculate the ranking list
        ranking_list = [tile.calculate_tile_score(method=ranking_method,
                disqualify_below_min=disqualify_below_min_range, 
                combined_weight=tile_unpick_settings["combined_weight"],
                exp_base=exp_base) for tile in candidate_tiles]

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
        
        # If the max of the ranking_list is now 0, try switching off  the disqualify flag
        if max(ranking_list) < 0.05 and disqualify_below_min_range:
            logging.info('Detected no remaining legal tiles - relaxing requirements')
            disqualify_below_min_range = False
            ranking_list = [tile.calculate_tile_score(
                            method=ranking_method, 
                            combined_weight=tile_unpick_settings["combined_weight"],
                            exp_base=exp_base, 
                            disqualify_below_min=disqualify_below_min_range) 
                            for tile in candidate_tiles]
                            
    return candidate_targets, candidate_targets_range, tile_list


def greedy_tile_mag_range(candidate_targets, standard_targets, guide_targets, 
                          candidate_tiles, ranking_method, disqualify_below_min_range, 
                          exp_base, mag_ranges, completeness_target, 
                          completeness_priority, recompute_difficulty, randomise_pa, 
                          tile_unpick_settings, mag_ranges_prioritise, range_ix, 
                          priority_normal, prioritise_extra):
    """Function to perform a greedy sky tiling for a given magnitude range.
    
    Single parameter output in the form of a list enables for parallel computation of 
    differing magnitude ranges using the joblib library.
    
    TODO: Clean up input list and dependencies, current form is minimally working version
    for initial parallel implementation.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    # Initialise the tile list
    tile_list = []
    
    mag_range = mag_ranges[range_ix]
    
    # Perform check to see if using priority magnitude ranges
    try:
        mag_range_prioritise = mag_ranges_prioritise[range_ix]
    except:
        mag_range_prioritise = None
    
    # Check to see if this is the final magnitude range to be considered
    last_range = not (range_ix < (len(mag_ranges) - 1))  
    
    # Determine targets, standards, and guides                                                   
    candidate_targets_range = get_targets_mag_range(candidate_targets, mag_range, 
                                                    [priority_normal,
                                                    prioritise_extra],
                                                    mag_range_prioritise, last_range)  
    
    standard_targets_range = get_standards_mag_range(standard_targets, mag_range)
    
    non_candidate_guide_targets = get_guides_mag_range(guide_targets, 
                                                       candidate_targets_range)
    
    logging.info("Mag range: {0:5.1f} {1:5.1f}".format(mag_range[0], mag_range[1]))
    logging.info("Mag range to prioritize: {0:5.1f} {1:5.1f}".format(
             mag_range_prioritise[0],mag_range_prioritise[1]))
    logging.info("Number of targets in this range: {0:d}".format(
             len(candidate_targets_range)))
    
    # Generate tiling for the magnitude range
    candidate_targets, candidate_targets_range, tile_list = perform_greedy_tiling(
            candidate_targets, candidate_targets_range, standard_targets_range, 
            non_candidate_guide_targets, candidate_tiles, ranking_method, 
            disqualify_below_min_range, exp_base, mag_range, completeness_target, 
            completeness_priority, recompute_difficulty, randomise_pa, 
            tile_unpick_settings)
      
    # Now return the priorities to as they were for the remaining targets.
    for target in candidate_targets_range:
        target.priority = target.priority_original
    
    # Consolidate the tiling
    tile_list = tiling_consolidate(tile_list)

    print "......Done"
      
    logging.info('For mag range: {0:3.1f} to {1:3.1f}, '.format(mag_range_prioritise[0], 
        mag_range_prioritise[1]))
    logging.info('Total Tiles so far = {0:d}'.format(len(tile_list))) 
    
    return [candidate_targets, candidate_targets_range, tile_list]
    
    

def generate_tiling_funnelweb_mp(candidate_targets, standard_targets,
                                 guide_targets, completeness_target = 1.0,
                                 ranking_method='priority-expsum',
                                 disqualify_below_min=True, tiling_method='SH', 
                                 randomise_pa=True, randomise_SH=True, 
                                 tiling_file='ipack.3.8192.txt', ra_min=0.0, ra_max=360.0, 
                                 dec_min=-90.0, dec_max=90.0,
                                 mag_ranges_prioritise=[[5,7],[7,8],[9,10],[11,12]],
                                 prioritise_extra=2, priority_normal=2,
                                 mag_ranges=[[5,8],[7,10],[9,12],[11,14]],
                                 tiling_set_size=1000,
                                 tile_unpick_method='combined_weighted',
                                 combined_weight=4.0, sequential_ordering=(2, 1), 
                                 rank_supplements=False, repick_after_complete=True, 
                                 exp_base=3.0, recompute_difficulty=True, n_cores=1):
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
      all targets in that part of the sky), then re-unpick the tiles in the set which are
      affected by the removal of the tile - i.e. the tile just replaced and 
      neighbouring tiles.
    - Repeat until no useful tiles remain, or the completeness target is
      reached.
    - Then go on to the next magnitude range until there are no magnitude ranges left.


    Parameters
    ----------
    candidate_targets, standard_targets, guide_targets : 
        The lists of science,
        standard and guide targets to consider, respectively. Should be lists
        of TaipanTarget objects.
        
    completeness_target : 
        A float in the range (0, 1] denoting the science
        target completeness to stop at. Defaults to 1.0 (full completeness).
        
    ranking_method : 
        The scheme to use for ranking the tiles. See the
        documentation for TaipanTile.calculate_tile_score for details.
        
    tiling_method : 
        The method by which to generate a tiling set. Currently,
        only 'SH' (Sloane-Harding tiling centres) are available.
        
    randomise_pa : 
        Optional Boolean, denoting whether to randomise the pa of
        seed tiles or not. Defaults to True.
        
    randomise_SH : 
        Optional Boolean, denoting whether or not to randomise the
        RA of the 'seed' of the SH tiling. Defaults to True.
        
    tiling_file : 
        The SH tiling file to use for generating tiling centres.
        Defaults to 'ipack.3.8192.txt'.
        
    ra_min, ra_max, dec_min, dec_max : 
        The RA and Dec bounds of the region to
        be considered, in decimal degrees. To have an RA range spanning across 
        0 deg RA, either use a negative value for ra_min, or give an ra_min >
        ra_max.
        
    mag_ranges :    
        The magnitude ranges for each set of tiles.
        
    mag_ranges_prioritise :
        The magnitude ranges to add extra priority to within each set
        of tiles.
        
    prioritise_extra :
        The additional priority to add within each of the mag_range_prioritise
    
    priority_normal :
        The standard priority level. Completeness is assessed at this priority level
        for stars in the priority magnitude range.
                
    tiling_set_size : 
        Not relevant at the current time.
        
    tile_unpick_method : 
        The scheme to be used for unpicking tiles. Defaults to
        'sequential'. See the documentation for TaipanTile.unpick_tile for 
        details.
        
    combined_weight, sequential_ordering : 
        Additional arguments to be used in
        the tile unpicking process. See the documentation for 
        TaipanTile.unpick_tile for details.
        
    rank_supplements : 
        Optional Boolean value, denoting whether to attempt to
        assign guides/standards in priority order. Defaults to False.
        
    repick_after_complete : 
        Boolean value, denoting whether to repick each tile
        after unpicking. Defaults to True.
        
    recompute_difficulty : 
        Boolean value, denoting whether to recompute target
        difficulties after a tile is moved to the results list. For funnelWeb,
        it also means recompute for each mag range. Defaults to
        True.
        
    exp_base : float, optional
        For priority-expsum, this is the base for the exponent (default 3.0)

    Returns
    -------
    tile_list : 
        The list of tiles making up the tiling.
        
    final_completeness : 
        The target completeness achieved.
        
    candidate_targets : 
        Any targets from candidate_targets that do not
        appear in the final tiling_list (i.e. were not assigned to a successful
        tile).
    """
    
    completeness_priority = priority_normal + prioritise_extra
    
    tile_lists = []

    # Input checking
    TILING_METHODS = ['SH',]     # Sloane-Harding

    if tiling_method not in TILING_METHODS:
        raise ValueError('tiling_method must be one of %s' % str(TILING_METHODS))

    tiling_set_size = int(tiling_set_size)
    if tiling_set_size <= 0:
        raise ValueError('tiling_set_size must be > 0')

    if completeness_target <= 0. or completeness_target > 1:
        raise ValueError('completeness_target must be in the range (0, 1]')
    
    # Dictionary of tiling settings
    # TODO: Pass this in separately, rather than defining here
    tile_unpick_settings = {"overwrite_existing":True, 
                            "check_tile_radius":True, 
                            "recompute_difficulty":False, 
                            "method":tile_unpick_method,
                            "combined_weight":combined_weight,
                            "sequential_ordering":sequential_ordering,
                            "rank_supplements":rank_supplements, 
                            "repick_after_complete":repick_after_complete,
                            "consider_removed_targets":False, 
                            "allow_standard_targets":True}
    
    # Push the coordinate limits into standard format
    ra_min, ra_max, dec_min, dec_max = compute_bounds(ra_min, ra_max, dec_min, dec_max)

    # Generate the SH tiling to cover the region of interest
    candidate_tiles = generate_SH_tiling(tiling_file, randomise_seed=randomise_SH, 
                                         randomise_pa=randomise_pa)
    candidate_tiles = [t for t in candidate_tiles
        if is_within_bounds(t, ra_min, ra_max, dec_min, dec_max)]

    candidate_targets_master = candidate_targets[:]
    #candidate_targets = mp.Manager().list(candidate_targets_master)
    
    # Initialise some of our counter variables
    no_submitted_targets = len(candidate_targets_master)
    if no_submitted_targets == 0:
        raise ValueError('Attempting to generate a tiling with no targets!')
    
    # Generate a greedy style tiling for each magnitude range    
    results = Parallel(n_jobs=n_cores, backend="multiprocessing")(
                delayed(greedy_tile_mag_range)(candidate_targets[:], standard_targets,
                                               guide_targets, 
                                               candidate_tiles, ranking_method, 
                                               disqualify_below_min, 
                                               exp_base, mag_ranges, 
                                               completeness_target, 
                                               completeness_priority, 
                                               recompute_difficulty, randomise_pa, 
                                               tile_unpick_settings, 
                                               mag_ranges_prioritise, range_ix,
                                               priority_normal, prioritise_extra)
                                        for range_ix, mag_range in enumerate(mag_ranges))  
        
    # Join everything back together, where results are of form:
    # [candidate_targets, candidate_targets_range, tile_list]
    # TODO: Combine separate versions of candidate_targets after each magnitude range 
    # returns, or create multiprocessing safe version
    for range_ix in xrange(len(mag_ranges)):
        tile_lists.append(results[range_ix][2])

    # Concatenate tiling lists from each range
    tile_list = []
    for mag_range_tiling in tile_lists:
        tile_list.extend(mag_range_tiling)

    # Return the tiling, the completeness factor and the remaining targets
    final_completeness = float(no_submitted_targets 
        - len(candidate_targets)) / float(no_submitted_targets)
    
    # Perform a global re-pick if not done during tiling
    if not repick_after_complete:
        start = time.time()
        print "Performing global repick...",
        
        for t in tile_list:
            t.repick_tile()
            
        finish = time.time()
        delta = finish - start
        print ("done in %d:%02.1f") % (delta/60, delta % 60.)
    
    print "Tiling complete! \n"
    
    return tile_list, final_completeness, candidate_targets 


# ----------------------------------------------------------------------------------------
# Legacy Tiling Code
# ----------------------------------------------------------------------------------------
    
#Uncomment the following line for FunnelWeb line_profile.
#kernprof -l funnelweb_generate_tiling.py
#python -m line_profiler funnelweb_generate_tiling.py.lprof
#@profile
def generate_tiling_funnelweb_legacy(candidate_targets, standard_targets,
                                     guide_targets,
                                     completeness_target = 1.0,
                                     ranking_method='priority-expsum',
                                     disqualify_below_min=True,
                                     tiling_method='SH', randomise_pa=True,
                                     randomise_SH=True, tiling_file='ipack.3.8192.txt',
                                     ra_min=0.0, ra_max=360.0, dec_min=-90.0,
                                     dec_max=90.0,
                                     mag_ranges_prioritise=[[5,7],[7,8],[9,10],[11,12]],
                                     prioritise_extra=2,
                                     priority_normal=2,
                                     mag_ranges=[[5,8],[7,10],[9,12],[11,14]],
                                     tiling_set_size=1000,
                                     tile_unpick_method='combined_weighted',
                                     combined_weight=4.0,
                                     sequential_ordering=(2, 1), rank_supplements=False,
                                     repick_after_complete=True, exp_base=3.0,
                                     recompute_difficulty=True, nthreads=0):
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
      all targets in that part of the sky), then re-unpick the tiles in the set which are
      affected by the removal of the tile - i.e. the tile just replaced and 
      neighbouring tiles.
    - Repeat until no useful tiles remain, or the completeness target is
      reached.
    - Then go on to the next magnitude range until there are no magnitude ranges left.


    Parameters
    ----------
    candidate_targets, standard_targets, guide_targets : 
        The lists of science,
        standard and guide targets to consider, respectively. Should be lists
        of TaipanTarget objects.
        
    completeness_target : 
        A float in the range (0, 1] denoting the science
        target completeness to stop at. Defaults to 1.0 (full completeness).
        
    ranking_method : 
        The scheme to use for ranking the tiles. See the
        documentation for TaipanTile.calculate_tile_score for details.
        
    tiling_method : 
        The method by which to generate a tiling set. Currently,
        only 'SH' (Sloane-Harding tiling centres) are available.
        
    randomise_pa : 
        Optional Boolean, denoting whether to randomise the pa of
        seed tiles or not. Defaults to True.
        
    randomise_SH : 
        Optional Boolean, denoting whether or not to randomise the
        RA of the 'seed' of the SH tiling. Defaults to True.
        
    tiling_file : 
        The SH tiling file to use for generating tiling centres.
        Defaults to 'ipack.3.8192.txt'.
        
    ra_min, ra_max, dec_min, dec_max : 
        The RA and Dec bounds of the region to
        be considered, in decimal degrees. To have an RA range spanning across 
        0 deg RA, either use a negative value for ra_min, or give an ra_min >
        ra_max.
        
    mag_ranges :    
        The magnitude ranges for each set of tiles.
        
    mag_ranges_prioritise :
        The magnitude ranges to add extra priority to within each set
        of tiles.
        
    prioritise_extra :
        The additional priority to add within each of the mag_range_prioritise
    
    priority_normal :
        The standard priority level. Completeness is assessed at this priority level
        for stars in the priority magnitude range.
                
    tiling_set_size : 
        Not relevant at the current time.
        
    tile_unpick_method : 
        The scheme to be used for unpicking tiles. Defaults to
        'sequential'. See the documentation for TaipanTile.unpick_tile for 
        details.
        
    combined_weight, sequential_ordering : 
        Additional arguments to be used in
        the tile unpicking process. See the documentation for 
        TaipanTile.unpick_tile for details.
        
    rank_supplements : 
        Optional Boolean value, denoting whether to attempt to
        assign guides/standards in priority order. Defaults to False.
        
    repick_after_complete : 
        Boolean value, denoting whether to repick each tile
        after unpicking. Defaults to True.
        
    recompute_difficulty : 
        Boolean value, denoting whether to recompute target
        difficulties after a tile is moved to the results list. For funnelWeb,
        it also means recompute for each mag range. Defaults to
        True.
        
    exp_base : float, optional
        For priority-expsum, this is the base for the exponent (default 3.0)

    Returns
    -------
    tile_list : 
        The list of tiles making up the tiling.
        
    final_completeness : 
        The target completeness achieved.
        
    candidate_targets : 
        Any targets from candidate_targets that do not
        appear in the final tiling_list (i.e. were not assigned to a successful
        tile).
    """
    
    completeness_priority = priority_normal + prioritise_extra
    
    tile_lists = []

    # Input checking
    TILING_METHODS = [
        'SH',               # Sloane-Harding
    ]
    if tiling_method not in TILING_METHODS:
        raise ValueError('tiling_method must be one of %s' 
            % str(TILING_METHODS))

    tiling_set_size = int(tiling_set_size)
    if tiling_set_size <= 0:
        raise ValueError('tiling_set_size must be > 0')

    if completeness_target <= 0. or completeness_target > 1:
        raise ValueError('completeness_target must be in the range (0, 1]')

    # Push the coordinate limits into standard format
    ra_min, ra_max, dec_min, dec_max = compute_bounds(ra_min, ra_max,
        dec_min, dec_max)

    # Generate the SH tiling to cover the region of interest
    candidate_tiles = generate_SH_tiling(tiling_file, 
        randomise_seed=randomise_SH, randomise_pa=randomise_pa)
    candidate_tiles = [t for t in candidate_tiles
        if is_within_bounds(t, ra_min, ra_max, dec_min, dec_max)]

    candidate_targets_master = candidate_targets[:]
    # Initialise some of our counter variables
    no_submitted_targets = len(candidate_targets_master)
    if no_submitted_targets == 0:
        raise ValueError('Attempting to generate a tiling with no targets!')
    
    #Loop over magnitude ranges.
    disqualify_below_min_range = disqualify_below_min
    for range_ix, mag_range in enumerate(mag_ranges):
        tile_list = []
                                                           
        try:
            mag_range_prioritise = mag_ranges_prioritise[range_ix]
        except:
            mag_range_prioritise = None
            
        #Find the candidates in the correct magnitude range. If we are not in the faintest
        #magnitude range, then we have to ignore high priority targets for now, as we'd
        #rather them be observed with a long exposure time
        if mag_range_prioritise:
            if range_ix < len(mag_ranges)-1:
                candidate_targets_range = [t for t in candidate_targets 
                    if ( (mag_range_prioritise[1] <= t.mag < mag_range[1]) and #faint
                    (t.priority <= priority_normal) ) or
                    ( (mag_range[0] <= t.mag < mag_range_prioritise[1]) )] #bright
                
                # Increase the priority for targets in the priority magnitude range
                for t in candidate_targets_range:
                    if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                        t.priority >= priority_normal):
                        t.priority += prioritise_extra
            
            # In faintest bin, consider all targets
            else:
                candidate_targets_range = [t for t in candidate_targets
                    if (mag_range[0] <= t.mag < mag_range[1])]
                for t in candidate_targets_range:
                    if ( (mag_range_prioritise[0] <= t.mag < mag_range_prioritise[1]) and
                        t.priority == priority_normal):
                        t.priority += prioritise_extra                
        
        # Keep a record of what was initially in candidate_targets_range
        candidate_targets_range_orig = candidate_targets_range[:]
        
        #Also find the standards in the correct magnitude range.
        standard_targets_range = [t for t in standard_targets 
            if mag_range[0] <= t.mag < mag_range[1]]
        
        logging.info("Mag range: {0:5.1f} {1:5.1f}".format(mag_range[0],
                                                           mag_range[1]))
        logging.info("Mag range to prioritize: {0:5.1f} {1:5.1f}".format(mag_range_prioritise[0],
                                                           mag_range_prioritise[1]))
        logging.info("Number of targets in this range: {0:d}".format(len(candidate_targets_range)))
                            
        #Find the guides that are not candidate targets only. These have to be copied, 
        #because the same target will be a guide for one field and not a guide for 
        #another field.
        non_candidate_guide_targets = []
        for potential_guide in guide_targets:
            if potential_guide not in candidate_targets_range:
                aguide = copy.copy(potential_guide)
                aguide.guide=True
                #WARNING: We have to set the standard and science flags as well, as this error
                #checking isn't done in core.py
                aguide.standard=False
                aguide.science=False
                non_candidate_guide_targets.append(aguide)
        
        if recompute_difficulty:
            logging.info("Computing difficulties...")
            tp.compute_target_difficulties(candidate_targets_range)

        # Unpick ALL of these tiles
        # Note that we are *not* updating candidate_targets during this process,
        # as overlap is allowed - instead, we will need to manually update
        # candidate_tiles once we pick the highest-ranked tile
        # Likewise, we don't want the target difficulties to change
        # Therefore, we'll assign the output of the function to a dummy variable
        logging.info('Creating initial tile unpicks...')
        i = 0
        if nthreads>0:
            logging.info('Creating {0:d} tiles in separate threads'.format(nthreads))
            output_lock = Lock() #Not yet used!!!
            threads = [None for i in range(nthreads)]
            for t_ix, tile in enumerate(candidate_tiles):
                threads[t_ix % nthreads] = Thread(target=tile.unpick_tile, \
                    args=(candidate_targets_range, standard_targets_range, 
                    non_candidate_guide_targets), \
                    kwargs={'overwrite_existing':True, 'check_tile_radius':True,
                    'recompute_difficulty':False,
                    'method':tile_unpick_method, 'combined_weight':combined_weight,
                    'sequential_ordering':sequential_ordering,
                    'rank_supplements':rank_supplements, 
                    'repick_after_complete':repick_after_complete,
                    'consider_removed_targets':False, 'allow_standard_targets':True})
                threads[t_ix % nthreads].start()
                if (t_ix % nthreads) == (nthreads-1):
                    for athread in threads:
                        athread.join()
            #Join any left-over threads
            for athread in threads:
                athread.join()
        else:         
            for tile in candidate_tiles:   
                burn = tile.unpick_tile(candidate_targets_range, standard_targets_range, 
                    non_candidate_guide_targets,
                    overwrite_existing=True, check_tile_radius=True,
                    recompute_difficulty=False,
                    method=tile_unpick_method, combined_weight=combined_weight,
                    sequential_ordering=sequential_ordering,
                    rank_supplements=rank_supplements, 
                    repick_after_complete=repick_after_complete,
                    consider_removed_targets=False, allow_standard_targets=True)
                i += 1
                logging.info('Created %d / %d tiles' % (i, len(candidate_tiles)))

        # Compute initial rankings for all of the tiles
        ranking_list = [tile.calculate_tile_score(method=ranking_method,
            disqualify_below_min=disqualify_below_min_range, combined_weight=combined_weight,
            exp_base=exp_base) for tile in candidate_tiles]
        # print ranking_list

        #The number of priority targets is the number of targets above
        #completeness_priority. This includes some targets that are also 
        #standards.
        n_priority_targets = 0
        for t in candidate_targets_range:
            if t.priority >= completeness_priority:
                n_priority_targets += 1
        if n_priority_targets == 0:
            raise ValueError('Require some priority targets in each mag range!')
        remaining_priority_targets = n_priority_targets


        # While we are below our completeness criteria AND the
        # highest-ranked tile
        # is not empty, perform the greedy algorithm
        logging.info('Starting greedy/Funnelweb tiling allocation...')        
        #PARALLEL - the following loop could copy tile_list, and run many versions of
        #this together. 
        i = 0
        while ((float(n_priority_targets - remaining_priority_targets) 
            / float(n_priority_targets)) < completeness_target) and (
            max(ranking_list) > 0.05): # !!! Warning: 0.05 is hardwirded here
            # - what does it mean??? It a simple proxy for max > 0

            # Find the highest-ranked tile in the candidates_list, and remove it
            i = np.argmax(ranking_list)
            tile_list.append(candidate_tiles.pop(i))
            best_ranking = ranking_list.pop(i)
            logging.info('Tile selected!')
                        
            # Record the ra and dec of the candidate for tile re-creation
            best_ra = tile_list[-1].ra
            best_dec = tile_list[-1].dec

            # Strip the now-assigned targets out of the candidate_targets list,
            # then recalculate difficulties for affected remaning targets
            logging.info('Re-computing target list...')
            assigned_targets = tile_list[-1].get_assigned_targets_science()

            # print assigned_targets
            before_targets_len = len(candidate_targets_range)
            reobserved_standards = []
            for t in assigned_targets:
                # Note: when candidate_targets contains stars with higher than normal 
                # *initial* priorities, it is possible for a star to be overlooked for
                # consideration as a science target (observations being preferred in a 
                # fainter magnitude bin for "high-priority" targets), but still 
                # considered for selection as a standard target. As such is important
                # to use candidate_targets_range, rather than simply candidate_targets 
                # when dealing with assigned targets.
                if t in candidate_targets_range:
                    candidate_targets.pop(candidate_targets.index(t))
                    candidate_targets_range.pop(candidate_targets_range.index(t))
  
                    #Count the priority targets we've just assigned in the same way
                    #as they were originally counted
                    if t.priority >= completeness_priority:
                        remaining_priority_targets -= 1
                    
                    #Change priorities back to normal for targets in our priority magnitude
                    #range
                    t.priority = t.priority_original
                elif t.standard:
                    reobserved_standards.append(t)
                    logging.info('Re-allocating standard ' + str(t.idn) + ' that is also a science target.')
                else:
                    logging.warning('### WARNING: Assigned a target that is neigher a candidate target nor a standard!')

            if len(set(assigned_targets)) != len(assigned_targets):
                logging.warning('### WARNING: target duplication detected')
            if len(candidate_targets_range) != before_targets_len - len(assigned_targets) + len(reobserved_standards):
                logging.warning('### WARNING: Discrepancy found '
                                'in target list reduction')
                logging.warning('Best tile had %d targets; '
                                'only %d removed from list' %
                                (len(assigned_targets),
                                 before_targets_len - len(candidate_targets)))
            if recompute_difficulty:
                logging.info('Re-computing target difficulties...')
                tp.compute_target_difficulties(tp.targets_in_range(
                    best_ra, best_dec, candidate_targets_range,
                    tp.TILE_RADIUS+tp.FIBRE_EXCLUSION_DIAMETER))
            # print 'e : %d' % len(candidate_targets)

            # Replace the removed tile in candidate_tiles, repick any tiles
            # within 2 * TILE_RADIUS of it, and then add to the ranking_list
            candidate_tiles.append(tp.TaipanTile(best_ra, best_dec, pa=gen_pa(
                randomise_pa)))
            j = 0
            logging.info('Re-picking affected tiles...')
            # print 'f : %d' % len(candidate_targets)
            # This is  a big n_tiles x n_assigned operation - lets make it faster by 
            # considering only the nearby candidate tiles (within 2 * TILE_RADIUS)
            nearby_candidate_tiles = \
                tp.targets_in_range(tile_list[-1].ra, tile_list[-1].dec, candidate_tiles, 2*tp.TILE_RADIUS)         
            affected_tiles = list({atile for atile in nearby_candidate_tiles for t in assigned_targets \
                                    if t in atile.get_assigned_targets_science()})
           
            # This won't cause the new tile to be re-picked,
            # so manually add that
            affected_tiles.append(candidate_tiles[-1])
            for tile in affected_tiles:
                # print 'inter: %d' % len(candidate_targets)
                burn = tile.unpick_tile(candidate_targets_range, standard_targets_range, 
                    non_candidate_guide_targets,
                    overwrite_existing=True, check_tile_radius=True,
                    recompute_difficulty=False,
                    method=tile_unpick_method, combined_weight=combined_weight,
                    sequential_ordering=sequential_ordering,
                    rank_supplements=rank_supplements, 
                    repick_after_complete=repick_after_complete,
                    consider_removed_targets=False, allow_standard_targets=True)
                j += 1
                logging.info('Completed %d / %d' % (j, len(affected_tiles)))
            # print 'g : %d' % len(candidate_targets)
            ranking_list = [tile.calculate_tile_score(method=ranking_method,
                disqualify_below_min=disqualify_below_min_range, combined_weight=combined_weight,
                exp_base=exp_base) for tile in candidate_tiles]
            # print ranking_list
            # print [len(t.get_assigned_targets_science()) for t in candidate_tiles]

            logging.info('Assigned tile at %3.1f, %2.1f' % (best_ra, best_dec))
            logging.info('Tile has ranking score %3.1f' % (best_ranking, ))
            logging.info('%d targets, %d standards, %d guides' %
                         (tile_list[-1].count_assigned_targets_science(),
                          tile_list[-1].count_assigned_targets_standard(),
                          tile_list[-1].count_assigned_targets_guide(), ))
            logging.info('Now assigned %d tiles' % (len(tile_list), ))
            logging.info('Priority completeness achieved: {0:1.4f}'.format(
                            (float(n_priority_targets - remaining_priority_targets) \
                            / float(n_priority_targets))) )
            logging.info('Remaining priority targets: {0:d} / {1:d}'.format(remaining_priority_targets, n_priority_targets))
            logging.info('Remaining guides & standards (this mag range): %d, %d' %
                         (len(non_candidate_guide_targets), len(standard_targets_range)))
                
            # Add the magnitude range information
            tile_list[-1].mag_min = mag_range[0]
            tile_list[-1].mag_max = mag_range[1]

            # If the max of the ranking_list is now 0, try switching off 
            # the disqualify flag
            if max(ranking_list) < 0.05 and disqualify_below_min_range:
                logging.info('Detected no remaining legal tiles - '
                             'relaxing requirements')
                disqualify_below_min_range = False
                ranking_list = [tile.calculate_tile_score(
                    method=ranking_method, combined_weight=combined_weight,
                    exp_base=exp_base, disqualify_below_min=disqualify_below_min_range) 
                    for tile in candidate_tiles]
                # print ranking_list
                
            #ZZZ Debugging plots
            if (False): #mag_range[1] > 13:
                cp = np.asarray([t.priority for t in candidate_targets])
                ptarg = np.where(cp>=5)[0]
                cras = np.asarray([t.ra for t in candidate_targets])
                cdecs = np.asarray([t.dec for t in candidate_targets])
                plt.clf()
                plt.plot(cras, cdecs, 'b.')
                plt.plot(cras[ptarg], cdecs[ptarg], 'gx')
                for tl in candidate_tiles: plt.plot(tl.ra, tl.dec, 'ro')
                plt.pause(0.001)
                #import pdb; pdb.set_trace()
                
        # Now return the priorities to as they were for the remaining targets.
        for t in candidate_targets_range:
            t.priority = t.priority_original
        #Log where we're up to:
        logging.info('** For mag range: {0:3.1f} to {1:3.1f}, '.format(mag_range_prioritise[0], mag_range_prioritise[1]))
        logging.info('Total Tiles so far = {0:d}'.format(len(tile_list)))

        # Consolidate the tiling. For FunnelWeb, we only do this for separate magnitude ranges.
        #!!! This doesn't seem to do much.
        tile_list = tiling_consolidate(tile_list)
        # print ranking_list
        tile_lists.append(tile_list)
    
    #Put all tiles in one big list now.
    tile_list=[]
    for l in tile_lists:
        tile_list.extend(l)

    # Return the tiling, the completeness factor and the remaining targets
    final_completeness = float(no_submitted_targets 
        - len(candidate_targets)) / float(no_submitted_targets)

    if not repick_after_complete:
        # Do a global re-pick, given we didn't do it on the fly
        for t in tile_list:
            t.repick_tile()

    return tile_list, final_completeness, candidate_targets