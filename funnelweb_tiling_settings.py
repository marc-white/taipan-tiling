"""File to easily contain parameters to tile the sky with FunnelWeb.

Modify parameters as required - this file is duplicated at the conclusion of 
the run for documentation purposes, along with a pickle of the results.

n_cores controls whether the code runs runs a multiprocessing implementation or
not, with the specific implementation determined by the "backend" parameter.
    n_cores = 0 --> single core (serial) implementation
    n_cores = 1 --> "single core" multiprocessing implementation
    n_cores > 1 --> multi core (parallel) implementation

Acceptable values for "backend" are currently:
    1 - "threading" --> joblib library
    2 - "multiprocessing" --> joblib library
    3 - "pool" --> multiprocessing library
"""
import numpy as np
import taipan.core as tp
from collections import OrderedDict

# Input catalogue and priority list
input_catalogue = ("/priv/mulga1/arains/Catalogues/fw_input_catalogue.fits")
input_priorities = ("/priv/mulga1/arains/Catalogues/fw_priorities.fits")
sky_catalogue = ("/priv/mulga1/arains/Catalogues/"
                 "skyfibers_v17_gaia_ucac4_final.fits")

# Ordered dictionary, so that format is always the same when writing to a file 

# Dictionary of *all* parameters required to construct taipan.fwtiling.FWTiler
# e.g. fwtiler = FWTiler(**tiler_input)
tiler_input = OrderedDict([("completeness_targets", [0.99, 0.99, 0.99, 0.10]),
                           ("ranking_method", "priority-expsum"),
                           ("disqualify_below_min", True),
                           ("tiling_method", "SH"),
                           ("randomise_pa", False),
                           ("randomise_SH", False),
                           ("tiling_file", "ipack.3.4112.txt"),
                           ("ra_min", 0),
                           ("ra_max", 360),
                           ("dec_min", -90),
                           ("dec_max", 0),
                           ("mag_ranges", [[5,8],[7,10],[9,12],[11,14]]),
                           ("mag_ranges_prioritise", 
                           [[5,7],[7,9],[9,11],[11,12]]),
                           ("priority_normal", 2),
                           ("prioritise_extra", 2),
                           ("tile_unpick_method", "sequential"),
                           ("combined_weight", 4.0),
                           ("sequential_ordering", (2,1)),
                           ("rank_supplements", False),
                           ("repick_after_complete", False),
                           ("exp_base", 3.0),
                           ("recompute_difficulty", True),
                           ("overwrite_existing", True),
                           ("check_tile_radius", True),
                           ("consider_removed_targets", False),
                           ("allow_standard_targets", True),
                           ("assign_sky_first", True),
                           ("n_cores", 0),
                           ("backend", "multiprocessing"),
                           ("enforce_min_tile_score", True),
                           ("assign_sky_fibres", True)])
 
# Dictionary of additional parameters not required by FWTiler            
script_settings = OrderedDict([("input_catalogue", input_catalogue),
                               ("input_priorities", input_priorities),
                               ("sky_catalogue", sky_catalogue),
                               ("tab_type", "gaia"),
                               ("gal_lat_limit", 0),
                               ("standard_frac", 0.1),
                               ("guide_range", [8,10.5]),
                               ("TARGET_PER_TILE", 139),
                               ("STANDARDS_PER_TILE", 4),
                               ("STANDARDS_PER_TILE_MIN", 3),
                               ("GUIDES_PER_TILE", 9),
                               ("GUIDES_PER_TILE_MIN", 3),
                               ("SKY_PER_TILE", 7),
                               ("SKY_PER_TILE_MIN", 7),
                               ("QUAD_RADII", 
                               [tp.TILE_RADIUS, 8164.03, 4082.02, 0.]),
                               ("QUAD_PER_RADII", [3, 3, 1]),
                               ("use_colour_cut", True),
                               ("colour_index_cut", 0.5)])
                        
