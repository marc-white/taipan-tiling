"""File to easily contain the necessary parameters to tile the sky with FunnelWeb.

Modify parameters as required - this file is duplicated at the conclusion of the run for 
documentation purposes, along with a pickle of the results.
"""
import numpy as np
from collections import OrderedDict

# Input catalogue 
catalogue = ("/Users/adamrains/Catalogues/all_tgas.fits")

# Ordered dictionary, so that format is always the same when writing to a file 

# Dictionary of *all* parameters required to construct taipan.fwtiling.FWTiler
# e.g. fwtiler = FWTiler(**tiler_input)
tiler_input = OrderedDict([("completeness_target", 0.99),
                           ("ranking_method", "priority-expsum"),
                           ("disqualify_below_min", True),
                           ("tiling_method", "SH"),
                           ("randomise_pa", False),
                           ("randomise_SH", False),
                           ("tiling_file", "ipack.3.4112.txt"),
                           ("ra_min", 0),
                           ("ra_max", 20),
                           ("dec_min", -30),
                           ("dec_max", 0),
                           ("mag_ranges", [[5,8],[7,10],[9,12],[11,14]]),
                           ("mag_ranges_prioritise", [[5,7],[7,9],[9,11],[11,12]]),
                           ("priority_normal", 2),
                           ("prioritise_extra", 2),
                           ("tile_unpick_method", "combined_weighted"),
                           ("combined_weight", 4.0),
                           ("sequential_ordering", (1,2)),
                           ("rank_supplements", False),
                           ("repick_after_complete", False),
                           ("exp_base", 4.0),
                           ("recompute_difficulty", True),
                           ("overwrite_existing", True),
                           ("check_tile_radius", True),
                           ("consider_removed_targets", False),
                           ("allow_standard_targets", True),
                           ("assign_sky_first", True),
                           ("n_cores", 0),
                           ("backend", "multiprocessing"),
                           ("enforce_min_tile_score", True)])
 
# Dictionary of additional parameters not required by FWTiler            
script_settings = OrderedDict([("input_catalogue", catalogue),
                               ("tab_type", "gaia"),
                               ("gal_lat_limit", 0),
                               ("inverse_standard_frac", 10),
                               ("guide_range", [8,10.5]),
                               ("TARGET_PER_TILE", 139),
                               ("STANDARDS_PER_TILE", 4),
                               ("STANDARDS_PER_TILE_MIN", 3),
                               ("SKY_PER_TILE", 7),
                               ("SKY_PER_TILE_MIN", 7),
                               ("GUIDES_PER_TILE", 9),
                               ("GUIDES_PER_TILE_MIN", 3),
                               ("prioritise_close", False)])
                        
