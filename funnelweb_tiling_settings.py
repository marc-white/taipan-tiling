"""File to easily contain the necessary parameters to tile the sky with FunnelWeb.

Modify parameters as required - this file is duplicated at the conclusion of the run for 
documentation purposes, along with a pickle of the results.
"""
import numpy as np
from collections import OrderedDict

# Input catalogue 
catalogue = ("/Users/adamrains/Google Drive/University/PhD/FunnelWeb/StellarParameters/"
             "M-dwarf Catalogues/all_tgas.fits")

# Ordered dictionary, so that format is always the same when writing to a file             
settings = OrderedDict([("input_catalogue", catalogue),
                        ("tab_type", "gaia"),
                        ("ra_min", 0),
                        ("ra_max", 10),
                        ("dec_min", -10),
                        ("dec_max", 0),
                        ("gal_lat_limit", 0),
                        ("tiling_method", "SH"),
                        ("alloc_method", "combined_weighted"),
                        ("combined_weight", 4.0),
                        ("sequential_ordering", (1,2)),
                        ("ranking_method", "priority-expsum"),
                        ("exp_base", 3.0),
                        ("completeness_target", 0.99),
                        ("inverse_standard_frac", 10),
                        ("mag_ranges", [[5,8],[7,10],[9,12],[11,14]]),
                        ("mag_ranges_prioritise", [[5,7],[7,9],[9,11],[11,12]]),
                        ("guide_range", [8,10.5]),
                        ("TARGET_PER_TILE", 139),
                        ("STANDARDS_PER_TILE", 4),
                        ("STANDARDS_PER_TILE_MIN", 3),
                        ("SKY_PER_TILE", 7),
                        ("SKY_PER_TILE_MIN", 7),
                        ("GUIDES_PER_TILE", 9),
                        ("GUIDES_PER_TILE_MIN", 3),
                        ("prioritise_extra", 2),
                        ("priority_normal", 2),
                        ("randomise_pa", True),
                        ("randomise_SH", True),
                        ("tiling_file", "ipack.3.4112.txt"),
                        ("rank_supplements", False),
                        ("repick_after_complete", False),
                        ("recompute_difficulty", True),
                        ("disqualify_below_min", True),
                        ("nthreads", 0)])
