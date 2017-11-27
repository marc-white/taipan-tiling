"""Quick script to test assigning sky fibres to dark sky coordinates.
"""
import cPickle
import numpy as np
import taipan.core as tp
import taipan.tiling as tl
import taipan.fwtiling as fwtl
from astropy.table import Table

try:
    if tiling:
        print "Already imported run"
except:
    # Import the results of the tiling run
    print "Importing tiling run"
    pkl_file_name = "results/172711_1533_46_fw_tiling.pkl"

    pkl_file = open(pkl_file_name, "rb")
    (tiling, remaining_targets, run_settings) = cPickle.load(pkl_file)
    
    # Import the sky catalogue and generate sky "targets"
    print "Importing dark sky catalogue"
    dark_sky_fits = "/Users/adamrains/Catalogues/skyfibers_v17_gaia_ucac4_final.fits"
    
    dark_sky_table = Table.read(dark_sky_fits)
    
    dark_coords = [tp.FWTarget(-1*int(star["pkey_id"]), star["ra"], star["dec"], 
                   priority=run_settings["priority_normal"], mag=25, difficulty=1, 
                   science=False, standard=False, guide=False, sky=True) 
                   for star in dark_sky_table 
                   if (run_settings["ra_min"] < star["ra"] < run_settings["ra_max"])
                   and (run_settings["dec_min"] < star["dec"] < run_settings["dec_max"])]
                   
    burn = [t.compute_usposn() for t in dark_coords]
    tp.compute_target_difficulties(dark_coords)

# All imported, now test assigning sky fibres - test with first tile
for tile_i, tile in enumerate(tiling):
    # Gather sky fibres
    sky_fibres = [fibre for fibre in tile.fibres if tile.fibres[fibre] == "sky"]

    # Get the local dark sky coordinates
    nearby_dark_coords = tp.targets_in_range(tile.ra, tile.dec, dark_coords, 
                                             1*tp.TILE_RADIUS) 
    
    successful_assignments = 0
    
    print "Assigning fibres for tile %4i..." % tile_i,
    
    # Assign sky fibres
    for fibre in sky_fibres:
        # Note that if a sky fibre fails to be assigned, it was reset to empty and has
        # lost any memory of being a "sky" fibre
        remaining_coords, former_tgt = tile.assign_fibre(fibre, nearby_dark_coords, 
                              check_patrol_radius=True, 
                              check_tile_radius=run_settings["check_tile_radius"],
                              recompute_difficulty=run_settings["recompute_difficulty"],
                              order_closest_secondary=True,
                              method=run_settings["tile_unpick_method"],
                              combined_weight=run_settings["combined_weight"],
                              sequential_ordering=(2,1,0))
                              
        if len(remaining_coords) < len(nearby_dark_coords):
            successful_assignments += 1
    
    print "%2i/%i sky fibres successfully assigned" % (successful_assignments, 
                                                       len(sky_fibres))