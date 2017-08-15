"""Script for generating the tiling for FunnelWeb

Notes about this script:
- To change the settings used for tiling generation, modify funnelweb_tiling_settings.py
- ipython "%run -i funnelweb_generate_tiling" to avoid having to read in target list twice
  for repeated runs
- Test speed with:
    kernprof -l funnelweb_generate_tiling.py
    python -m line_profiler script_to_profile.py.lprof
"""
import taipan.core as tp
import taipan.tiling as tl
from astropy.table import Table
import random
import sys
import datetime
import logging
import numpy as np
import cPickle
import time
import os
import funnelweb_plotting as fwplt
import funnelweb_tiling_settings as fwts
from shutil import copyfile
from collections import OrderedDict

#-----------------------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------------------
def calc_dist_priority(parallax):
    """Function to return a taipan priority integer based on the target distance
    
    Parameters
    ----------
    parallax: float
        Parallax of the star in milli-arcsec
    
    Returns
    -------
    priority: int
        Priority of the star
    """
    # Calculate the distance, where parallax is in milli-arcsec
    distance = 1000. / parallax
    
    if np.abs(distance) <= 150:
        priority = 3
    else:
        priority = 2

    return priority
    
#-----------------------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------------------
# Reload fwts for repeated runs
reload(fwts)

#Change defaults. NB By changing in tp, we chance in tl.tp also.
tp.TARGET_PER_TILE = fwts.settings["TARGET_PER_TILE"]
tp.STANDARDS_PER_TILE = fwts.settings["STANDARDS_PER_TILE"]
tp.STANDARDS_PER_TILE_MIN = fwts.settings["STANDARDS_PER_TILE_MIN"]

#Enough to fit a linear trend and get a chi-squared uncertainty distribution with
#4 degrees of freedom, with 1 bad fiber.
tp.SKY_PER_TILE = fwts.settings["SKY_PER_TILE"]
tp.SKY_PER_TILE_MIN = fwts.settings["SKY_PER_TILE_MIN"]
tp.GUIDES_PER_TILE = fwts.settings["GUIDES_PER_TILE"]
tp.GUIDES_PER_TILE_MIN = fwts.settings["GUIDES_PER_TILE_MIN"]

# Save a copy of the settings file for future reference
# The file will be appropriately timestamped on completion of the tiling
settings_file = "funnelweb_tiling_settings.py"
temp_settings_file = "results/temp_" + time.strftime("%y%d%m_%H%M_") + settings_file
copyfile(settings_file, temp_settings_file)

# Prompt user for the description or motivation of the run
run_description = raw_input("Description/motivation for tiling run: ")
if run_description == "": run_description = "NA"

# Begin logging
logging.basicConfig(filename='funnelweb_generate_tiling.log', level=logging.INFO)

#-----------------------------------------------------------------------------------------
# Target Input, Priorities, Standards, and Guides
#-----------------------------------------------------------------------------------------
try:
    if all_targets:
        pass
except NameError:
    print 'Importing test data...'
    start = datetime.datetime.now()
    #It is important not to duplicate targets, i.e. if a science targets is a guide and
    #a standard, then it should become a single instance of a TaipanTarget, appearing in
    #all lists.
    tabdata = Table.read(fwts.settings["input_catalogue"])
    print 'Generating targets...'
    if fwts.settings["tab_type"] == '2mass':
        all_targets = [tp.TaipanTarget(str(r['mainid']), r['raj2000'], r['dej2000'], 
            priority=2, mag=r['imag'],difficulty=1) for r in tabdata 
            if fwts.settings["ra_min"] < r['raj2000'] < fwts.settings["ra_max"] and 
            fwts.settings["dec_min"] < r['dej2000'] < fwts.settings["dec_min"]]
    elif fwts.settings["tab_type"] == 'gaia':
        all_targets = [tp.TaipanTarget(int(r['source_id']), r['ra'], r['dec'], 
            priority=calc_dist_priority(r["parallax"]), mag=r['phot_g_mean_mag'],
            difficulty=1) for r in tabdata 
            if fwts.settings["ra_min"] < r['ra'] <  fwts.settings["ra_max"] and 
            fwts.settings["dec_min"] < r['dec'] < fwts.settings["dec_max"] and  
            np.abs(r['b']) > fwts.settings["gal_lat_limit"]]
    else: 
        raise UserWarning("Unknown table type") 
    
    #Take standards from the science targets, and assign the standard property to True
    for t in all_targets:
        if (int(t.mag*100) % int(fwts.settings["inverse_standard_frac"]))==0:
            t.standard=True
    #    if t.mag > 8:
    #        t.guide=True
    end = datetime.datetime.now()
    delta = end - start
    print ('Imported & generated %d targets'
        ' in %d:%02.1f') % (
        len(all_targets),
        delta.total_seconds()/60, 
        delta.total_seconds() % 60.)
    
    # No need to double-up for standard/guide targets as these are drawn from all_targets
    print 'Calculating target US positions...'
    burn = [t.compute_usposn() for t in all_targets]

# tp.compute_target_difficulties(all_targets, ncpu=4)
# sys.exit()

# Ensure the objects are re type-cast as new instances of TaipanTarget (not needed?)
#for t in all_targets:
#    t.__class__ = tp.TaipanTarget

# Make a copy of all_targets list for use in assigning fibres
candidate_targets = all_targets[:]
random.shuffle(candidate_targets)

#Make sure that standards and guides are drawn from candidate_targets, not our master
#all_targets list.
standard_targets = [t for t in candidate_targets if t.standard==True]
guide_targets = [t for t in candidate_targets if fwts.settings["guide_range"][0] < t.mag < 
                 fwts.settings["guide_range"][1]]
print '{0:d} science, {1:d} standard and {2:d} guide targets'.format(
    len(candidate_targets), len(standard_targets), len(guide_targets))

#-----------------------------------------------------------------------------------------
# Generate Tiling
#-----------------------------------------------------------------------------------------
print 'Commencing tiling...'
start = datetime.datetime.now()
tiling, completeness, remaining_targets = tl.generate_tiling_funnelweb(
    candidate_targets, 
    standard_targets, 
    guide_targets,
    mag_ranges_prioritise=fwts.settings["mag_ranges_prioritise"],
    prioritise_extra=fwts.settings["prioritise_extra"],
    priority_normal=fwts.settings["priority_normal"],
    mag_ranges=fwts.settings["mag_ranges"],
    completeness_target=fwts.settings["completeness_target"],
    ranking_method=fwts.settings["ranking_method"],
    tiling_method=fwts.settings["tiling_method"], 
    randomise_pa=fwts.settings["randomise_pa"], 
    ra_min=fwts.settings["ra_min"], 
    ra_max=fwts.settings["ra_max"], 
    dec_min=fwts.settings["dec_min"], 
    dec_max=fwts.settings["dec_max"],
    randomise_SH=fwts.settings["randomise_SH"], 
    tiling_file=fwts.settings["tiling_file"],
    tile_unpick_method=fwts.settings["alloc_method"], 
    sequential_ordering=fwts.settings["sequential_ordering"],
    combined_weight=fwts.settings["combined_weight"],
    rank_supplements=fwts.settings["rank_supplements"], 
    repick_after_complete=fwts.settings["repick_after_complete"], 
    exp_base=fwts.settings["exp_base"],
    recompute_difficulty=fwts.settings["recompute_difficulty"], 
    disqualify_below_min=fwts.settings["disqualify_below_min"], 
    nthreads=fwts.settings["nthreads"])
end = datetime.datetime.now()

#-----------------------------------------------------------------------------------------
# Analysis
#-----------------------------------------------------------------------------------------
time_to_complete = (end - start).total_seconds()
non_standard_targets_per_tile = [t.count_assigned_targets_science(
                                 include_science_standards=False) for t in tiling]
targets_per_tile = [t.count_assigned_targets_science() for t in tiling]
standards_per_tile = [t.count_assigned_targets_standard() for t in tiling]
guides_per_tile = [t.count_assigned_targets_guide() for t in tiling]

print 'TILING STATS'
print '------------'
print 'Greedy FW tiling complete in %d:%2.1f' % (int(np.floor(time_to_complete/60.)),
    time_to_complete % 60.)
print '%d targets required %d tiles' % (len(all_targets), len(tiling), )
print 'Average %3.1f targets per tile' % np.average(targets_per_tile)
print '(min %d, max %d, median %d, std %2.1f)' % (min(targets_per_tile),
    max(targets_per_tile), np.median(targets_per_tile), 
    np.std(targets_per_tile))
    
#-----------------------------------------------------------------------------------------
# Saving Tiling Outputs
#-----------------------------------------------------------------------------------------
# Use time stamp as run ID
date_time = time.strftime("%y%d%m_%H%M_")

# Document the settings and results of the tiling run
# Dictionary used to easily load results/settings of past runs, OrderedDict so txt has 
# same format for every run (i.e. the keys are in the order added)
fwts.settings.update([("run_id", date_time[:-1]),
                        ("description", run_description),
                        ("mins_to_complete", time_to_complete),
                        ("num_targets", len(all_targets)),
                        ("num_tiles", len(tiling)),
                        ("avg_targets_per_tile", np.average(targets_per_tile)),
                        ("min_targets_per_tile", min(targets_per_tile)),
                        ("max_targets_per_tile", max(targets_per_tile)),
                        ("median_targets_per_tile", np.median(targets_per_tile)),
                        ("std_targets_per_tile", np.std(targets_per_tile)),
                        ("tiling_completeness", completeness),
                        ("remaining_targets", len(remaining_targets)),
                        ("non_standard_targets_per_tile", 
                         non_standard_targets_per_tile),
                        ("targets_per_tile", targets_per_tile),
                        ("standards_per_tile", standards_per_tile),
                        ("guides_per_tile", guides_per_tile)])  

# Use pickle to save outputs of tiling in a binary format
name = "results/" + date_time + "fw_tiling.pkl"
output = open(name, "wb")
cPickle.dump( (tiling, remaining_targets, fwts.settings), output, -1)
output.close()

# Timestamp the copy of the settings file from earlier
final_settings_file = "results/" + date_time + settings_file
os.rename(temp_settings_file, final_settings_file)

print "Output files saved as results/%s*" % date_time
                
#-----------------------------------------------------------------------------------------
# Plotting
#-----------------------------------------------------------------------------------------
fwplt.plot_tiling(tiling, fwts.settings)