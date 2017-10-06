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
import taipan.fwtiling as fwtl
from taipan.fwtiling import FWTiler
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
def load_targets(catalogue, ra_min, ra_max, dec_min, dec_max, prioritise_close, 
                 gal_lat_limit, tab_type):
    """Function to import the input catalogue and make any required cuts.
    
    Parameters
    ----------
    catalogue: string
        File path to the input catalogue
    ra_min: float
        Minimum RA to consider (degrees)
    ra_max: float
        Maximum RA to consider (degrees)  
    dec_min: float
        Minimum DEC to consider (degrees)
    dec_max: float
        Maximum DEC to consider (degrees)
    prioritise_close: boolean
        Boolean value indicating whether closer targets get increased priorities. A 
        stand-in function for any initial prioritisation for FW targets (thus subject to
        change).
    gal_lat_limit: float
        The limit off the Galactic plane to consider (degrees).
    tab_type: string
        The format of the input catalogue, currently either '2mass' or 'gaia'.
        
    Returns
    -------
    all_targets: list of TaipanTarget objects
        A list containing all candidate targets within the constraints given.            
    """
    # Load in the entire input catalogue
    tabdata = Table.read(catalogue)
    
    print 'Making mag/DEC/RA/b cuts and generating targets...'
    if tab_type == '2mass':
        all_targets = [tp.TaipanTarget(str(r['mainid']), r['raj2000'], r['dej2000'], 
            priority=2, mag=r['imag'],difficulty=1) for r in tabdata 
            if ra_min < r['raj2000'] < ra_max and dec_min < r['dej2000'] < dec_max]
    elif tab_type == 'gaia':
        all_targets = [tp.TaipanTarget(int(r['source_id']), r['ra'], r['dec'], 
            priority=calc_dist_priority(r["parallax"], prioritise_close), 
            mag=r['phot_g_mean_mag'], difficulty=1) for r in tabdata 
            if ra_min < r['ra'] <  ra_max and dec_min < r['dec'] < dec_max and  
            np.abs(r['b']) > gal_lat_limit]
    else: 
        raise UserWarning("Unknown table type")
        
    return all_targets

def calc_dist_priority(parallax, prioritise_close=False):
    """Function to return a taipan priority integer based on the target distance
    
    Parameters
    ----------
    parallax: float
        Parallax of the star in milli-arcsec
    
    prioritise_close: Boolean
        Boolean indicating whether to upweight closer stars, or have constant priorities.
    
    Returns
    -------
    priority: int
        Priority of the star
    """
    # Check to avoid changing priorities
    if not prioritise_close:
        return 2
    
    # If prioritise_close, assign a higher priority to closer targets
    
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
tp.TARGET_PER_TILE = fwts.script_settings["TARGET_PER_TILE"]
tp.STANDARDS_PER_TILE = fwts.script_settings["STANDARDS_PER_TILE"]
tp.STANDARDS_PER_TILE_MIN = fwts.script_settings["STANDARDS_PER_TILE_MIN"]

#Enough to fit a linear trend and get a chi-squared uncertainty distribution with
#4 degrees of freedom, with 1 bad fiber.
tp.SKY_PER_TILE = fwts.script_settings["SKY_PER_TILE"]
tp.SKY_PER_TILE_MIN = fwts.script_settings["SKY_PER_TILE_MIN"]
tp.GUIDES_PER_TILE = fwts.script_settings["GUIDES_PER_TILE"]
tp.GUIDES_PER_TILE_MIN = fwts.script_settings["GUIDES_PER_TILE_MIN"]

# Save a copy of the settings file for future reference
# The file will be appropriately timestamped on completion of the tiling
temp_timestamp = time.strftime("%y%d%m_%H%M_%S_")
settings_file = "funnelweb_tiling_settings.py"
temp_settings_file = "results/temp_" + temp_timestamp + settings_file
copyfile(settings_file, temp_settings_file)

# Prompt user for the description or motivation of the run
run_description = raw_input("Description/motivation for tiling run: ")

# No description given, assign one based on on-sky area and number of cores
if run_description == "": 
    ra_range = fwts.tiler_input["ra_max"] - fwts.tiler_input["ra_min"]
    dec_range = fwts.tiler_input["dec_max"] - fwts.tiler_input["dec_min"]
    run_description = "%ix%i, n_cores=%i" % (ra_range, dec_range, 
                                             fwts.tiler_input["n_cores"])

# Begin logging, ensuring that we create a new log file handler unique to this run
log_file = "funnelweb_generate_tiling.log"
temp_log_file = "results/temp_" + temp_timestamp + log_file
log_file_handler = logging.FileHandler(temp_log_file, "a")
logging.getLogger().handlers = [log_file_handler]

# Initialise the tiler
fwtiler = FWTiler(**fwts.tiler_input)

#-----------------------------------------------------------------------------------------
# Target Input, Priorities, Standards, and Guides
#-----------------------------------------------------------------------------------------
try:
    if all_targets:
        print "Previously loaded catalogue will be used, %i stars" % len(all_targets)
        
except NameError:
    print 'Importing input catalogue...'
    start = datetime.datetime.now()
    # It is important not to duplicate targets, i.e. if a science targets is a guide and
    # a standard, then it should become a single instance of a TaipanTarget, appearing in
    # all lists.
    all_targets = load_targets(fwts.script_settings["input_catalogue"],
                               fwts.tiler_input["ra_min"], fwts.tiler_input["ra_max"], 
                               fwts.tiler_input["dec_min"], fwts.tiler_input["dec_max"], 
                               fwts.script_settings["prioritise_close"],
                               fwts.script_settings["gal_lat_limit"],
                               fwts.script_settings["tab_type"])
    
    # Deallocate tabdata 
    # (Not required now that import is in function)
    # del tabdata
    
    #Take standards from the science targets, and assign the standard property to True
    for t in all_targets:
        if (int(t.mag*100) % int(fwts.script_settings["inverse_standard_frac"]))==0:
            t.standard=True
    #    if t.mag > 8:
    #        t.guide=True
    end = datetime.datetime.now()
    delta = end - start
    print ("Imported & generated %d targets in %d:%02.1f") % (len(all_targets),
                                                              delta.total_seconds()/60, 
                                                              delta.total_seconds() % 60.)
    
    # No need to double-up for standard/guide targets as these are drawn from all_targets
    print 'Calculating target US positions...'
    burn = [t.compute_usposn() for t in all_targets]

# Make a copy of all_targets list for use in assigning fibres
candidate_targets = all_targets[:]
random.shuffle(candidate_targets)

#Make sure that standards and guides are drawn from candidate_targets, not our master
#all_targets list.
standard_targets = [t for t in candidate_targets if t.standard==True]
guide_targets = [t for t in candidate_targets if fwts.script_settings["guide_range"][0] 
                    < t.mag < fwts.script_settings["guide_range"][1]]
print '{0:d} science, {1:d} standard and {2:d} guide targets'.format(
    len(candidate_targets), len(standard_targets), len(guide_targets))

#-----------------------------------------------------------------------------------------
# Generate Tiling
#-----------------------------------------------------------------------------------------
print 'Commencing tiling with %i core/s...' % (fwts.tiler_input["n_cores"])
start = datetime.datetime.now()
tiling, completeness, remaining_targets = fwtiler.generate_tiling_funnelweb_mp(
                                            candidate_targets, standard_targets, 
                                            guide_targets)

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
print 'FW tiling complete using %i core/s in %d:%2.1f' % (fwts.tiler_input["n_cores"],
        int(np.floor(time_to_complete/60.)), time_to_complete % 60.)
print '%d targets required %d tiles' % (len(all_targets), len(tiling), )
print 'Tiling completeness = %4.4f, %i targets remaining' % (completeness, 
        len(remaining_targets))
print 'Average %3.1f targets per tile' % np.average(targets_per_tile)
print '(min %d, max %d, median %d, std %2.1f)' % (min(targets_per_tile),
    max(targets_per_tile), np.median(targets_per_tile), 
    np.std(targets_per_tile))
print "%i total targets, %i unique targets, %i duplicate targets" % \
    fwtl.count_unique_science_targets(tiling)
    
#-----------------------------------------------------------------------------------------
# Saving Tiling Outputs
#-----------------------------------------------------------------------------------------
# Use time stamp as run ID
date_time = time.strftime("%y%d%m_%H%M_%S_")

# Document the settings and results of the tiling run
# Dictionary used to easily load results/settings of past runs, OrderedDict so txt has 
# same format for every run (i.e. the keys are in the order added)
run_settings = OrderedDict([("run_id", date_time[:-1]),
                            ("description", run_description),
                            ("mins_to_complete", time_to_complete/60.),
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

# Append input parameters for plotting/saving
run_settings.update(fwts.tiler_input) 
run_settings.update(fwts.script_settings)                             

# Use pickle to save outputs of tiling in a binary format
name = "results/" + date_time + "fw_tiling.pkl"
output = open(name, "wb")
cPickle.dump( (tiling, remaining_targets, run_settings), output, -1)
output.close()

# Timestamp the copy of the settings and log files from earlier
final_settings_file = "results/" + date_time + settings_file
os.rename(temp_settings_file, final_settings_file)

final_log_file = "results/" + date_time + log_file
os.rename(temp_log_file, final_log_file)

print "Output files saved as results/%s*" % date_time
                
#-----------------------------------------------------------------------------------------
# Plotting
#-----------------------------------------------------------------------------------------
fwplt.plot_tiling(tiling, run_settings)