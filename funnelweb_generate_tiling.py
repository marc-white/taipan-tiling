"""Script for generating the tiling for FunnelWeb

ipython --pylab
i.e.
run -i funnelweb_generate_tiling

Test the various implementations of assign_fibre

Test speed with: 
kernprof -l funnelweb_generate_tiling.py
python -m line_profiler script_to_profile.py.lprof
"""

import taipan.core as tp
import taipan.tiling as tl
from astropy.table import Table
import matplotlib.patches as mpatches
# from mpl_toolkits.basemap import Basemap
import random
import sys
import datetime
import logging
import numpy as np
import cPickle
import time
import os
import funnelweb_plotting as fwplt
from shutil import copyfile
from collections import OrderedDict

logging.basicConfig(filename='funnelweb_generate_tiling.log',level=logging.INFO)

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
        priority = 2
    else:
        priority = 2

    return priority
    
    
#-----------------------------------------------------------------------------------------
# Tiling Settings/Parameters
#-----------------------------------------------------------------------------------------
#Change defaults. NB By changing in tp, we chance in tl.tp also.
tp.TARGET_PER_TILE = 139
tp.STANDARDS_PER_TILE = 4
tp.STANDARDS_PER_TILE_MIN = 3

#Enough to fit a linear trend and get a chi-squared uncertainty distribution with
#4 degrees of freedom, with 1 bad fiber.
tp.SKY_PER_TILE = 7
tp.SKY_PER_TILE_MIN = 7
tp.GUIDES_PER_TILE = 9
tp.GUIDES_PER_TILE_MIN = 3

#Limits for right-ascension and declination
ra_lims = [0,10]
de_lims = [-10,0]

#Range of magnitudes for the guide stars. Note that this range isn't allowed to be 
#completely within one of the mag_ranges below
guide_range=[8,10.5]
gal_lat_limit = 0 #For Gaia data only

#infile = '/Users/mireland/tel/funnelweb/2mass_AAA_i7-10_offplane.csv'; tabtype='2mass'

### Change this below for your path!
infile = '/Users/adamrains/Google Drive/University/PhD/FunnelWeb/StellarParameters/M-dwarf Catalogues/all_tgas.fits'
tabtype = 'gaia'

#Magnitude (Gaia) ranges for each exposure time.
mag_ranges = [[5,8],[7,10],[9,12],[11,14]]

#Magnitude ranges to prioritise within each range. We make sure that these are 
#mostly complete (up to completenes_target below)
mag_ranges_prioritise = [[5,7],[7,9],[9,11],[11,12]]

#Method for tiling the sky. The following two parameter determine where the field 
#centers are.
tiling_method = 'SH'

#Method for prioritising fibers 
alloc_method = 'combined_weighted'

# Assign weighting for difficulty vs priority weight priority --> 1:4 weight vs priority
combined_weight = 4.0
sequential_ordering = (1,2) #Not used for combined_weighted. Just for 'sequential'

#Method for prioritising fields
ranking_method = 'priority-expsum'
exp_base = 3.0 #For priority-expsum, one fiber of priority k is worth exp_base fibers of priority k-1

#We move on to the next magnitude range after reaching this completeness on the priority
#targets
completeness_target = 0.99  #Originally 0.999

#Randomly choose this fraction of stars as standards. In practice, we will use colour cuts
#to choose B and A stars. An inverse standard frac of 10 will mean 1 in 10 stars are
#standards.
inverse_standard_frac = 10

#NB We have repick_after_complete set as False below - this can be done at the end.

# Save a copy of the script for future reference
# The file will be appropriately timestamped on completion of the tiling
script_name = "funnelweb_generate_tiling.py"
temp_script_name = "results/temp_" + time.strftime("%y%d%m_%H%M_") + script_name
copyfile(script_name, temp_script_name)

# Prompt user for the description or motivation of the run
run_description = raw_input("Description/motivation for tiling run: ")
if run_description == "": run_description = "NA"

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
    tabdata = Table.read(infile)
    print 'Generating targets...'
    if tabtype == '2mass':
        all_targets = [tp.TaipanTarget(str(r['mainid']), r['raj2000'], r['dej2000'], 
            priority=2, mag=r['imag'],difficulty=1) for r in tabdata 
            if ra_lims[0] < r['raj2000'] < ra_lims[1] and de_lims[0] < r['dej2000'] < de_lims[1]]
    elif tabtype == 'gaia':
        all_targets = [tp.TaipanTarget(int(r['source_id']), r['ra'], r['dec'], 
            priority=calc_dist_priority(r["parallax"]), mag=r['phot_g_mean_mag'],
            difficulty=1) for r in tabdata if ra_lims[0] < r['ra'] < ra_lims[1] and 
            de_lims[0] < r['dec'] < de_lims[1] and np.abs(r['b']) > gal_lat_limit]
    else: 
        raise UserWarning("Unknown table type") 
    
    #Take standards from the science targets, and assign the standard property to True
    for t in all_targets:
        if (int(t.mag*100) % int(inverse_standard_frac))==0:
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
    
    print 'Calculating target US positions...'
    #NB No need to double-up for standard targets or guide_targets as these are drawn from all_targets
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
guide_targets = [t for t in candidate_targets if guide_range[0]<t.mag<guide_range[1]]
print '{0:d} science, {1:d} standard and {2:d} guide targets'.format(
    len(candidate_targets), len(standard_targets), len(guide_targets))

#-----------------------------------------------------------------------------------------
# Generate Tiling
#-----------------------------------------------------------------------------------------
print 'Commencing tiling...'
start = datetime.datetime.now()
test_tiling, tiling_completeness, remaining_targets = tl.generate_tiling_funnelweb(
    candidate_targets, standard_targets, guide_targets,
    mag_ranges_prioritise = mag_ranges_prioritise,
    prioritise_extra = 2,
    priority_normal = 2,
    mag_ranges = mag_ranges,
    completeness_target=completeness_target,
    ranking_method=ranking_method,
    tiling_method=tiling_method, randomise_pa=True, 
    ra_min=ra_lims[0], ra_max=ra_lims[1], dec_min=de_lims[0], dec_max=de_lims[1],
    randomise_SH=True, tiling_file='ipack.3.4112.txt',
    tile_unpick_method=alloc_method, sequential_ordering=sequential_ordering,
    combined_weight=combined_weight,
    rank_supplements=False, repick_after_complete=False, exp_base=exp_base,
    recompute_difficulty=True, disqualify_below_min=True, nthreads=0)
end = datetime.datetime.now()

#-----------------------------------------------------------------------------------------
# Analysis
#-----------------------------------------------------------------------------------------
time_to_complete = (end - start).total_seconds()
non_standard_targets_per_tile = [t.count_assigned_targets_science(include_science_standards=False) for t in test_tiling]
targets_per_tile = [t.count_assigned_targets_science() for t in test_tiling]
standards_per_tile = [t.count_assigned_targets_standard() for t in test_tiling]
guides_per_tile = [t.count_assigned_targets_guide() for t in test_tiling]

print 'TILING STATS'
print '------------'
print 'Greedy/FunnelWeb tiling complete in %d:%2.1f' % (int(np.floor(time_to_complete/60.)),
    time_to_complete % 60.)
print '%d targets required %d tiles' % (len(all_targets), len(test_tiling), )
print 'Average %3.1f targets per tile' % np.average(targets_per_tile)
print '(min %d, max %d, median %d, std %2.1f' % (min(targets_per_tile),
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
run_settings = OrderedDict([("run_id", date_time[:-1]),
                            ("description", run_description),
                            ("input_catalogue", infile.split("/")[-1]),
                            ("ra_min", ra_lims[0]),
                            ("ra_max", ra_lims[1]),
                            ("dec_min", de_lims[0]),
                            ("dec_max", de_lims[1]),
                            ("gal_lat_limit", gal_lat_limit),
                            ("tiling_method", tiling_method),
                            ("alloc_method", alloc_method),
                            ("combined_weight", combined_weight),
                            ("ranking_method", ranking_method),
                            ("exp_base", exp_base),
                            ("completeness_target", completeness_target),
                            ("inverse_standard_frac", inverse_standard_frac),
                            ("mins_to_complete", time_to_complete),
                            ("num_targets", len(all_targets)),
                            ("num_tiles", len(test_tiling)),
                            ("avg_targets_per_tile", np.average(targets_per_tile)),
                            ("min_targets_per_tile", min(targets_per_tile)),
                            ("max_targets_per_tile", max(targets_per_tile)),
                            ("median_targets_per_tile", np.median(targets_per_tile)),
                            ("std_targets_per_tile", np.std(targets_per_tile)),
                            ("tiling_completeness", tiling_completeness),
                            ("remaining_targets", len(remaining_targets)),
                            ("mag_ranges", mag_ranges),
                            ("mag_ranges_prioritise", mag_ranges_prioritise),
                            ("non_standard_targets_per_tile", non_standard_targets_per_tile),
                            ("targets_per_tile", targets_per_tile),
                            ("standards_per_tile", standards_per_tile),
                            ("guides_per_tile", guides_per_tile)])  

# Use pickle to save outputs of tiling in a binary format
name = "results/" + date_time + "fw_tiling.pkl"
output = open(name, "wb")
cPickle.dump( (test_tiling, remaining_targets, run_settings), output, -1)
output.close()

# Timestamp the copy of the script from earlier
final_script_name = "results/" + date_time + script_name
os.rename(temp_script_name, final_script_name)

# Save a copy of the run settings (Not including the last six list entries)
txt_name = "results/" + date_time + "tiling_settings.txt"
run_settings_fmt = np.array([run_settings.keys()[:-6], run_settings.values()[:-6]]).T
np.savetxt(txt_name, run_settings_fmt, fmt="%s", delimiter="\t")
                
#-----------------------------------------------------------------------------------------
# Plotting
#-----------------------------------------------------------------------------------------
fwplt.plot_tiling(test_tiling, run_settings)