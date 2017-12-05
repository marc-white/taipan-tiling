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
import time
import logging
import numpy as np
import cPickle
import time
import os
import platform
import funnelweb_plotting as fwplt
import funnelweb_tiling_settings as fwts
from shutil import copyfile
from collections import OrderedDict

#-----------------------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------------------
def load_targets(catalogue, ra_min, ra_max, dec_min, dec_max, gal_lat_limit, tab_type,
                 priorities=None, priority_normal=2, use_colour_cut=False, 
                 standard_frac=0.1, colour_index_cut=0.5):
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
    gal_lat_limit: float
        The limit off the Galactic plane to consider (degrees).
    tab_type: string
        The format of the input catalogue, currently either 'gaia' or 'fw'.
    priorities: string or None
        Either a string filepath to a .fits file with ID-priority pairs, or None if every
        star should be assigned the same initial priority of priority_normal.
    priority_normal: int
        The normal priority to assign to all stars if we don't have the ID-priority pairs.
    use_colour_cut: boolean
        Boolean indicating whether standard stars are selected based on a colour cut, or 
        whether to simply use a fraction of the total stars as standards.
    standard_frac: float
        Fraction between 0.0 and 1.0 representing the fraction of stars to be considered
        standards if not using a colour cut.
    colour_index_cut: float
        The colour index cut below which (i.e. hotter than which) standards are selected. 
              
    Returns
    -------
    all_targets: list of TaipanTarget objects
        A list containing all candidate targets within the constraints given.            
    """
    # Load in the entire input catalogue
    start = time.time()
    tabdata = Table.read(catalogue)
    delta = time.time() - start
    print ("Loaded input catalogue in %d:%02.1f") % (delta/60, delta % 60.)
    
    # If provided, import the target priorities
    if priorities:
        start = time.time()
        priorities = import_target_priorities(priorities)
        delta = time.time() - start
        print ("Loaded target priorities in %d:%02.1f") % (delta/60, delta % 60.)
    
    start = time.time()
    all_targets = []
    
    # Create TaipanTarget objects for those targets meeting RA/DEC/b requirements
    if tab_type == 'gaia':
        for star in tabdata:
            # Only consider targets which satisfy RA/DEC/b restrictions
            if (ra_min < star["ra"] < ra_max) and (dec_min < star["dec"] < dec_max) \
                and (np.abs(star['b']) > gal_lat_limit):
               # Target is acceptable, create with parameters necessary for tiling
                target = tp.FWTarget(int(star["source_id"]), star["ra"], star["dec"], 
                                         priority=get_target_priority(star["source_id"], 
                                                                      priorities, 
                                                                      priority_normal), 
                                         mag=star["phot_g_mean_mag"], difficulty=1)
                
                # Now check whether the star is a standard or not. We cannot (currently)
                # do this in a list comprehension as assigning anything to non-standard
                # stars (i.e. False or None) results in bugs within the tiling/taipan 
                # code, the cause of which is unclear and fixing it is not currently a
                # priority. We have to do this here if we want to access other magnitudes
                # for making cuts based on colour index (i.e. using 2MASS J & K), numbers
                # which are not required elsewhere in the tiling code so it is needlessly
                # complex to assign them as class parameters.
                if is_standard(star, use_colour_cut, standard_frac, tab_type):
                    target.standard = True           
            
                # All done, add to the master list
                all_targets.append(target)    
            
    elif tab_type == "fw":
        for star in tabdata:
            # Only consider targets which satisfy RA/DEC/b restrictions
            if (ra_min < star["RA_ep2015"] < ra_max) \
                and (dec_min < star["Dec_ep2015"] < dec_max) \
                and (np.abs(star['b']) > gal_lat_limit) and star["Gaia_G_mag"] <= 30:
               # Target is acceptable, create with parameters necessary for tiling
                target = tp.FWTarget(int(star["Gaia_ID"]), star["RA_ep2015"], 
                                         star["Dec_ep2015"], 
                                         priority=get_target_priority(star["Gaia_ID"], 
                                                                      priorities, 
                                                                      priority_normal), 
                                         mag=star["Gaia_G_mag"], difficulty=1)
                
                # Check if star is a standard (same reasoning as above for tab_type=gaia)
                if is_standard(star, use_colour_cut, standard_frac, tab_type):
                    target.standard = True           
            
                # All done, add to the master list
                all_targets.append(target)
    
    else: 
        raise UserWarning("Unknown table type")
     
    delta = time.time() - start
    print "Generated targets in %d:%02.1f" % (delta/60, delta % 60.)
    
    # Compute the unit sphere (US) positions for each target
    start = time.time()
    burn = [t.compute_usposn() for t in all_targets]
    delta = time.time() - start
    print "Calculated target US positions in %d:%02.1f" % (delta/60, delta % 60.)
    
    return all_targets
    

def is_standard(star, use_colour_cut=False, standard_frac=0.1, tabtype="fw", 
                colour_index_cut=0.5):
    """Prototype function to determine whether a star can be considered a standard or not.
    
    Parameters
    ----------
    star: list
        Row of the input catalogue corresponding the the star to be checked.
    use_colour_cut: boolean
        Boolean indicating whether standard stars are selected based on a colour cut, or 
        whether to simply use a fraction of the total stars as standards.
    standard_frac: float
        Fraction between 0.0 and 1.0 representing the fraction of stars to be considered
        standards if not using a colour cut.
    tab_type: string
        The format of the input catalogue, currently either 'gaia' or 'fw'.
    colour_index_cut: float
        The colour index cut below which (i.e. hotter than which) standards are selected.
                
    Returns
    -------
    is_standard: boolean
        Boolean value indicating whether the star can be considered a standard or not.
    """
    if use_colour_cut:
        # Calculate the colour indices
        G_minus_J = star["Gaia_G_mag"] - star["2MASS_J_mag"]
        J_minus_K = star["2MASS_J_mag"] - star["2MASS_Ks_mag"]

        # Select as standards only those stars below the colour index cut
        if G_minus_J <= colour_index_cut:
            is_standard = True
        else:
            is_standard = False
    else:
        # Not using a colour cut - assign standards based on a simple fraction.
        # TODO: Update to this to allow for stars without G mags
        if tabtype == "fw" and (int(star["Gaia_G_mag"]*100) % int(1/standard_frac)) == 0:
            is_standard = True
            
        elif tabtype == "gaia" \
            and (int(star["phot_g_mean_mag"]*100) % int(1/standard_frac)) == 0:
            is_standard = True
            
        else:
            is_standard = False
    
    return is_standard


def get_target_priority(target_id, target_priorities=None, normal_priority=2):
    """Attempts to lookup the provided target ID in the master list of target priorities,
    returning the priority if found. Otherwise, returns the accepted "normal" priority.
    
    Parameters
    ----------
    target_id: int
        The target ID to lookup and find the priority of.
    target_priorities: dict
        Dictionary mapping target IDs to priority.
    normal_priority: int
        The "normal" priority level to be used if the ID cannot be found.
        
    Returns
    -------
    priority: int
        The target priority. 
    """
    # First check that we have a dictionary
    if not target_priorities:
        return normal_priority
    
    # We have a dictionary of priorities, check for membership    
    try:
        priority = target_priorities[target_id]
    except:
        priority = normal_priority
        
        # TODO: keep track of which targets don't have priorities
        
    return priority
    
    
def import_target_priorities(priorities_fits_file):
    """Imports the target priorities and converts to a dictionary format to optimise 
    priority lookup when constructing targets.
    
    Parameters
    ----------
    priorities_fits_file: string
        The file path to the on-disk list of target IDs and priorities.
        
    Returns
    -------
    target_priorities: dict
        Dictionary mapping target IDs to priority.
    """
    # Load in the input priority list
    target_priorities_table = Table.read(priorities_fits_file)
    
    # Initialise the dictionary that will be used to store the ID-priority pairs
    # Quicker to construct dictionary from keys (i.e. pre-size), then populate
    ids = set(target_priorities_table["Gaia_ID"])
    target_priorities = dict.fromkeys(ids)
    
    for target in target_priorities_table:
        target_priorities[target[0]] = target[1]
        
    return target_priorities


def load_sky_targets(dark_sky_fits, ra_min, ra_max, dec_min, dec_max, priority_normal=2):
    """Function to import the input sky catalogue as FWTargets and make any required cuts.
    
    Parameters
    ----------
    dark_sky_fits: string
        File path to the sky catalogue
    ra_min: float
        Minimum RA to consider (degrees)
    ra_max: float
        Maximum RA to consider (degrees)  
    dec_min: float
        Minimum DEC to consider (degrees)
    dec_max: float
        Maximum DEC to consider (degrees)
    priority_normal: int
        The normal priority to assign to all sky positions.
            
    Returns
    -------
    sky_targets: list of FWTarget objects
        A list containing all candidate targets within the constraints given.            
    """
    # Import the dark sky catalogue
    start = time.time()
    sky_table = Table.read(dark_sky_fits)
    delta = time.time() - start
    print ("Loaded sky catalogue in %d:%02.1f") % (delta/60, delta % 60.)
    
    # Generate FWTargets for each sky object within our bounds
    start = time.time()
    sky_targets = [tp.FWTarget(-1*int(sky["pkey_id"]), sky["ra"], sky["dec"], 
                   priority=priority_normal, mag=25, difficulty=1, 
                   science=False, standard=False, guide=False, sky=True) 
                   for sky in sky_table 
                   if (ra_min < sky["ra"] < ra_max) and (dec_min < sky["dec"] < dec_max)]
    
    
    delta = time.time() - start
    print "Generated sky targets in %d:%02.1f" % (delta/60, delta % 60.)
    
    # Pre-compute initial params
    start = time.time()               
    burn = [t.compute_usposn() for t in sky_targets]
    tp.compute_target_difficulties(sky_targets)
    delta = time.time() - start
    print "Calculated sky US positions in %d:%02.1f" % (delta/60, delta % 60.)
    
    return sky_targets
    
    
def update_taipan_quadrants():
    """Function to recompute the number and placement of quadrants that split up a field
    and define sky fibre locations.
    """
    if len(tp.QUAD_PER_RADII) != len(tp.QUAD_RADII)-1:
        raise ValueError('QUAD_PER_RADII must have one less element than '
                     'QUAD_RADII')
    if sum(tp.QUAD_PER_RADII) != tp.SKY_PER_TILE:
        raise UserWarning('The number of defined tile quandrants does not match '
                          'SKY_PER_TILE. These are meant to be the same!')

    tp.FIBRES_PER_QUAD = []
    for i in range(len(tp.QUAD_RADII))[:-1]:
        theta = 360. / tp.QUAD_PER_RADII[i]
        for j in range(tp.QUAD_PER_RADII[i]):
            tp.FIBRES_PER_QUAD.append(
                [k for k in tp.BUGPOS_OFFSET.keys() if
                 k not in tp.FIBRES_GUIDE and
                 tp.QUAD_RADII[i+1] <= tp.BUGPOS_OFFSET[k][0] < tp.QUAD_RADII[i] and
                 j*theta <= tp.BUGPOS_OFFSET[k][1] < (j+1)*theta])
                       
#-----------------------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------------------
# Reload fwts for repeated runs
reload(fwts)

# Change defaults. NB By changing in tp, we chance in tl.tp and fwtl.tp also.
tp.TARGET_PER_TILE = fwts.script_settings["TARGET_PER_TILE"]
tp.STANDARDS_PER_TILE = fwts.script_settings["STANDARDS_PER_TILE"]
tp.STANDARDS_PER_TILE_MIN = fwts.script_settings["STANDARDS_PER_TILE_MIN"]
tp.GUIDES_PER_TILE = fwts.script_settings["GUIDES_PER_TILE"]
tp.GUIDES_PER_TILE_MIN = fwts.script_settings["GUIDES_PER_TILE_MIN"]

# For guides and sky fibres: enough to fit a linear trend and get a chi-squared 
# uncertainty distribution with 4 degrees of freedom, with 1 bad fibre.

# To update the number of sky fibres, you need to update the following four parameters
# and recompute the how the tile area itself is segmented into "quadrants". Sky fibres are
# allocated in increments of the number of quadrants, so this needs to be recomputed in 
# order to properly update the number of sky fibres used.
tp.SKY_PER_TILE = fwts.script_settings["SKY_PER_TILE"]
tp.SKY_PER_TILE_MIN = fwts.script_settings["SKY_PER_TILE_MIN"]
tp.QUAD_RADII = fwts.script_settings["QUAD_RADII"]
tp.QUAD_PER_RADII = fwts.script_settings["QUAD_PER_RADII"]

update_taipan_quadrants()

# Save a copy of the settings file for future reference
# The file will be appropriately timestamped on completion of the tiling
temp_timestamp = time.strftime("%y%d%m_%H%M_%S_")
settings_file = "funnelweb_tiling_settings.py"
temp_settings_file = "results/temp_" + temp_timestamp + settings_file
copyfile(settings_file, temp_settings_file)

# Prompt user for the description or motivation of the run
run_description = raw_input("Description/motivation for tiling run: ")

# No description given, assign one based on on-sky area, machine, and number of cores
if run_description == "": 
    ra_range = fwts.tiler_input["ra_max"] - fwts.tiler_input["ra_min"]
    dec_range = fwts.tiler_input["dec_max"] - fwts.tiler_input["dec_min"]
    run_description = "%s, %ix%i, backend=%s, n_cores=%i" % (platform.node(),
                                                             ra_range, dec_range, 
                                                             fwts.tiler_input["backend"],
                                                             fwts.tiler_input["n_cores"])

# Begin logging, ensuring that we create a new log file handler unique to this run
log_file = "funnelweb_generate_tiling.log"
temp_log_file = "results/temp_" + temp_timestamp + log_file
log_file_handler = logging.FileHandler(temp_log_file, "a")
logging.getLogger().handlers = [log_file_handler]

# Initialise the tiler
fwtiler = FWTiler(**fwts.tiler_input)

#-----------------------------------------------------------------------------------------
# Importing & Generating Science, Standard, and Guide Targets
#-----------------------------------------------------------------------------------------
try:
    # Check to see if we already have the targets imported, thus saving time
    if all_targets:
        print "Previously loaded catalogue will be used, on-sky area: %ix%i" % (ra_range, 
                                                                                dec_range)
except NameError:
    # Not already imported, so import targets and generate TaipanTarget objects for each
    cat = fwts.script_settings["input_catalogue"].split("/")[-1]
    print "Importing catalogue '%s' for on-sky area: %ix%i" % (cat, ra_range, dec_range)

    all_targets = load_targets(fwts.script_settings["input_catalogue"],
                               fwts.tiler_input["ra_min"], fwts.tiler_input["ra_max"], 
                               fwts.tiler_input["dec_min"], fwts.tiler_input["dec_max"],
                               fwts.script_settings["gal_lat_limit"],
                               fwts.script_settings["tab_type"],
                               fwts.script_settings["input_priorities"], 
                               fwts.tiler_input["priority_normal"], 
                               fwts.script_settings["use_colour_cut"], 
                               fwts.script_settings["standard_frac"],
                               fwts.script_settings["colour_index_cut"])
    
    all_sky = load_sky_targets(fwts.script_settings["sky_catalogue"],
                               fwts.tiler_input["ra_min"], fwts.tiler_input["ra_max"], 
                               fwts.tiler_input["dec_min"], fwts.tiler_input["dec_max"],
                               fwts.tiler_input["priority_normal"])
                               
# Make a copy of all_targets list for use in assigning fibres. For speed, this is a set
candidate_targets = set(all_targets)

sky_targets = all_sky[:]

# Now create separate lists for standard and guide targets. These will be drawn from the 
# candidate_targets set (i.e. using references to the same objects, rather than copies).

# Standard stars are those that we previously flagged the 'standard' flag for
standard_targets = [t for t in candidate_targets if t.standard==True]

# Guide targets are drawn from a separate magnitude range and will have their flags set
# later during the tiling itself
guide_targets = set([t for t in candidate_targets 
                     if fwts.script_settings["guide_range"][0] < t.mag 
                     < fwts.script_settings["guide_range"][1]])          
                                                  
#-----------------------------------------------------------------------------------------
# Generate Tiling
#-----------------------------------------------------------------------------------------
print "Commencing tiling with %i core/s using" % (fwts.tiler_input["n_cores"]),
print "%i science, %i standard, %i guide, & %i sky targets\n" % (len(candidate_targets), 
                                                                 len(standard_targets), 
                                                                 len(guide_targets),
                                                                 len(sky_targets))
start = datetime.datetime.now()
tiling, completeness, remaining_targets = fwtiler.generate_tiling(candidate_targets, 
                                                                  standard_targets, 
                                                                  guide_targets, 
                                                                  sky_targets)
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
sky_per_tile = [t.count_assigned_targets_sky() for t in tiling]

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
                            ("guides_per_tile", guides_per_tile),
                            ("sky_per_tile", sky_per_tile)])

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
# Create a pdf summary of the tiling run, with histograms broken up by magnitude range
fwplt.plot_tiling(tiling, run_settings)