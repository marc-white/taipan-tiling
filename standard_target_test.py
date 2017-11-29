"""
"""
import numpy as np
import taipan.core as tp
import taipan.fwtiling as fwtl
from astropy.table import Table
import funnelweb_tiling_settings as fwts
import time

#-----------------------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------------------
def load_targets(catalogue, ra_min, ra_max, dec_min, dec_max, gal_lat_limit, tab_type,
                 priorities=None, priority_normal=2, use_colour_cut=False, 
                 standard_frac=0.1, colour_index="G-J"):
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
                if is_standard(star, use_colour_cut, standard_frac, tab_type, "G-J"):
                    target.standard = True           
                if is_standard(star, use_colour_cut, standard_frac, tab_type, "J-K"):
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
                
                # Dodgy, but useful method for counting:
                # Assign stars that satisfy the G-J colour index as standards, and those
                # that satisfy the J-K index as guides
                if is_standard(star, use_colour_cut, standard_frac, tab_type, "G-J"):
                    target.standard = True   
                            
                if is_standard(star, use_colour_cut, standard_frac, tab_type, "J-K"):
                    target.guide = True         
            
                # All done, add to the master list
                all_targets.append(target)
    
    else: 
        raise UserWarning("Unknown table type")
     
    delta = time.time() - start
    print "Generated targets in %d:%02.1f" % (delta/60, delta % 60.)
        
    return all_targets
    

def is_standard(star, use_colour_cut=False, standard_frac=0.1, tabtype="fw",
                colour_index="G-J"):
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
                
    Returns
    -------
    is_standard: boolean
        Boolean value indicating whether the star can be considered a standard or not.
    """
    if use_colour_cut:
        # Define the limits for the colour indices used to select A-type stars
        #COLOUR_INDEX_CENTRE = 0
        #COLOUR_INDEX_RANGE = 0.1

        COLOUR_INDEX_MIN = -0.05
        COLOUR_INDEX_MAX = 0.15

        # Calculate the colour indices
        G_minus_J = star["Gaia_G_mag"] - star["2MASS_J_mag"]
        J_minus_K = star["2MASS_J_mag"] - star["2MASS_Ks_mag"]

        # Select as standards only those stars within the allowed colour indices
        if colour_index=="G-J" and (COLOUR_INDEX_MIN <= G_minus_J <= COLOUR_INDEX_MAX):
            is_standard = True
            
        elif colour_index=="J-K" and (COLOUR_INDEX_MIN <= J_minus_K <= COLOUR_INDEX_MAX):
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
# Importing & Generating Science, Standard, and Guide Targets
#-----------------------------------------------------------------------------------------
ra_range = fwts.tiler_input["ra_max"] - fwts.tiler_input["ra_min"]
dec_range = fwts.tiler_input["dec_max"] - fwts.tiler_input["dec_min"]
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
                               fwts.script_settings["standard_frac"])

# Make a copy of all_targets list for use in assigning fibres. For speed, this is a set
candidate_targets = set(all_targets)

# Now create separate lists for standard and guide targets. These will be drawn from the 
# candidate_targets set (i.e. using references to the same objects, rather than copies).

# Standard stars are those that we previously flagged the 'standard' flag for
gj_targets = [t for t in candidate_targets if t.standard==True]

print "%i stars satisfy G-J cut for standards" % len(gj_targets)

jk_targets = [t for t in candidate_targets if t.guide==True]

print "%i stars satisfy J-K cut for standards" % len(jk_targets)