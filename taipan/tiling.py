#!python

# Module for creating blank TaipanTiles on a survey

# These routines are designed to return parameters for creating a TaipanTile
# based on certain criteria, e.g. maximal target density, minimal target
# density, an even grid of tiles etc.

# For creating a single tile using ra, dec and PA, the correct method is to
# directly call a new TaipanTile object

import core as tp
import time
import random
import math
import numpy as np



# ------
# UTILITY FUNCTIONS
# ------

def compute_bounds(ra_min, ra_max, dec_min, dec_max):
    """
    Compute the bounds for tile generation from user inputs.

    This function is a helper for other functions in this module, to allow
    for ranges which span 0 RA and/or 0 Dec. It returns the min and max values,
    but modified so they can be directly be used as ranges. This means that
    some values will become negative, so will need to be modified by (x % 360.)
    before they can be directly used.

    Inputs:
    ra_min, ra_max -- Min and max RA values of the region to tile, in decimal
        degrees. To have a range that includes 0 degrees RA, either specify
        a negative ra_min, or have ra_min > ra_max. Defaults to None.
    dec_min, dec_max -- Min and max for declination, in decimal 
        degrees. Because of the way declination works,
        dec_min < dec_max by necessity. If this condition is not
        satisfied, dec_min and dec_max will be flipped to make it so.

    Returns:
    ra_min, ra_max, dec_min, dec_max -- The limits modified so they can be used
        as standard ranges.
    """

    if dec_min < -90. or dec_min > 90.0:
        raise ValueError('Min declination must be >= -90.0 and <= 90.0')
    if dec_max < -90. or dec_max > 90.0:
        raise ValueError('Max declination must be >= -90.0 and <= 90.0')

    # Get the entry coordinates into the standard range
    ra_max %= 360.
    if ra_min > ra_max:
        ra_min = (ra_min % 360.0) - 360.0

    dec_min = (dec_min + 90.) % 180. - 90.
    dec_max = (dec_max + 90.) % 180. - 90.
    if dec_min > dec_max:
        dec_min_old = dec_min
        dec_min = dec_max
        dec_max = dec_min_old

    return ra_min, ra_max, dec_min, dec_max


def is_within_bounds(tile, ra_min, ra_max, dec_min, dec_max, 
    compute_bounds_forcoords=True):
    """
    Check if the tile is within the specified bounds.

    Inputs:
    tile -- The TaipanTile instance to check.
    ra_min, ra_max, dec_min, dec_max -- The bounds to check.
    compute_bounds -- Boolean value, denoting whether to use the convert_bounds
        function to ensure the bounds are in standard format. Defaults to True.

    Returns:
    within_bounds -- Boolean value denoting whether the tile centre is within
        the bounds (True) or not (False).
    """
    if compute_bounds_forcoords:
        ra_min, ra_max, dec_min, dec_max = compute_bounds(ra_min, ra_max,
            dec_min, dec_max)

    within_ra = (tile.ra >= ra_min) and (tile.ra <= ra_max)
    # Special case for ra_min < 0
    if ra_min < 0.:
        within_ra = within_ra or (tile.ra - 360. >= ra_min)

    within_dec = (tile.dec >= dec_min) and (tile.dec <= dec_max)

    # print 'Individual RA, dec: %s, %s' % (str(within_ra), str(within_dec))
    within_bounds = within_ra and within_dec
    return within_bounds


# -------
# TILE CREATION FUNCTIONS
# -------

def generate_random_tile(ra_min=0.0, ra_max=360.0,
    dec_min=-90.0, dec_max=90.0, randomise_pa=False):
    """
    Generate a randomly-placed TaipanTile within the constraints provided.

    Inputs:
    ra_min, ra_max -- Min and max RA values of the region to tile, in decimal 
        degrees. To have a range that includes 0 degrees RA, either specify a 
        negative ra_min, or have ra_min > ra_max.
        Defaults to 0.0 deg and 180.0 deg, respectively.
    dec_min, dec_max -- Min and max for declination, in decimal 
        degrees. Because of the way declination works,
        dec_min < dec_max by necessity. If this condition is not
        satisfied, dec_min and dec_max will be flipped to make it so.
        Defaults to -90. and + 90., respectively.
    randomise_pa -- Boolean value denoting whether to randomise the position
        angle of the generated tiles, or use the default PA of 0 degrees.
        Defaults to False.

    Returns:
    tile -- The generated TaipanTile object.
    """

    ra_min, ra_max, dec_min, dec_max = compute_bounds(ra_min, ra_max, 
        dec_min, dec_max)

    # Generate a random RA and dec for the tile centre
    ra_tile = random.uniform(ra_min, ra_max) % 360.
    dec_tile = random.uniform(dec_min, dec_max)
    pa_tile = 0.
    if randomise_pa:
        pa_tile = random.uniform(0., 360.)

    new_tile = tp.TaipanTile(ra_tile, dec_tile, pa=pa_tile)
    return new_tile


def generate_SH_tiling(tiling_file, randomise_seed=True, randomise_pa=False):
    """
    Generate a list of tiles from a Sloane-Harding tiling list.

    Inputs:
    tiling_file -- The text file holding the Sloane-Harding tiling. These
        should be downloaded from http://neilsloane.com/icosahedral.codes/.
    randomise_seed -- Boolean value denoting whether to randomise the location
        of the 'seed' tile, i.e. the tile that would sit at 0 RA, 0 Dec in the
        list of Sloane-Harding tiles. Randomises in RA coordinate only.
        Defaults to True
    randomise_pa -- Boolean value denoting whether to randomise the position
        angle of the generated tiles. Defaults to False.


    Returns:
    tile_list -- A list of TaipanTiles that have been generated from the
        Sloane-Harding tiling.
    """

    with open(tiling_file, 'r') as tiling_fileobj:
        textlines = tiling_fileobj.readlines()

    # Group the tile centres
    tile_cents = [(float(textlines[i*3]), float(textlines[i*3+1]), 
        float(textlines[i*3+2]))
        for i in range(len(textlines) / 3)]
    # Convert X, Y, Z to RA, Dec
    tile_cents = [(np.degrees(math.atan2(c[1], c[0])),
        np.degrees(math.acos(c[2]))- 90.) 
        for c in tile_cents]
    # print tile_cents
    # Randomise positions, if necessary
    if randomise_seed:
        ra_delta = random.uniform(0.0, 180.)
        # print '(%3.1f, %2.1f)' % (ra_delta, dec_delta)
        tile_cents = [((c[0] + ra_delta + 180.) % 360. - 180., c[1])
            for c in tile_cents]
        # print tile_cents[0]
    # print len(tile_cents)

    def gen_pa(randomise_pa):
        if randomise_pa:
            return random.uniform(0., 360.)
        return 0.

    tile_list = [tp.TaipanTile(c[0] + 180., c[1], pa=gen_pa(randomise_pa))
        for c in tile_cents]

    return tile_list


# -------
# TILING FUNCTIONS
# -------

def tiling_consolidate(tile_list):
    """
    Attempt to consolidate a tiling into as few tiles as possible.

    It is conceivable that some tiling algorithms will produce results which
    are sub-optimal in terms of individual tile completeness. This may result
    in a number of tiles with very few science targets which could, in fact,
    be included on other, more populated tiles. This function attempts to 
    'consolidate' a tiling by shifting targets off poorly-complete tiles and
    on to more complete ones.

    Inputs:
    tile_list -- The list of TaipanTile objects that constitute the tiling.

    Returns:
    consolidated_list -- The list of TaipanTile objects representing the
        consolidation of tile_list. consolidated_list will NOT preserve the
        ordering in tile_list.
    """

    print 'Consolidating tiling...'
    # Strip off any tiles that have no science targets assigned
    tiles_orig = len(tile_list)
    tile_list = [t for t in tile_list if t.count_assigned_targets_science() > 0]
    tiles_removed = tiles_orig - len(tile_list)
    # Sort the input tiling list by number of targets
    # Most-complete tiles appear at the start of the list
    tile_list.sort(key=lambda x: -1 * x.count_assigned_fibres())

    # We can reduce effort now by starting off the consolidated_list with all
    # those tiles that are already 100% complete
    consolidated_list = [t for t in tile_list 
        if t.count_assigned_fibres() == tp.FIBRES_PER_TILE]
    tile_list = tile_list[len(consolidated_list):]

    # Step through the tile list, attempting to re-assign targets to the more-
    # complete tiles
    # Stop once we run out of tiles to check
    targets_moved = 0
    while len(tile_list) > 0:
        print 'Remaining tiles to consolidate: %d' % len(tile_list)
        # Grab the targets out of the lowest-completeness tile
        targets_to_redo = tile_list[-1].get_assigned_targets_science(
            return_dict=True)
        # Try to assign these targets to another, more-complete tile
        # Be sure not to try re-assignment to the current worst tile!
        for (fibre, target) in targets_to_redo.iteritems():
            tiles_to_try = [t for t in tile_list[:-1] 
                if target.dist_point((t.ra, t.dec)) < tp.TILE_RADIUS]
            target_reassigned = False
            while len(tiles_to_try) > 0 and target_reassigned == False:
                # Attempt to assign target to tile
                targets_returned, removed_targets = tiles_to_try[0].assign_tile(
                    [target], check_tile_radius=False, 
                    recompute_difficulty=False,
                    overwrite_existing=False,
                    method='priority')
                if len(targets_returned) == 0:
                    # Target has been re-assigned
                    target_reassigned = True
                    targets_moved += 1
                    removed_target = tile_list[-1].unassign_fibre(fibre)
                else:
                    # Remove the top-ranked candidate tile and go again
                    removed_tile = tiles_to_try.pop(0)
        # We have now been through all the targets to try and re-assign from
        # this tile. There are now two options:
        # If all science targets have been re-assigned, we can burn this tile
        # Otherwise, the tile needs to be added to the consolidated_list
        if tile_list[-1].count_assigned_targets_science() == 0:
            clipped_tile = tile_list.pop(-1)
            tiles_removed += 1
        else:
            consolidated_list.append(tile_list.pop(-1))

    print '%d targets shifted, %d tiles removed' % (targets_moved, 
        tiles_removed,)
    return consolidated_list



def generate_tiling_byorder(candidate_targets, standard_targets, guide_targets,
    completeness_target = 1.0,
    tiling_method='SH', randomise_pa=True,
    tiling_order='random',
    randomise_SH=True, tiling_file='ipack.3.8192.txt',
    ra_min=0.0, ra_max=360.0, dec_min=-90.0, dec_max=90.0,
    tiling_set_size=1000,
    tile_unpick_method='sequential', combined_weight=1.0, 
    sequential_ordering=(1,2), rank_supplements=False,
    repick_after_complete=True, recompute_difficulty=True):
    """
    Generate a complete tiling based on a 'by-roder' algorithm.

    This algorithm will completely fill one tile before moving
    on to the next tile in the sequence. This function will create tiles,
    attempt to completely populate them, and repeat until the requested
    completeness_target has been reached.

    There are several options available for the generation of tiles:
    'SH' -- Sloane-Harding tiling centres. In this method, a full grid of
        SH tiles are generated, picked in a greedy fashion, and then
        consolidated. This procedure is repeated until the completeness_target
        is reached.
    'random' -- A tile is randomly generated within the specified RA and Dec
        limits, and then picked. The process is repeated until the
        completeness_target is reached.
    'random-set' -- As for 'random', but tiling_set_size tiles are generated
        at once.
    'random-target' -- A tile is centred on a randomly-selected remaining
        science target and is unpicked. Process is repeated until the
        completeness_target is reached.
    'random-target-set' -- As for 'random-target', but tiling_set_size tiles
        are generated at once.
    'average' -- A tile is generated at the average RA, Dec of the remaining
        science targets (this is a computationally cheap way of finding the
        location of highest remaining target density). The tile is then
        unpicked. The process repeats until completeness_target is reached,
        or until a tile cannot have science targets assigned to it (i.e. the
        average position contains no targets), and which point the tiling_method
        is switched to random_target.

    Inputs:
    candidate_targets -- The list of TaipanTargets (science) to tile. Each
        target in the list will appear once somewhere in the tiling, unless the
        completeness_target is reached first.
    guide_targets, standard_targets -- Guide and standard TaipanTargets to
        assign to the tilings. These may be repeated across tiles.
    completeness_target -- A float in the range (0, 1] denoting what level of
        completeness is required to be achieved before the tiling can be 
        considered complete. Defaults to 1.0 (that is, all science targets 
        must be assigned).
    tiling_method -- String denoting which tiling method to use (see above). 
        Defaults to 'SH' (Sloane-Harding tile distribution.)
    randomise_pa -- Boolean value, denoting whether to randomise the PA of the
        generated tiles. Defaults to True.
    tiling_order -- String denoting the order in which to attempt to unpick
        tiles. Only has an effect if tiling_method = 'SH', 'random-set' or
        'random-target-set'. May have one of the following values:
        random - Randomised order
        density - Tiles with the highest number of candidates will be tiled 
            first. 
        priority - Tiles with the highest cumulative target priority will be
            tiled first.
    randomise_SH -- Boolean value denoting whether to randomise the RA position
        of the 'seed' tile in the Sloane-Harding tiling. Only has an effect if
        tiling_method='SH'. Defaults to True.
    tiling_file -- String containing the filepath to the Sloane-Harding tiling
        to use if tiling_method = 'SH'. Defaults to 'ipack.3.8192.txt',
        which is the best-coverage tiling for Taipan-sized tiles.
    ra_min, ra_max, dec_min, dec_max -- The min/max values for tile centre RA
        and Dec. Defaults to 0., 360., -90. and 90., respectively.
    tiling_set_size -- The number of tiles to generate at a time for the 
        random-set and random-target-set tiling methods. Defaults to 1000.
    tile_unpick_method, combined_weight, sequential_ordering, rank_supplements,
    repick_after_complete -- Values to pass to the tile's unpick_tile method for
        target assignment. See the documentation for taipan.core for the meaning
        and limits of these values.

    Returns:
    tile_list -- The list of TaipanTiles corresponding to the tiling generated.
    completeness -- A float in the range [0, 1] describing the level of
        completeness achieved, that is, the percentage of targets successfully
        assigned.
    remaining_targets -- The list of science TaipanTargets that were not
        assigned during this tiling.
    """

    TILING_METHODS = [
        'SH',               # Sloane-Harding
        'random',           # Randomised position
        'random-set',       # Create multiples at once
        'random-target',    # Random target used as tile centre
        'random-target-set',# Create multiples at once
        'average',          # Tile position based on avg RA, Dec of targets
    ]
    if tiling_method not in TILING_METHODS:
        raise ValueError('tiling_method must be one of %s' 
            % str(TILING_METHODS))
    TILING_METHODS_SET = [TILING_METHODS[i] for i in [0, 2, 4]]
    TILING_ORDERS = [
        'random',
        'density', 
        'priority',
    ]
    if tiling_order not in TILING_ORDERS:
        raise ValueError('tiling_order must be one of %s'
            % str(TILING_ORDERS))

    tiling_set_size = int(tiling_set_size)
    if tiling_set_size <= 0:
        raise ValueError('tiling_set_size must be > 0')

    if completeness_target <= 0. or completeness_target > 1:
        raise ValueError('completeness_target must be in the range (0, 1]')

    # Push the coordinate limits into standard format
    ra_min, ra_max, dec_min, dec_max = compute_bounds(ra_min, ra_max,
        dec_min, dec_max)

    # Store the number of originally-submitted targets so we can calculate
    # the completeness achieved
    no_submitted_targets = len(candidate_targets)
    prior_tiles = []

    # Define helper function to handle randomising PAs if tile generation
    # doesn't already have it built in
    def gen_pa(randomise_pa):
        if randomise_pa:
            pa = random.uniform(0., 360.)
            return pa
        return 0.

    # Do the tiling while completeness is < the target
    print 'Commencing tiling, %d targets...' % no_submitted_targets
    while (float(no_submitted_targets - len(candidate_targets)) 
        / float(no_submitted_targets)) < completeness_target:
        # Generate the next tile(s) to unpick

        if tiling_method == 'SH':
            new_tiles = generate_SH_tiling(tiling_file, 
                randomise_seed=randomise_SH, randomise_pa=randomise_pa)
        elif tiling_method == 'random':
            new_tiles = generate_random_tile(ra_min=ra_min, ra_max=ra_max,
                dec_min=dec_min, dec_max=dec_max, randomise_pa=randomise_pa)
            # print (new_tiles.ra, new_tiles.dec)
            new_tiles = [new_tiles]
        elif tiling_method == 'random-set':
            new_tiles = [generate_random_tile(ra_min=ra_min, ra_max=ra_max,
                dec_min=dec_min, dec_max=dec_max, randomise_pa=randomise_pa)
                for i in range(tiling_set_size)]
        elif tiling_method == 'random-target':
            random_tgt = random.choice(candidate_targets)
            new_tiles = tp.TaipanTile(random_tgt.ra, random_tgt.dec, 
                pa=gen_pa(randomise_pa))
            new_tiles = [new_tiles]
        elif tiling_method == 'random-target-set':
            new_tiles = []
            for i in range(tiling_set_size):
                random_tgt = random.choice(candidate_targets)
                new_tiles.append(tp.TaipanTile(random_tgt.ra, random_tgt.dec, 
                    pa=gen_pa(randomise_pa)))
        elif tiling_method == 'average':
            new_tiles = tp.TaipanTile(np.average([t.ra 
                for t in candidate_targets]),
                np.average([t.dec for t in candidate_targets]), 
                pa=gen_pa(randomise_pa))
            new_tiles = [new_tiles]

        if tiling_method in TILING_METHODS_SET:
            # Trim down to the requested RA/Dec limits
            # print ra_min, ra_max, dec_min, dec_max
            # print len(new_tiles)
            # for t in new_tiles:
            #   print t.ra, t.dec
            new_tiles = [t for t in new_tiles 
                if is_within_bounds(t, ra_min, ra_max, dec_min, dec_max,
                    compute_bounds_forcoords=True)]
            # Order the tiles as requested
            if tiling_order == 'random':
                random.shuffle(new_tiles)
            elif tiling_order == 'density':
                new_tiles.sort(key=lambda x: len(x.available_targets(
                    candidate_targets)))
            elif tiling_order == 'priority':
                new_tiles.sort(key=lambda x: np.sum([t.priority for t
                    in x.available_targets(candidate_targets)]))

        # We now need to unpick the tile(s) we have just created, using
        # existing functions
        targets_before = len(candidate_targets)
        print 'Beginning to tile %d tiles, %d targets...' % (len(new_tiles),
            targets_before, )
        i = 0
        for tile in new_tiles:
            candidate_targets, removed_targets = tile.unpick_tile(
                candidate_targets, standard_targets, guide_targets,
                overwrite_existing=False, check_tile_radius=True,
                method=tile_unpick_method, combined_weight=combined_weight,
                sequential_ordering=sequential_ordering,
                rank_supplements=rank_supplements,
                repick_after_complete=repick_after_complete,
                recompute_difficulty=recompute_difficulty)
            i += 1
            print 'Tile %d complete...' % i
        print 'Tiling complete!'
        # If we are using 'random' or 'average' tiling_method, and no targets
        # have been successfully assigned, switch over to 'random-target' method
        # and return to the top of the loop
        if tiling_method in ['random', 'average'] and len(
            candidate_targets) == targets_before:
            print 'Failure detected in %s mode - switching to random-target mode' % tiling_method
            tiling_method = 'random-target'
            continue

        # Combine the new tiles with existing ones
        prior_tiles += new_tiles

        # If using a 'set'/'SH' tiling method, consolidate the tiling
        if tiling_method in TILING_METHODS_SET:
            prior_tiles = tiling_consolidate(prior_tiles)

    if tiling_method not in TILING_METHODS_SET:
        prior_tiles = tiling_consolidate(prior_tiles)

    # Return the tiling, the completeness factor and the remaining targets
    final_completeness = float(no_submitted_targets 
        - len(candidate_targets)) / float(no_submitted_targets)

    return prior_tiles, final_completeness, candidate_targets


def generate_tiling_greedy(candidate_targets, standard_targets, guide_targets,
    completeness_target = 1.0,
    ranking_method='completeness',
    disqualify_below_min=True,
    tiling_method='SH', randomise_pa=True,
    randomise_SH=True, tiling_file='ipack.3.8192.txt',
    ra_min=0.0, ra_max=360.0, dec_min=-90.0, dec_max=90.0,
    tiling_set_size=1000,
    tile_unpick_method='sequential', combined_weight=1.0, 
    sequential_ordering=(1,2), rank_supplements=False,
    repick_after_complete=True, recompute_difficulty=True):
    """
    Generate a tiling based on the greedy algorithm.

    The greedy algorithm works as follows:
    - Generate a set of tiles covering the area of interest.
    - Unpick each tile, allowing for target duplication between tiles.
    - Select the 'best' tile from this set, and add it to the resultant
        tiling.
    - Replace the removed tile, then re-unpick the tiles in the set which are
        affected by the removal of the tile;
    - Repeat until no useful tiles remain, or the completeness target is
        reached.

    Inputs:
    candidate_targets, standard_targets, guide_targets -- The lists of science,
        standard and guide targets to consider, respectively. Should be lists
        of TaipanTarget objects.
    completeness_target -- A float in the range (0, 1] denoting the science
        target completeness to stop at. Defaults to 1.0 (full completeness).
    ranking_method -- The scheme to use for ranking the tiles. See the
        documentation for TaipanTile.calculate_tile_score for details.
    tiling_method -- The method by which to generate a tiling set. Currently,
        only 'SH' (Sloane-Harding tiling centres) are available.
    randomise_pa -- Optional Boolean, denoting whether to randomise the pa of
        seed tiles or not. Defaults to True.
    randomise_SH -- Optional Boolean, denoting whether or not to randomise the
        RA of the 'seed' of the SH tiling. Defaults to True.
    tiling_file -- The SH tiling file to use for generating tiling centres.
        Defaults to 'ipack.3.8192.txt'.
    ra_min, ra_max, dec_min, dec_max -- The RA and Dec bounds of the region to
        be considered, in decimal degrees. To have an RA range spanning across 
        0 deg RA, either use a negative value for ra_min, or give an ra_min >
        ra_max.
    tiling_set_size -- Not relevant at the current time.
    tile_unpick_method -- The scheme to be used for unpicking tiles. Defaults to
        'sequential'. See the documentation for TaipanTile.unpick_tile for 
        details.
    combined_weight, sequential_ordering -- Additional arguments to be used in
        the tile unpicking process. See the documentation for 
        TaipanTile.unpick_tile for details.
    rank_supplements -- Optional Boolean value, denoting whether to attempt to
        assign guides/standards in priority order. Defaults to False.
    repick_after_complete -- Boolean value, denoting whether to repick each tile
        after unpicking. Defaults to True.
    recompute_difficulty -- Boolean value, denoting whether to recompute target
        difficulties after a tile is moved to the results lsit. Defaults to
        True.

    Returns:
    tile_list -- The list of tiles making up the tiling.
    final_completeness -- The target completeness achieved.
    candidate_targets -- Any targets from candidate_targets that do not
        appear in the final tiling_list (i.e. were not assigned to a successful
        tile).
    """
    
    tile_list = []

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

    # Unpick ALL of these tiles
    # Note that we are *not* updating candidate_targets during this process,
    # as overlap is allowed - instead, we will need to manually update
    # candidate_tiles once we pick the highest-ranked tile
    # Likewise, we don't want the target difficulties to change
    # Therefore, we'll assign the output of the function to a dummy variable
    print 'Creating initial tile unpicks...'
    i = 0
    # print len(candidate_targets)
    candidate_targets_master = candidate_targets[:]
    for tile in candidate_tiles:
        # print 'inter: %d' % len(candidate_targets)
        burn = tile.unpick_tile(candidate_targets, standard_targets, 
            guide_targets,
            overwrite_existing=True, check_tile_radius=True,
            recompute_difficulty=False,
            method=tile_unpick_method, combined_weight=combined_weight,
            sequential_ordering=sequential_ordering,
            rank_supplements=rank_supplements, 
            repick_after_complete=repick_after_complete,
            consider_removed_targets=False)
        i += 1
        print 'Created %d / %d tiles' % (i, len(candidate_tiles))
    # print len(candidate_targets)

    # Initialise some of our counter variables
    no_submitted_targets = len(candidate_targets)
    # Compute initial rankings for all of the tiles
    ranking_list = [tile.calculate_tile_score(method=ranking_method,
        disqualify_below_min=disqualify_below_min) for tile in candidate_tiles]
    # print ranking_list

    # Define a helper function
    def gen_pa(randomise_pa):
        pa = 0.
        if randomise_pa:
            pa = random.uniform(0., 360.)
        return pa

    # While we are below our completeness criteria AND the highest-ranked tile
    # is not empty, perform the greedy algorithm
    print 'Starting greedy tiling allocation...'
    i = 0
    while ((float(no_submitted_targets - len(candidate_targets)) 
        / float(no_submitted_targets)) < completeness_target) and (
        max(ranking_list) > 0.05):

        # Find the highest-ranked tile in the candidates_list, and remove it
        # print 'a : %d' % len(candidate_targets)
        i = np.argmax(ranking_list)
        tile_list.append(candidate_tiles.pop(i))
        best_ranking = ranking_list.pop(i)
        print 'Tile selected!'
        # Record the ra and dec of the candidate for tile re-creation
        best_ra = tile_list[-1].ra
        best_dec = tile_list[-1].dec
        # print 'b : %d' % len(candidate_targets)

        # Force a wait here to see if it solves the target problem
        # time.sleep(5)

        # Strip the now-assigned targets out of the candidate_targets list,
        # then recalculate difficulties for affected remaning targets
        print 'Re-computing target list...'
        # print 'c : %d' % len(candidate_targets)
        # ERROR: Something goes wrong here with the target reduction -- not all
        # of the assigned targets appear to be removed from candidate_targets
        # It works correctly for the first pass or two, and then start to not
        # work correctly
        # What's odd is that all of these variations on stripping the assigned
        # targets fail, but in the return test_tiling, ALL of the objects
        # within the tiling are members of the originally passed master
        # list of targets
        assigned_targets = tile_list[-1].get_assigned_targets_science()
        # targets_not_in_cands = [t for t in assigned_targets if t not in
        #   candidate_targets]

        # print assigned_targets
        before_targets_len = len(candidate_targets)
        for t in assigned_targets:
            candidate_targets.pop(candidate_targets.index(t))

        if len(set(assigned_targets)) != len(assigned_targets):
            print '### WARNING: target duplication detected'
        if len(candidate_targets) != before_targets_len - len(assigned_targets):
            print '### WARNING: Discrepancy found in target list reduction'
            print 'Best tile had %d targets; only %d removed from list' % (
                len(assigned_targets), before_targets_len 
                - len(candidate_targets))
            print 'I have %d assigned targets apparently not in master list' % (
                len(targets_not_in_cands))
        if recompute_difficulty:
            print 'Re-computing target difficulties...'
            tp.compute_target_difficulties(tp.targets_in_range(
                best_ra, best_dec, candidate_targets,
                tp.TILE_RADIUS+tp.FIBRE_EXCLUSION_RADIUS))
        # print 'e : %d' % len(candidate_targets)

        # Replace the removed tile in candidate_targets, repick any tiles
        # within 2 * TILE_RADIUS of it, and then add to the ranking_list
        candidate_tiles.append(tp.TaipanTile(best_ra, best_dec, pa=gen_pa(
            randomise_pa)))
        j = 0
        print 'Re-picking affected tiles...'
        # print 'f : %d' % len(candidate_targets)
        affected_tiles = [t for t in candidate_tiles
            if np.any(map(lambda x: x in t.get_assigned_targets_science(),
                assigned_targets))]
        # This won't cause the new tile to be re-picked, so manually add that
        affected_tiles.append(candidate_tiles[-1])
        for tile in affected_tiles:
            # print 'inter: %d' % len(candidate_targets)
            burn = tile.unpick_tile(candidate_targets, standard_targets, 
                guide_targets,
                overwrite_existing=True, check_tile_radius=True,
                recompute_difficulty=False,
                method=tile_unpick_method, combined_weight=combined_weight,
                sequential_ordering=sequential_ordering,
                rank_supplements=rank_supplements, 
                repick_after_complete=repick_after_complete,
                consider_removed_targets=False)
            j += 1
            print 'Completed %d / %d' % (j, len(affected_tiles))
        # print 'g : %d' % len(candidate_targets)
        ranking_list = [tile.calculate_tile_score(method=ranking_method,
            disqualify_below_min=disqualify_below_min) 
            for tile in candidate_tiles]
        # print ranking_list
        # print [len(t.get_assigned_targets_science()) for t in candidate_tiles]

        print 'Assigned tile at %3.1f, %2.1f' % (best_ra, best_dec)
        print 'Tile has ranking score %3.1f' % (best_ranking, )
        print '%d targets, %d standards, %d guides' % (
            tile_list[-1].count_assigned_targets_science(),
            tile_list[-1].count_assigned_targets_standard(),
            tile_list[-1].count_assigned_targets_guide(), )
        print 'Now assigned %d tiles' % (len(tile_list), )
        print 'Completeness achieved: %1.4f' % (float(no_submitted_targets 
        - len(candidate_targets)) / float(no_submitted_targets))
        print 'Remaining targets: %d' % len(candidate_targets)
        print 'Remaining guides & standards: %d, %d' % (len(guide_targets),
            len(standard_targets))

        # If the max of the ranking_list is now 0, try switching off 
        # the disqualify flag
        if max(ranking_list) < 0.05 and disqualify_below_min:
            print 'Detected no remaining legal tiles - relaxing requirements'
            disqualify_below_min = False
            ranking_list = [tile.calculate_tile_score(method=ranking_method,
                disqualify_below_min=disqualify_below_min) 
                for tile in candidate_tiles]
            print ranking_list

    # Consolidate the tiling
    tile_list = tiling_consolidate(tile_list)
    # print ranking_list

    # Return the tiling, the completeness factor and the remaining targets
    final_completeness = float(no_submitted_targets 
        - len(candidate_targets)) / float(no_submitted_targets)

    if not repick_after_complete:
        # Do a global re-pick, given we didn't do it on the fly
        for t in tile_list:
            t.repick_tile()

    return tile_list, final_completeness, candidate_targets
