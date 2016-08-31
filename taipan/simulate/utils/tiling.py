# Utility functions for the simulator to ease tiling operations

import taipan.tiling as tl

from src.resources.v0_0_1.readout.readScience import execute as rScexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readCentroids import execute as rCexec

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec

from src.resources.v0_0_1.manipulate.makeTargetPosn import execute as mTPexec


def retile_fields(cursor, field_list, tiles_per_field=1):
    """
    Re-tile the fields passed.

    Parameters
    ----------
    cursor:
        A psycopg2 cursor for communicating with the database
    field_list:
        A list of fields to be re-tiled. Should be a list of field IDs.
    tiles_per_field:
        Optional int, denoting how many tiles to generate per field. Defaults
        to 1.

    Returns
    -------
    Nil. The following will occur:
    - New tiles will be generated in local memory;
    - Redundant tiles will be eliminated from the database;
    - New tiles will be pushed into the database.
    """
    pass

    # Get the required targets from the database
    candidate_targets = rScexec(cursor, unobserved=True)
    guide_targets = rGexec(cursor)
    standard_targets = rSexec(cursor)

    # Read in prototype tiles for the fields that need re-tiling
    fields_to_tile = rCexec(cursor, field_ids=field_list)

    # Execute a re-tile of the affected fields to the required depth
    tile_list, targets_after_tile = \
        tl.generate_tiling_greedy_npasses(candidate_targets, standard_targets,
                                          guide_targets, tiles_per_field,
                                          tiles=fields_to_tile)

    # Eliminate the redundant tiles
    # TODO: Implement deletion functions

    # Write the tile back to the database
    iTexec(cursor, tile_list)
    mTPexec(cursor, fields=fields_to_tile)

    return
