# Utility functions for the simulator to ease tiling operations

import datetime
import logging

import taipan.tiling as tl

from src.resources.v0_0_1.readout.readScience import execute as rScexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readCentroids import execute as rCexec

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec

from src.resources.v0_0_1.delete.deleteTiles import execute as dTexec


def retile_fields(cursor, field_list, tiles_per_field=1,
                  tiling_time=datetime.datetime.now()):
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
    tiling_time:
        Optional; time to record as the tiling (i.e. configuration) time for
        the new tiles. Defaults to datetime.datetime.now().

    Returns
    -------
    Nil. The following will occur:
    - New tiles will be generated in local memory;
    - Redundant tiles will be eliminated from the database;
    - New tiles will be pushed into the database.
    """
    logging.debug('Retiling fields w/ recorded datetime %s') % (
        tiling_time.strftime('%y-%m-%d %H:%M:%S'),
    )

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
    dTexec(cursor, field_list=field_list, obs_status=False)

    # Write the new tiles back to the database
    iTexec(cursor, tile_list, config_time=tiling_time)

    return
