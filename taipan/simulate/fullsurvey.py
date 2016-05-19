# Simulate a full tiling of the Taipan galaxy survey

import sys
import logging
import taipan.core as tp
import taipan.tiling as tl
import taipan.scheduling as ts
import simulate as tsim

import pickle

import numpy as np
import atpy
import ephem
import random

from src.resources.v0_0_1.readout.readCentroids import execute as rCexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readScience import execute as rScexec

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec


def execute(cursor, date_start, date_end, output_loc='.'):
    """
    Execute the simulation
    Parameters
    ----------
    cursor:
        psycopg2 cursor for communicating with the database.
    output_loc:
        String providing the path for placing the output plotting images.
        Defaults to '.' (ie. the present working directory). Directory must
        already exist.

    Returns
    -------
    Nil. Tiling outputs are written to the database (to simulate the action of
    the virtual observer), and plots are generated and placed in the output
    folder.
    """

    SIMULATE_LOG_PREFIX = 'SIMULATOR: '

    # This is just a rough scaffold to show the steps the code will need to
    # take

    # construct_league_table()
    # read_league_table()
    #
    # generate_initial_tiles()
    # write_tiles_to_db() # This creates the league table
    #
    # # DO EITHER:
    # date_curr = date_start
    # while date_curr < date_end:
    #     observe_night() # This will select & 'observe' tiles,
    #     # return the tiles observed
    #     manipulate_targets() # Update flags on successfully observe targets
    #     retile_fields () # Retile the affected fields
    #     curr_date += 1 # day
    #
    # # OR DO THIS INSTEAD:
    # observe_timeframe(date_start, date_end)
    # # This function will handle all of the above, but re-tile after each
    # # observation (and do all necessary DB interaction). This will be faster,
    # # as all the target handling can be done internally without
    # # reading/writing DB, *but* the function that do that then won't be
    # # prototyped
    #
    # read_in_observed_tiles()
    # generate_outputs()

    # Ge the field centres in from the database
    logging.info(SIMULATE_LOG_PREFIX+'Loading targets')
    field_tiles = rCexec(cursor)
    candidate_targets = rScexec(cursor)
    guide_targets = rGexec(cursor)
    standard_targets = rSexec(cursor)

    logging.info(SIMULATE_LOG_PREFIX+'Generating first pass of tiles')
    # TEST ONLY: Trim the tile list to 10 to test DB write-out
    # field_tiles = random.sample(field_tiles, 40)
    candidate_tiles = tl.generate_tiling_greedy_npasses(candidate_targets,
                                                        standard_targets,
                                                        guide_targets,
                                                        1,
                                                        tiles=field_tiles,
                                                        )

    # 'Pickle' the tiles so they don't need to be regenerated later for tests
    with open('candidate_targets.pobj', 'w') as cfile:
        pickle.dump(candidate_targets, cfile)
    with open('standard_targets.pobj', 'w') as sfile:
        pickle.dump(standard_targets, sfile)
    with open('guide_targets.pobj', 'w') as gfile:
        pickle.dump(guide_targets, gfile)
    with open('tiles.pobj', 'w') as tfile:
        pickle.dump(candidate_tiles, tfile)

    # Write the tiles to DB
    iTexec(cursor, candidate_tiles)

    return


if __name__ == '__main__':
    # Get a cursor
    # TODO: Correct package imports & references
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    execute()