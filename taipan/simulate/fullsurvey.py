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

import src.resources.v0_0_1.manipulate.makeTargetsRemain as mTR

from src.scripts.connection import get_connection

SIMULATE_LOG_PREFIX = 'SIMULATOR: '


def sim_prepare_db(cursor):
    """
    This initial step prepares the database for the simulation run by getting
    the fields in from the database, performing the initial tiling of fields,
    and then returning that information to the database for later use.

    Parameters
    ----------
    cursor

    Returns
    -------

    """

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
    logging.info('First tile pass complete!')

    # 'Pickle' the tiles so they don't need to be regenerated later for tests
    with open('tiles.pobj', 'w') as tfile:
        pickle.dump(candidate_tiles, tfile)

    # Write the tiles to DB
    iTexec(cursor, candidate_tiles)

    # Compute the n_sci_rem and n_sci_obs for these tiles
    mTR.execute(cursor)

    return


def sim_do_night(cursor, date, date_start, date_end,
                 almanac_list=None, dark_almanac_list=None):
    """
    Do a simulated 'night' of observations. This involves:
    - Determine the tiles to do tonight
    - 'Observe' them
    - Update the DB appropriately

    Parameters
    ----------
    cursor:
        The psycopg2 cursor for interacting with the database
    date:
        Python datetime.date object. This should be the local date that the
        night *starts* on, eg. the night of 23-24 July should be passed as
        23 July.
    date_start, date_end:
        The dates the observing run starts and ends on. These are required
        in order to compute the amount of time a certain field will remain
        observable.
    almanac_list:
        List of taipan.scheduling.Almanac objects used for determining
        visibility. Needs to be one per field. If sim_do_night does not
        find an almanac covering the required field and date, one will
        be generated (at added computational cost). Optional.
    dark_almanac_list:
        As for almanac_list, but holds the dark almanacs, which simply
        specify dark or grey time on a per-datetime basis. Optional,
        defaults to None (so the necessary DarkAlmanacs will be created).

    Returns
    -------
    Nil. All actions are internal or apply to the database.

    """
    pass


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

    # TODO: Add check to skip this step if tables already exist
    # Currently dummied out with an is False
    if False:
        sim_prepare_db(cursor)

    fields = rCexec(cursor)
    # Construct the almanacs required
    logging.info('Constructing dark almanac...')
    dark_almanac = ts.DarkAlmanac(date_start, end_date=date_end,
                                  resolution=15.)
    dark_almanac.save()
    logging.info('Constructing field almanacs...')
    almanacs = {field.field_id: ts.Almanac(field.ra, field.dec, date_start,
                                           end_date=date_end, resolution=15.,
                                           minimum_airmass=2)
                for field in fields}

    logging.info('Saving almanacs to disc...')
    for almanac in almanacs.itervalues():
        almanac.save()

    return almanacs, dark_almanac


if __name__ == '__main__':
    # Set the logging to write to terminal
    logging.info('Executing fullsurvey.py as file')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    logging.debug('Doing scripts execute function')
    execute(cursor, None, None)