# Simulate a full tiling of the Taipan galaxy survey

import sys
import logging
import taipan.core as tp
import taipan.tiling as tl
import taipan.scheduling as ts
import simulate as tsim

import numpy as np
import atpy
import ephem

from TaipanDB.src.resources.v0_0_1.readout import *
from TaipanDB.src.resources.v0_0_1.manipulate import *


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

    # Ge the field centres in from the database
    field_tiles = loadCentroids.execute(cursor)



if __name__ == '__main__':
    # Get a cursor
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    execute()