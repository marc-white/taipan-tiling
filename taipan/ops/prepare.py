"""
Prepare the Taipan system for observing.

This module contains functions designed to be run before observing
begins. It creates an initial tiling that functions within
:any:`taipan.ops.planner` can schedule.
"""

import logging
import pickle
import datetime
import sys
import traceback

import taipan.tiling as tl

from taipandb.resources.stable.readout.readCentroids import execute as rCexec
from taipandb.resources.stable.readout.readScience import execute as rScexec
from taipandb.resources.stable.readout.readGuides import execute as rGexec
from taipandb.resources.stable.readout.readStandards import execute as rSexec
from taipandb.resources.stable.readout.readSkies import execute as rSkexec

from taipandb.resources.stable.insert.insertTiles import execute as iTexec

from taipandb.resources.stable.manipulate.makeNSciTargets import execute as mNScT

from taipandb.scripts.connection import get_connection


def do_initial_tile(
        cursor=None,
        assign_sky_fibres=True,
        commit=True,
        output_tiles=True):
    """
    Do an initial tiling, based on the targets available in the database.

    Parameters
    ----------
    cursor : :obj:`psycopg2.connection.cursor`
        Cursor for database communication. Defaults to :obj:`None`.
    assign_sky_fibres : :obj:`bool`, default ``True``
        Should sky fibres be assigned to tiles?
    commit : :obj:`bool`, default ``True``
        Should the changes to the database be committed?
    output_tiles : :obj:`bool`, default ``True``
        If ``True``, the tiles generated are saved using :any:`pickle` to a
        file called ``tiles.pobj`` in the present working directory.
    """
    # Read in the various targets
    logging.info('Loading targets')
    field_tiles = rCexec(cursor)
    candidate_targets = rScexec(cursor)
    guide_targets = rGexec(cursor)
    standard_targets = rSexec(cursor)
    if assign_sky_fibres:
        sky_targets = rSkexec(cursor)

    # Tile them together
    candidate_tiles, targets_remain = tl.generate_tiling_greedy_npasses(
        candidate_targets, standard_targets, guide_targets,
        1,  # npasses
        sky_targets=sky_targets if assign_sky_fibres else None,
        tiles=field_tiles, repeat_targets=True,
        tile_unpick_method='sequential',
        sequential_ordering=(2, 1),
        multicores=7,
    )

    # Write outputs to file if necessary
    if output_tiles:
        with open('tiles.pobj', 'w') as tfile:
            pickle.dump(candidate_tiles, tfile)

    # Write tiles to database
    iTexec(cursor, candidate_tiles, config_time=datetime.datetime.now(),
           disqualify_below_min=False, remove_index=True)

    # Compute the n_sci_rem and n_sci_obs for these tiles
    # mTPexec(cursor)
    # Given all fields have been tiled, don't need to specify which fields to
    # compute for
    mNScT.execute(cursor)

    # Commit if requested
    if commit:
        cursor.connection.commit()

    return

if __name__ == '__main__':
    # Override the sys.excepthook behaviour to log any errors
    # http://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def excepthook_override(exctype, value, tb):
        # logging.error(
        #     'My Error Information\nType: %s\nValue: %s\nTraceback: %s' %
        #     (exctype, value, traceback.print_tb(tb), ))
        # logging.error('Uncaught error/exception detected',
        #               exctype=(exctype, value, tb))
        logging.critical(''.join(traceback.format_tb(tb)))
        logging.critical('{0}: {1}'.format(exctype, value))
        # logging.error('Type:', exctype)
        # logging.error('Value:', value)
        # logging.error('Traceback:', tb)
        return


    sys.excepthook = excepthook_override

    # Set the logging to write to terminal AND file
    logging.basicConfig(
        level=logging.INFO,
        filename='./tiling.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.info('*** COMMENCING INITIAL TILING')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()
    logging.debug('Doing do_initial_tile function')
    do_initial_tile(cursor, assign_sky_fibres=True,
                    commit=True, output_tiles=True)