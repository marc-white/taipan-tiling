# Test the ability to do parallel re-tiling

from src.resources.v0_0_1.readout import readScience, readGuides, \
    readStandards, readCentroids
from src.scripts.connection import get_connection

from taipan.tiling import generate_tiling_greedy_npasses

import datetime
import random
import logging
import sys
import traceback
import multiprocessing

import numpy as np

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
        level=logging.WARNING,
        filename='./verifylog_parallel_tiling.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('VERIFYING PARALLEL RE-TILE')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    logging.warning('Getting DB info...')
    fields = readCentroids.execute(cursor)
    tiles = random.sample(fields, 50)
    tgts = readScience.execute(cursor,
                               field_list=[_.field_id for _ in tiles])
    gds = readGuides.execute(cursor,
                             field_list=[_.field_id for _ in tiles])
    stds = readStandards.execute(cursor,
                                 field_list=[_.field_id for _ in tiles])
    logging.warning('...done!')

    logging.warning('Generating single-threaded tiling...')

    tgts_in = tgts[:]
    gds_in = gds[:]
    stds_in = stds[:]

    random.seed(8574)
    np.random.seed(8574)

    tiles_single, _ = generate_tiling_greedy_npasses(tgts_in, stds_in, gds_in,
                                                     1,
                                                     tiles=tiles,
                                                     sequential_ordering=(2, 1),
                                                     recompute_difficulty=True,
                                                     repeat_targets=True,
                                                     repick_after_complete=
                                                     False,
                                                     multicores=1)

    logging.warning('...done!')
    logging.warning('Generating multi-threaded tiling...')

    tgts_in = tgts[:]
    gds_in = gds[:]
    stds_in = stds[:]

    random.seed(8574)
    np.random.seed(8574)

    tiles_parall, _ = generate_tiling_greedy_npasses(tgts_in, stds_in, gds_in,
                                                     1,
                                                     tiles=tiles,
                                                     sequential_ordering=(2, 1),
                                                     recompute_difficulty=True,
                                                     repeat_targets=True,
                                                     repick_after_complete=
                                                     False,
                                                     multicores=
                                                     max(1,
                                                         int(0.8 *
                                                             multiprocessing.
                                                             cpu_count())))

    logging.warning('...done!')

    if len(tiles_single) != len(tiles_parall):
        raise RuntimeError('Something has gone wrong - length of tile lists '
                           'do not match!')

    # Ensure the tile lists are ordered so they match
    tiles_single.sort(key=lambda x: x.field_id)
    tiles_parall.sort(key=lambda x: x.field_id)

    def get_sky_fibre_id(tile):
        return set([k for k, v in tile._fibres if v == 'sky'])

    def get_assigned_target_ids(tile):
        return set([t.idn for t in tile.get_assigned_targets()])

    problem_found = False

    for i in range(len(tiles_single)):
        if get_sky_fibre_id(tiles_single[i]) != \
                get_sky_fibre_id(tiles_parall[i]):
            logging.warning('Mismatch between sky assignments for field %d' %
                            tiles_single[i].field_id)
            problem_found = True

        if get_assigned_target_ids(tiles_single[i]) != \
                get_assigned_target_ids(tiles_parall[i]):
            logging.warning('Mismatch between target (of some type) '
                            'assignments for field %d' %
                            tiles_single[i].field_id)
            problem_found = True

        logging.warning('Checked field %d' % tiles_single[i].field_id)

    if problem_found:
        logging.critical('PROBLEMS IDENTIFIED')
    else:
        logging.warning('No issues detected!')
