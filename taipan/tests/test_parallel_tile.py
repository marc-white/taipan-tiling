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
        filename='./testlog_parallel_tiling.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('TESTING PARALLEL RE-TILE')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    # Get the targets and fields
    logging.warning('Getting DB info...')
    tgts = readScience.execute(cursor)
    gds = readGuides.execute(cursor)
    stds = readStandards.execute(cursor)
    fields = readCentroids.execute(cursor)
    logging.warning('...done!')

    # Do the tests
    for workers in [10, 3, 1]:
        logging.warning('Testing with %d workers...' % workers)
        for t in [1, 10, 50]:
            start = datetime.datetime.now()

            # Select some tiles
            tiles = random.sample(fields, t)
            a, b = generate_tiling_greedy_npasses(tgts, stds, gds, 1,
                                                  tiles=tiles,
                                                  sequential_ordering=(2,1),
                                                  recompute_difficulty=True,
                                                  repeat_targets=True,
                                                  repick_after_complete=False,
                                                  multicores=workers)

            end = datetime.datetime.now()
            delta = end - start
            logging.warning('Tiled %d tiles w/ %d workers in %4.1f s' % (
                t, workers, delta.total_seconds(),
            ))
        logging.warning('...done with % workers!' % workers)
