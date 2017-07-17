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
        filename='./testlog_parallel_tiling.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('TESTING PARALLEL RE-TILE')
    logging.warning('NOTE: Your system has %d cores' %
                    multiprocessing.cpu_count())

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    workers_values = [1, 4,
                        int(0.8*multiprocessing.cpu_count()),
                        multiprocessing.cpu_count(), ]
    no_tiles = [1, 10, 50]

    # Construct the results array
    results_array = np.zeros((len(workers_values), len(no_tiles), ),
                             dtype=object)
    for i in range(len(workers_values)):
        for j in range(len(no_tiles)):
            results_array[i, j] = []


    # Get the targets and fields
    for i in range(5):
        logging.warning('Getting DB info...')
        fields = readCentroids.execute(cursor)
        fields = random.sample(fields, 50)
        tgts = readScience.execute(cursor,
                                   field_list=[_.field_id for _ in fields])
        gds = readGuides.execute(cursor,
                                 field_list=[_.field_id for _ in fields])
        stds = readStandards.execute(cursor,
                                     field_list=[_.field_id for _ in fields])
        logging.warning('...done!')

        # Do the tests
        for w in range(len(workers_values)):
            workers = workers_values[w]
            logging.warning('Testing with %d workers...' % workers)
            for t in range(len(no_tiles)):
                tile = no_tiles[t]
                logging.warning('Testing with %d input tiles...' % tile)
                start = datetime.datetime.now()

                # Select some tiles
                tiles = random.sample(fields, tile)
                a, b = generate_tiling_greedy_npasses(tgts, stds, gds, 1,
                                                      tiles=tiles,
                                                      sequential_ordering=(2,1),
                                                      recompute_difficulty=True,
                                                      repeat_targets=True,
                                                      repick_after_complete=
                                                      False,
                                                      multicores=workers)

                end = datetime.datetime.now()
                delta = end - start
                logging.warning('Tiled %d tiles w/ %d workers in %4.1f s' % (
                    t, workers, delta.total_seconds(),
                ))
                results_array[w, t].append(delta.total_seconds())
            logging.warning('...done with %d workers!' % workers)

    for i in range(len(workers_values)):
        for j in range(len(workers_values)):
            results_array[i, j] = np.avg(results_array[i, j])

    logging.warning('RESULTS (%d passes)' % 5)
    logging.warning('-------')
    logging.warning(str(results_array))
