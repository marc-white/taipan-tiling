# Test the ability to do parallel re-tiling

from src.resources.v0_0_1.readout import readScience, readGuides, \
    readStandards, readCentroids
from src.scripts.connection import get_connection

from taipan.simulate.fullsurvey import select_best_tile

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
        filename='./testlog_parallel_scoring.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('TESTING PARALLEL SCORING')
    logging.warning('NOTE: Your system has %d cores' %
                    multiprocessing.cpu_count())

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    workers_values = [
        # 1,
        # 4,
        # int(0.8 * multiprocessing.cpu_count()),
        multiprocessing.cpu_count(),
        # 2 * multiprocessing.cpu_count(),
        # 50,
        # 100,
    ]

    results_dict = {w: [] for w in workers_values}

    dt = datetime.datetime(2017, 9, 2, 18, 29, 59)
    per_end = datetime.datetime(2017, 9, 3, 0, 0, 0)
    midday_end = datetime.datetime(2019, 1, 1, 0, 0)

    # Get the targets and fields
    for i in range(3):
        logging.warning('Pass %d...' % i)
        # Do the tests
        for w in workers_values:
            logging.warning('   %3d workers' % w)
            start = datetime.datetime.now()

            select_best_tile(cursor, dt, per_end, midday_end,
                             multipool_workers=w)

            end = datetime.datetime.now()
            delta = end - start
            results_dict[w].append(delta.total_seconds())

        logging.warning('... pass complete!')

    logging.warning('RESULTS (%d passes)' % 5)
    logging.warning('-------')
    for w in workers_values:
        logging.warning('%3d workers: %4.1f +/- %3.1f' % (
            w, np.average(results_dict[w]), np.std(results_dict[w]),
        ))
    logging.warning('-------')
