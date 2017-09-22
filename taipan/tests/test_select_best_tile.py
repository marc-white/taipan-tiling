# Test the ability to do parallel re-tiling


import taipan.simulate.fullsurvey as tfs

from src.scripts import get_connection
import src.resources.v0_0_1.readout.readAlmanacStats as rAS

import datetime
import logging
import sys
import traceback

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
        filename='./testlog_select_best_tile.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('TESTING OPTIONS FOR select_best_tile')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    new_schema = [True, False]
    passes = 5
    results = {True: [],
               False: [],
               }

    init_dt = datetime.datetime(2020, 5, 1, 14, 0)

    dark_start, dark_end = rAS.next_night_period(cursor, init_dt, field_id=1)
    test_dt = dark_start + datetime.timedelta(minutes=15.)

    for b in new_schema:
        tfs.USE_NEW_FIELDS_AVAIL = b

        for i in range(passes):
            start = datetime.datetime.now()
            _ = tfs.select_best_tile(cursor, test_dt, dark_end,
                                     midday_end=datetime.datetime(2022, 1, 1, 0,
                                                                  0))
            end = datetime.datetime.now()
            delta = end - start
            results[b].append(delta.total_seconds())


    logging.warning('RESULTS (%d passes)' % 5)
    logging.warning('Boolean denotes if new scheme being used')
    logging.warning('-------')
    logging.warning(str(results))
    for k, r in results.items():
        logging.warning('%5s: %5.1f +/- %4.1f' % (str(k),
                                                  np.average(r), np.std(r), ))
    logging.warning('-------')
