# Test the speed of running next_observable_period in parallel

from src.resources.stable.readout import readScience, readGuides, \
    readStandards, readCentroids, readAlmanacStats
from src.scripts.extract import extract_from
from src.scripts.connection import get_connection

from taipan.simulate.fullsurvey import select_best_tile

import datetime
import random
import logging
import sys
import traceback
import multiprocessing

from functools import partial

import numpy as np


def _field_period_reshuffle(f,
                            dt=None, per_end=None,
                            ):
    """
    Helper function for parallelizing the retrieval of next available field
    period

    Parameters
    ----------
    f : :obj:`int`
        Field ID
    dt : :obj:`datetime.datetime`
        Datetime that we are searching forward from (inclusively)
    per_end : :obj:`datetime.datetime`
        End of the time period we are searching through

    Returns
    -------
    period : (:obj:`datetime.datetime`, :obj:`datetime.datetime`) or
    (:obj:`None`, :obj:`None`)
        Time period that this field is observable for
    """
    with get_connection().cursor() as cursor_int:
        period = readAlmanacStats.next_observable_period(
            cursor_int, f, dt,
            datetime_to=per_end,
        )
    return period

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
        filename='./testlog_parallel_nextobsper.log',
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.warning('TESTING PARALLEL next_observable_period')
    logging.warning('NOTE: Your system has %d cores' %
                    multiprocessing.cpu_count())

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()

    workers_values = [
        1,
        4,
        int(0.8 * multiprocessing.cpu_count()),
        multiprocessing.cpu_count(),
        # 2 * multiprocessing.cpu_count(),
        # 50,
        # 100,
    ]

    results_dict = {w: [] for w in workers_values}

    dt = datetime.datetime(2017, 9, 2, 18, 29, 59)
    per_end = datetime.datetime(2017, 9, 3, 0, 0, 0)
    midday_end = datetime.datetime(2019, 1, 1, 0, 0)

    all_fields = extract_from(cursor, 'field', conditions=[('is_active',
                                                            '=',
                                                            True), ],
                              columns=['field_id'], distinct=True)['field_id']

    passes = 10
    no_of_fields = 3600

    # Get the targets and fields
    for i in range(passes):
        logging.warning('Pass %d...' % i)
        fields_available = random.sample(all_fields, no_of_fields)
        # Do the tests
        random.shuffle(workers_values)
        for w in workers_values:
            logging.warning('   %3d workers' % w)
            start = datetime.datetime.now()

            if w == 1:
                field_periods = {r: readAlmanacStats.next_observable_period(
                    cursor, r, dt,
                    datetime_to=per_end,
                ) for
                    r in
                    # scores_array
                    fields_available
                }
            else:
                field_periods_partial = partial(_field_period_reshuffle, dt=dt,
                                                per_end=per_end)
                pool = multiprocessing.Pool(w)
                field_periods = pool.map(field_periods_partial,
                                         fields_available)
                pool.close()
                pool.join()
                field_periods = {
                    fields_available[i]: field_periods[i] for i
                    in range(len(field_periods))}

            end = datetime.datetime.now()
            delta = end - start
            results_dict[w].append(delta.total_seconds())

        logging.warning('... pass complete!')

    logging.warning('RESULTS (%d passes)' % passes)
    logging.warning('-------')
    for w in workers_values:
        logging.warning('%3d workers: %4.1f +/- %3.1f' % (
            w, np.average(results_dict[w]), np.std(results_dict[w]),
        ))
    logging.warning('-------')
