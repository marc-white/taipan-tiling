# Restart a full tiling of the Taipan galaxy survey
# Assumes you want to run a fibre changover on the restart

import taipan.simulate.fullsurvey as tfs
import taipan.scheduling as ts
import taipan.core as tp

import logging
import sys
import datetime
import random
import numpy as np
import math
import traceback

from utils.tiling import retile_fields
from utils.updatesci import update_science_targets

from src.scripts.connection import get_connection

from src.resources.v0_0_1.readout.readCentroids import execute as rCexec
from src.resources.v0_0_1.readout import readScienceTypes as rST

from src.resources.v0_0_1.manipulate import makeScienceDiff as mScD
from src.resources.v0_0_1.manipulate import makeSciencePriorities as mScP
from src.resources.v0_0_1.manipulate import makeScienceTypes as mScTy
from src.resources.v0_0_1.manipulate import makeScienceDiff as mSD
from src.resources.v0_0_1.manipulate import makeCentroidSwitch as mCS

import taipan.simulate.logic as tsl


SIMULATE_LOG_PREFIX = 'SIMULATOR: '


def execute(cursor, date_start, date_end, output_loc='.', prep_db=False,
            instant_dq=False, seed=None, kill_time=None,
            prior_lowz_end=None, weather_loss=0.4):
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
    date_start:
        The start date of the simulated observing period. Should be a
        datetime.date instance.
    date_end:
        The final day of the simulated observing period. Should be a
        datetime.date instance.
    output_loc:
        Optional; location for storing any command-line returns from the
        simulation. Defaults to '.' (i.e. present working directory).
    prep_db:
        Boolean value, denoting whether or not to invoke the sim_prepare_db
        function before beginning the simulation. Defaults to True.
    instant_dq:
        Optional Boolean value, denoting whether to immediately apply
        simulated data quality checks at the tile selection phase (effectively,
        assume instantaneous data processing; True) or not, which requires
        a re-tile of all affected fields at the end of the night (False).
        Defaults to False.
    seed:
        Optional hashable object (i.e. any random thing), used to seed the
        random number module and force deterministic output from randomized
        simulator operations. This is useful for making comparisons between
        different simulator runs. Defaults to None, such that no seed is
        applied.
        .. warning:: Supplying a seed will not work if the Python environment
                     variable ``PYTHONHASHSEED`` has been set.
    prior_lowz_end : datetime.timedelta object, optional
        Denotes for how long after the start of the survey that lowz targets
        should be prioritized. Defaults to None, and which point lowz fields
        will always be prioritized (if prioritize_lowz=True).
    weather_loss: float, in the range [0, 1)
        Percentage of nights lost to weather every calendar year. The nights
        to be lost will be computed at the start of each calendar year, to
        ensure exactly 40% of nights are lost per calendar year (or part
        thereof).

    Returns
    -------
    Nil. Tiling outputs are written to the database (to simulate the action of
    the virtual observer). Anything generated and written out to file will end
    up in output_loc (although currently nothing is).
    """

    # Seed the random number generator
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # if prep_db:
    #     start = datetime.datetime.now()
    #     tfs.sim_prepare_db(cursor,
    #                        prepare_time=ts.utc_local_dt(
    #                            datetime.datetime.combine(
    #                                date_start, datetime.time(12, 0))),
    #                        commit=True)
    #     end = datetime.datetime.now()
    #     delta = end - start
    #     logging.info('Completed DB prep in %d:%2.1f' %
    #                  (delta.total_seconds() / 60, delta.total_seconds() % 60.))



    fields = rCexec(cursor)
    # Construct the almanacs required
    # start = datetime.datetime.now()
    # logging.info('Constructing dark almanac...')
    # dark_almanac = ts.DarkAlmanac(date_start, end_date=date_end,
    #                               resolution=15.)
    # dark_almanac.save()
    # logging.info('Constructing field almanacs...')
    # almanacs = {field.field_id: ts.Almanac(field.ra, field.dec, date_start,
    #                                        end_date=date_end, resolution=15.,
    #                                        minimum_airmass=2)
    #             for field in fields}
    # # Work out which of the field almanacs already exist on disk
    # files_on_disk = os.listdir(output_loc)
    # almanacs_existing = {k: v.generate_file_name() in files_on_disk
    #                      for (k, v) in almanacs.iteritems()}
    # logging.info('Saving almanacs to disc...')
    # for k in [k for (k, v) in almanacs_existing.iteritems() if not v]:
    #     almanacs[k].save()
    # end = datetime.datetime.now()
    # delta = end - start
    # logging.info('Completed almanac prep in %d:%2.1f' %
    #              (delta.total_seconds() / 60, delta.total_seconds() % 60.))
    almanacs = None
    dark_almanac = None

    year_in_days = 365

    # Run the actual observing
    logging.info('Commencing observing...')
    curr_date = date_start
    weather_fails = {curr_date + datetime.timedelta(days=i): random.random() for
                     i in range(year_in_days)}
    weather_fail_thresh = np.percentile(weather_fails.values(),
                                        weather_loss*100.)
    while curr_date <= date_end:
        if curr_date == date_start:
            logging.warning('TILE CHANGEOVER')
            # logging.warning('ABORTING at tile changeover')
            # sys.exit()
            # Modify taipan.core.BUGPOS_MM to add in another 150
            # science fibres
            # We do this by adding a new fibre in between each pair of
            # consecutive science fibres
            # tp.FIBRES_NORMAL.sort()
            # last_fibre = np.max(tp.BUGPOS_MM.keys())
            # for i in range(len(tp.FIBRES_NORMAL)):
            #     pos_avg = (
            #         np.average([tp.BUGPOS_MM[tp.FIBRES_NORMAL[i]][0],
            #                     tp.BUGPOS_MM[tp.FIBRES_NORMAL[i + 1]][0]]),
            #         np.average([tp.BUGPOS_MM[tp.FIBRES_NORMAL[i]][1],
            #                     tp.BUGPOS_MM[tp.FIBRES_NORMAL[i + 1]][1]]),
            #     )
            #     tp.BUGPOS_MM[last_fibre + i + 1] = pos_avg
            #     tp.BUGPOS_ARCSEC[last_fibre + i + 1] = (
            #         pos_avg[0] * tp.ARCSEC_PER_MM,
            #         pos_avg[1] * tp.ARCSEC_PER_MM,
            #     )
            #     tp.BUGPOS_OFFSET[last_fibre + i + 1] = (
            #         math.sqrt(pos_avg[0]**2 + pos_avg[1]**2),
            #         math.degrees(math.atan2(pos_avg[0], pos_avg[1])) % 360.,
            #     )
            #     tp.FIBRES_NORMAL.append(last_fibre + i + 1)
            # tp.TARGET_PER_TILE = 270
            # tp.FIBRES_PER_TILE = 309
            # tp.INSTALLED_FIBRES = 309
            tp._alter_fibres(no_fibres=300)
            # Lose 30 days to the upgrade
            curr_date += datetime.timedelta(days=30.)
            # Need to now do:
            # Complete re-compute of target types, priorities and difficulties
            update_science_targets(cursor, target_list=None,
                                   do_d=True,
                                   prisci=False)
            # Switch the field statuses
            # mCS.execute(cursor, remove_inactive_tiles=True)
            # mScD.execute(cursor)t
            # Complete re-tile of all fields
            fields_to_retile = [t.field_id for t in rCexec(cursor)]
            retile_fields(cursor, fields_to_retile, tiles_per_field=1,
                          tiling_time=ts.utc_local_dt(datetime.datetime.combine(
                              curr_date,
                              datetime.time(12,0,0)
                          )),
                          disqualify_below_min=False,
                          multicores=7
                          )
        if weather_fails[curr_date] > weather_fail_thresh:
            tfs.sim_do_night(cursor, curr_date, date_start, date_end,
                             almanac_dict=almanacs, dark_almanac=dark_almanac,
                             instant_dq=instant_dq, check_almanacs=False,
                             commit=True, kill_time=kill_time,
                             prisci_end=prior_lowz_end)
        else:
            logging.info('WEATHER LOSS: Lost %s to weather' %
                         curr_date.strftime('%Y-%m-%d'))
        curr_date += datetime.timedelta(1.)
        # if curr_date.day == 1:
        #     mSD.execute(cursor)
        if curr_date not in weather_fails.keys():
            weather_fails = {
                curr_date + datetime.timedelta(days=i): random.random() for
                i in range(year_in_days)
                }
            weather_fail_thresh = np.percentile(weather_fails.values(),
                                                weather_loss * 100.)


        # if curr_date == datetime.date(2017, 4, 5):
        #     break

    logging.info('----------')
    logging.info('OBSERVING COMPLETE')
    logging.info('----------')


if __name__ == '__main__':

    sim_start = datetime.date(2019, 5, 1)
    sim_end = datetime.date(2022, 7, 1)
    global_start = datetime.datetime.now()
    prior_lowz_end = datetime.date(2022, 7, 1) - sim_start

    kill_time = None
    # kill_time = datetime.datetime(2019, 1, 1, 0, 0, 0)

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
        filename='./simlog_%s_to_%s_at_%s' % (
            sim_start.strftime('%Y%m%d'),
            sim_end.strftime('%Y%m%d'),
            global_start.strftime('%Y%m%d-%H%M'),
        ),
        filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.info('Executing fullsurvey_baseline_resume.py as file')
    logging.info('*** THIS IS AN INSTANT-FEEDBACK SIMULATION')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    logging.debug('Doing execute function')
    execute(cursor, sim_start, sim_end,
            instant_dq=True,
            output_loc='.', prep_db=False, kill_time=kill_time,
            seed=100, prior_lowz_end=prior_lowz_end)

    global_end = datetime.datetime.now()
    global_delta = global_end - global_start
    logging.info('')
    logging.info('--------')
    logging.info('SIMULATION COMPLETE')
    logging.info('Simulated %d nights' % (sim_end - sim_start).days)
    logging.info('Simulation time:')
    logging.info('%dh %dm %2.1fs' % (
        global_delta.total_seconds() // 3600,
        (global_delta.total_seconds() % 3600) // 60,
        global_delta.total_seconds() % 60,
    ))
