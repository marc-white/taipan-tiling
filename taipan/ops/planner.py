"""
Plan a tranche of Taipan observations.
"""

import logging
import datetime
import copy

import taipan.scheduling as ts
import taipan.simulate.fullsurvey as tfs

from taipan.simulate.utils import updatesci as utils_updatesci
from taipan.simulate.utils import tiling as utils_tiling

from src.resources.stable.readout import readAlmanacStats as rAS
from src.resources.stable.readout import readCentroidsAffected as rCA

from src.resources.stable.manipulate import makeTilesQueued as mTQ

from src.resources.stable.output import outputObsDefFile as oODF


def plan_period(cursor,
                start_dt, end_dt, date_start, date_end,
                prisci=False, prisci_end=None,
                check_dark_bounds=False,
                output_dir='.', resolution=15.):
    """
    Create an observing plan for a set period of time.

    Note that this function assumes that the operating database has been
    correctly prepared, and has a selection tiles ready for observation.

    The period of time specified needs to be *all* dark time. An error
    will be thrown if this is not the case and
     ``check_dark_bounds`` is switched on.

    This function is an adaptation of
    :any:`taipan.simulate.fullsurvey.sim_do_night`.

    Parameters
    ----------
    cursor : :obj:`psybopg2.connection.cursor`
        Cursor for communicating with the database
    start_dt : :obj:`datetime.datetime`
        Start datetime of the period to plan for. Should be naive, but in UTC.
    end_dt : :obj:`datetime.datetime`
        End datetime of the period to plan for. Should be naive, but in UTC.
    check_dark_bounds : :obj:`bool`
        Should the bounds be checked to see if the period given is actually
        all dark time? Note this is not necessary if this function has been
        invoked by a properly constructed wrapper function, e.g.
        :any:`plan_night`. Defaults to False. If True, and the period given is
        found not to contain all dark time, :any:`ValueError` will be raised.
    output_dir : :obj:`str`
        Directory to write the tile configuration files to. Defaults to './'
        (i.e. the present working directory).
    date_start, date_end : :obj:`datetime.date`
        The start and end dates of the survey in local time. Required for
        calculation of, e.g. time remaining to observe fields.
    prisci : obj:`bool`
        Denotes whether this period should be considered a 'priority science'
        period or not. Defaults to False.
    prisci_end : :obj:`datetime.timedelta`
        Denotes the length of time after survey start (measured from midday of
        the first day, local time) that the priority science period runs for.
        Defaults to None. An error will be thrown if ``prisci=True`` and
        ``prisci_end`` is None.
    resolution : :obj:`float`, minutes
        Resolution of the almanacs (observability information) stored in the
        database. Defaults to 15 (minutes).


    Returns
    -------
    file_names : :obj:`str`
        A list of the (absolute) file names generated

    """
    # Input checking
    if start_dt is None or end_dt is None:
        raise ValueError('Must provide both start and end datetimes!')
    if start_dt > end_dt:
        temp = end_dt
        end_dt = start_dt
        start_dt = temp
    if prisci and prisci_end is None:
        raise ValueError('Must provide prisci_end when prisci=True')

    logging.info('Planning observing for period %s to %s' % (
        start_dt.strftime('%y-%m-%d %H:%M:%S'),
        end_dt.strftime('%y-%m-%d %H:%M:%S'),
    ))

    if check_dark_bounds:
        start_dark, end_dark = rAS.next_night_period(
            cursor, start_dt - datetime.timedelta(minutes=resolution)
        )
        if start_dark > start_dt or end_dark < end_dt:
            raise ValueError('The specified period is not all dark time')

    local_utc_now = copy.copy(start_dt)
    midday_start = ts.utc_local_dt(datetime.datetime.combine(date_start,
                                                             datetime.time(12,
                                                                           0,
                                                                           0)))
    midday_end = ts.utc_local_dt(datetime.datetime.combine(date_end,
                                                           datetime.time(12,
                                                                         0,
                                                                         0)))

    tiles_queued = []

    while local_utc_now < end_dt - datetime.timedelta(days=ts.POINTING_TIME):

        tile_to_obs, fields_available, tiles_scores, scores_array, \
        field_periods, fields_by_tile, hours_obs = tfs.select_best_tile(
            cursor, local_utc_now,
            end_dt, None,  # midday_end no longer used
            prioritize_lowz=prisci,
            midday_start=midday_start
        )

        if tile_to_obs is None:
            logging.info('No tiles available - advancing')
            local_utc_now = rAS.next_time_available(
                cursor, local_utc_now,
                end_dt=end_dt, minimum_airmass=2.0,
                resolution=resolution,
            )
            continue

        # Set the tile up for observation
        logging.info('Observing tile %d (score: %.1f), field %d at '
                     'time %s, RA %3.1f, DEC %2.1f' %
                     (tile_to_obs, tiles_scores[tile_to_obs],
                      fields_by_tile[tile_to_obs],
                      local_utc_now.strftime('%Y-%m-%d %H:%M:%S'),
                      [x['ra'] for x in scores_array if
                       x['tile_pk'] == tile_to_obs][0],
                      [x['dec'] for x in scores_array if
                       x['tile_pk'] == tile_to_obs][0],
                      ))
        # Queue the tile in the system
        mTQ.execute(cursor, [tile_to_obs, ], time_obs=[local_utc_now, ])
        tiles_queued.append(tile_to_obs)

        # Re-tile the affect areas
        fields_to_retile = rCA.execute(cursor, tile_list=[tile_to_obs])
        # Re-do the priorities just to be sure
        utils_updatesci.update_science_targets(cursor,
                                               field_list=fields_to_retile,
                                               do_tp=True, do_d=False,
                                               prisci=prisci)
        logging.info('Retiling fields %s' % ', '.join(str(i) for i in
                                                      fields_to_retile))
        # Switch the logger to DEBUG
        # logger.setLevel(logging.DEBUG)
        utils_tiling.retile_fields(cursor, fields_to_retile, tiles_per_field=1,
                                   tiling_time=local_utc_now,
                                   disqualify_below_min=False,
                                   multicores=4,
                                   # prisci=prioritize_lowz_today,
                                   )

        local_utc_now += datetime.timedelta(days=ts.POINTING_TIME)

    logging.info('Queued %4d tiles in the requested period' %
                 len(tiles_queued))

    # Now need to write the tile files out to the specified directory
    file_names = oODF.execute(cursor, unobserved=True, unqueued=False,
                              output_dir=output_dir, local_tz=ts.UKST_TIMEZONE)
    file_names.sort()

    return file_names


def plan_night(cursor, night, date_start, date_end,
               prisci=False, prisci_end=None,
               output_dir='.', resolution=15.):
    """
    Create an observing plan for a night of Taipan observations.

    This function is predominantly a wrapper for :any:`plan_period`, but does
    additional useful things like computing the bounds of the night's
    dark time, etc.

    Note that this function assumes that the operating database has been
    correctly prepared, and has a selection tiles ready for observation.

    Parameters
    ----------
    cursor : :obj:`psybopg2.connection.cursor`
        Cursor for communicating with the database
    night : :obj:`datetime.date`
        The night to plan for. The date corresponds to the local date at the
        *start* of the night (i.e. for the night of 23/24 April, ``night``
        should correspond to April 23.)
    output_dir : :obj:`str`
        Directory to write the tile configuration files to. Defaults to './'
        (i.e. the present working directory).
    date_start, date_end : :obj:`datetime.date`
        The start and end dates of the survey in local time. Required for
        calculation of, e.g. time remaining to observe fields.
    prisci : obj:`bool`
        Denotes whether this period should be considered a 'priority science'
        period or not. Defaults to False.
    prisci_end : :obj:`datetime.timedelta`
        Denotes the length of time after survey start (measured from midday of
        the first day, local time) that the priority science period runs for.
        Defaults to None. An error will be thrown if ``prisci=True`` and
        ``prisci_end`` is None.
    output_dir : :obj:`str`
        Directory to write the tile configuration files to. Defaults to './'
        (i.e. the present working directory).
    resolution : :obj:`float`, minutes
        Resolution of the almanacs (observability information) stored in the
        database. Defaults to 15 (minutes).

    Returns
    -------

    """
    # Input checking
    if date_end < date_start:
        temp = copy.copy(date_end)
        date_end = date_start
        date_start = temp
    if night < date_start or night > date_end:
        raise ValueError('The night you have requested to schedule is '
                         'outside the bounds of the survey dates!')
    if prisci and prisci_end is None:
        raise ValueError('Must provide prisci_end when prisci=True')

    logging.info('Preparing for observing night %s' %
                 night.strftime('%Y-%m-%d'))

    # Function variables
    file_names = []

    # Start at midday of the night in question
    local_utc_now = datetime.datetime.combine(night ,datetime.time(12, 0, 0))
    local_utc_now = ts.utc_local_dt(local_utc_now)
    local_utc_stop = datetime.datetime.combine(
        night + datetime.timedelta(days=1),
        datetime.time(12, 0, 0))
    local_utc_stop = ts.utc_local_dt(local_utc_stop)

    while local_utc_now < local_utc_stop:
        dark_start, dark_end = rAS.next_night_period(cursor, local_utc_now)
        if dark_start > local_utc_stop: break

        file_names += plan_period(cursor, dark_start, dark_end,
                                  date_start, date_end,
                                  prisci=prisci, prisci_end=prisci_end,
                                  check_dark_bounds=False,  # Already done
                                  output_dir=output_dir, resolution=resolution)

        local_utc_now = dark_end + datetime.timedelta(minutes=resolution/2.)

    logging.info('Preparation for observing night %s complete!' %
                 night.strftime('%Y-%m-%d'))

    return file_names
