# Generate reports on completed simulations

from src.resources.v0_0_1.readout.readObservingLog import execute as rOLexec
from src.resources.v0_0_1.readout.readScienceObservingInfo import execute as \
    rSOIexec
from src.resources.v0_0_1.readout.readTileObservingInfo import execute as \
    rTOIexec
from src.resources.v0_0_1.readout.readTileScores import execute as rTSexec
from src.resources.v0_0_1.readout.readCentroidsByTarget import execute as \
    rCBTexec
import src.resources.v0_0_1.readout.readAlmanacStats as rAS
from src.resources.v0_0_1.readout.readScience import execute as rScexec

import taipan.scheduling as ts

import numpy as np
import datetime


def generate_report(cursor):
    """
    Generate a report on the simulation from the database results.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for interacting with the database.

    Returns
    -------
    Nil. Output is displayed to terminal (and can be piped to a file if this
    module is run as a script).
    """
    pass


def generate_tile_choice(cursor, dt, prioritize_lowz=True, midday_end=None,
                         output=True):
    """
    Generate a report on why a particular tile was chosen at a particular time.

    Parameters
    ----------
    cursor : psycopg2 cursor object
        For communicating with the simulation database
    dt : datetime.datetime object
        The time at which the tile was chosen. The function will find and
        report on the selection of the first tile observed at or after
        this time. Should be in UT (although the datetime object itself should
        be naive).

    Returns
    -------
    Nil. Output is printed to the terminal.
    """

    obs_log = rOLexec(cursor)
    tile_obs = rTOIexec(cursor)
    _, _, targets_completed = rSOIexec(cursor)
    tile_obs.sort(order='date_obs')
    tile_scores = rTSexec(cursor, unobserved_only=False,
                          metrics=['prior_sum', 'n_sci_rem'])
    # print('Original tile scores:')
    # print(tile_scores)

    if midday_end is None:
        midday_end = np.max(tile_obs['date_obs'])

    # Work out which tile was observed at or after the given date
    tile_to_check = tile_obs[tile_obs['date_obs'] >= dt][0]

    # print('Date of observation:')
    # print(tile_to_check['date_obs'])

    # Get the scores of the tiles that were in contention at this point
    tile_scores = tile_scores[np.in1d(tile_scores['tile_pk'],
                                      tile_obs[np.logical_and(
                                          tile_obs['date_config'] <=
                                          max(
                                              tile_to_check['date_obs'],
                                              datetime.datetime(2017,4,1,12,1)),
                                          tile_obs['date_obs'] >=
                                          tile_to_check['date_obs']
                                      )]['tile_pk']
                                      )]

    # print('Trimmed tile scores:')
    # print(tile_scores)

    # Work out which fields were observable at this time
    dark_start, dark_end = rAS.next_night_period(cursor,
                                                 tile_to_check['date_obs'],
                                                 limiting_dt=tile_to_check[
                                                                 'date_obs'] +
                                                             datetime.timedelta(
                                                     1.),
                                                 dark=True, grey=False)
    field_periods = {r: rAS.next_observable_period(
        cursor, r, tile_to_check['date_obs'],
        datetime_to=dark_end) for
                     r in list(set(tile_scores['field_id']))}
    fields_available = [f for f, v in field_periods.iteritems() if
                        v[0] is not None and
                        v[0] - datetime.timedelta(seconds=ts.SLEW_TIME)
                        < tile_to_check['date_obs'] and
                        (v[1] is None or
                         v[1] > tile_to_check['date_obs'] +
                         datetime.timedelta(seconds=ts.POINTING_TIME))]

    # Restrict tile_scores to those fields actually available
    tile_scores = tile_scores[np.in1d(tile_scores['field_id'],
                                      np.asarray(fields_available))]

    # Compute the hours remaining for all fields
    if prioritize_lowz:
        lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                               unobserved=True)
        hours_obs_lowz = {f: rAS.hours_observable(cursor, f,
                                                  tile_to_check['date_obs'],
                                                  datetime_to=max(
                                                      midday_end,
                                                      tile_to_check['date_obs'] +
                                                      datetime.
                                                      timedelta(30.)
                                                  ),
                                                  hours_better=True) for
                          f in tile_scores['field_id'] if
                          f in lowz_fields}
        hours_obs_oth = {f: rAS.hours_observable(cursor, f,
                                                 tile_to_check['date_obs'],
                                                 datetime_to=
                                                 tile_to_check['date_obs'] +
                                                 datetime.timedelta(
                                                     365),
                                                 hours_better=True) for
                         f in tile_scores['field_id'] if
                         f not in lowz_fields}
        hours_obs = dict(hours_obs_lowz, **hours_obs_oth)
    else:
        hours_obs = {f: rAS.hours_observable(cursor, f, tile_to_check['date_obs'],
                                             datetime_to=tile_to_check['date_obs'] +
                                                 datetime.timedelta(
                                                     365),
                                             hours_better=True) for
                     f in tile_scores['field_id']}

    # This is the hardest part - we need to compute the n_sci_rem for each
    # tile at this point in time
    # n_sci_rem = {}
    # for field in list(set(tile_scores['field_id'])):
    #     tgt_this_field = np.asarray([t.idn for t in rScexec(cursor, field_list=[field,])])
    #     tgt_rem_this_field = np.logical_and(
    #         np.in1d(tgt_this_field,
    #                 obs_log[obs_log['date_obs'] < tile_to_check['date_obs']]['target_id']),
    #         ~np.in1d(tgt_this_field,
    #                 obs_log[obs_log['date_obs'] > tile_to_check['date_obs']]['target_id']),
    #     )
    #     tgt_rem_this_field = np.logical_and(
    #         tgt_rem_this_field, ~np.in1d(tgt_this_field, targets_completed['target_id'])
    #     )
    #     n_sci_rem[field] = np.count_nonzero(~tgt_rem_this_field)

    # print('Tile scores field id:')
    # print(sorted(tile_scores['field_id']))
    # if prioritize_lowz:
    #     print('lowz fields:')
    #     print(sorted(lowz_fields))


    # Report on the following:
    # - The highest scoring field
    # - The highest scoring lowz field
    # - The field with the highest n_sci_rem
    # - The field with lowest hours_obs
    # - The field that was actually selected
    tile_scores.sort(order='prior_sum')
    highest_score = tile_scores[::-1][0]
    if prioritize_lowz:
        highest_score_lowz = \
            tile_scores[np.in1d(tile_scores['field_id'], lowz_fields)][::-1][0]
    highest_n_sci = tile_scores[np.argmax(tile_scores['n_sci_rem'])]
    lowest_hrs = tile_scores[tile_scores['field_id'] == min(hours_obs, key=hours_obs.get)][0]

    stat_type = [
        'Highest raw score',
        'Highest n. sci. rem.',
        'Lowest rem. hours obs.',
        'Selected field',
    ]
    if prioritize_lowz:
        stat_type.append('Highest raw score (low-z)')

    stats = [
        highest_score['tile_pk'],
        highest_n_sci['tile_pk'],
        lowest_hrs['tile_pk'],
        tile_to_check['tile_pk']
    ]
    if prioritize_lowz:
        stats.append(highest_score_lowz['tile_pk'])

    if output:
        print(' TILE SELECTION REPORT ')
        print(' --------------------- ')
        print(' %s | %s | %s | %s | %s | %s | %s ' % (
            'Tile type'.ljust(25),
            'PK'.ljust(5),
            'Field',
            'Score',
            'NSciR',
            'hours',
            'Final score',
        ))
        for i in range(len(stats)):
            print(' %s | %5d | %5d | %4.1f | %5d | %4.1f | %4.1f ' % (
                stat_type[i].ljust(25),
                stats[i],
                tile_scores[tile_scores['tile_pk'] == stats[i]][0]['field_id'],
                tile_scores[tile_scores['tile_pk'] == stats[i]][0]['prior_sum'],
                tile_scores[tile_scores['tile_pk'] == stats[i]][0]['n_sci_rem'],
                hours_obs[
                    tile_scores[tile_scores['tile_pk'] == stats[i]][0]['field_id']],
                (tile_scores[tile_scores['tile_pk'] == stats[i]][0]['prior_sum'] * tile_scores[tile_scores['tile_pk'] == stats[i]][0]['n_sci_rem']) / hours_obs[
                    tile_scores[tile_scores['tile_pk'] == stats[i]][0]['field_id']],
            ))

    scores = []
    for i in range(len(stats)):
        scores.append((tile_scores[tile_scores['tile_pk'] == stats[i]][0]['prior_sum'] * tile_scores[tile_scores['tile_pk'] == stats[i]][0]['n_sci_rem']) / hours_obs[
                tile_scores[tile_scores['tile_pk'] == stats[i]][0]['field_id']])

    # Return True if the best possible tile was picked; return False otherwise
    if scores[3] == np.max(scores):
        return True
    else:
        return False


def check_tile_choice(cursor, midday_end=None):
    """
    Check all tile choices for correctness
    Parameters
    ----------
    cursor
    midday_end

    Returns
    -------

    """
    # Read in tile observation info
    obs_log = rOLexec(cursor)
    tile_obs = rTOIexec(cursor)
    _, _, targets_completed = rSOIexec(cursor)
    tile_obs.sort(order='date_obs')
    tile_scores = rTSexec(cursor, unobserved_only=False,
                          metrics=['prior_sum', 'n_sci_rem'])
    # print('Original tile scores:')
    # print(tile_scores)

    if midday_end is None:
        midday_end = np.max(tile_obs['date_obs'])

    no_tiles = len(tile_obs)
    no_fails = 0

    for dt in tile_obs['date_obs']:
        print(dt.strftime('%Y-%m-%d %H:%M:%S'))
        # Work out which tile was observed at or after the given date
        tile_to_check = tile_obs[tile_obs['date_obs'] >= dt][0]

        # print('Date of observation:')
        # print(tile_to_check['date_obs'])

        # Get the scores of the tiles that were in contention at this point
        tile_scores = tile_scores[np.in1d(tile_scores['tile_pk'],
                                          tile_obs[np.logical_and(
                                              tile_obs['date_config'] <=
                                              max(
                                                  tile_to_check['date_obs'],
                                                  datetime.datetime(2017, 4, 1, 12,
                                                                    1)),
                                              tile_obs['date_obs'] >=
                                              tile_to_check['date_obs']
                                          )]['tile_pk']
                                          )]

        # print('Trimmed tile scores:')
        # print(tile_scores)

        # Work out which fields were observable at this time
        dark_start, dark_end = rAS.next_night_period(cursor,
                                                     tile_to_check['date_obs'],
                                                     limiting_dt=tile_to_check[
                                                                     'date_obs'] +
                                                                 datetime.timedelta(
                                                                     1.),
                                                     dark=True, grey=False)
        field_periods = {r: rAS.next_observable_period(
            cursor, r, tile_to_check['date_obs'],
            datetime_to=dark_end) for
                         r in list(set(tile_scores['field_id']))}
        fields_available = [f for f, v in field_periods.iteritems() if
                            v[0] is not None and
                            v[0] - datetime.timedelta(seconds=ts.SLEW_TIME)
                            < tile_to_check['date_obs'] and
                            (v[1] is None or
                             v[1] > tile_to_check['date_obs'] +
                             datetime.timedelta(seconds=ts.POINTING_TIME))]

        # Restrict tile_scores to those fields actually available
        tile_scores = tile_scores[np.in1d(tile_scores['field_id'],
                                          np.asarray(fields_available))]

        # Compute the hours remaining for all fields
        # if prioritize_lowz:
        lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                               unobserved=True)
        hours_obs_lowz = {f: rAS.hours_observable(cursor, f,
                                                  tile_to_check['date_obs'],
                                                  datetime_to=max(
                                                      midday_end,
                                                      tile_to_check[
                                                          'date_obs'] +
                                                      datetime.
                                                      timedelta(30.)
                                                  ),
                                                  hours_better=True) for
                          f in tile_scores['field_id'] if
                          f in lowz_fields}
        hours_obs_oth = {f: rAS.hours_observable(cursor, f,
                                                 tile_to_check['date_obs'],
                                                 datetime_to=
                                                 tile_to_check['date_obs'] +
                                                 datetime.timedelta(
                                                     365),
                                                 hours_better=True) for
                         f in tile_scores['field_id'] if
                         f not in lowz_fields}
        hours_obs = dict(hours_obs_lowz, **hours_obs_oth)
        # else:
        #     hours_obs = {
        #     f: rAS.hours_observable(cursor, f, tile_to_check['date_obs'],
        #                             datetime_to=tile_to_check['date_obs'] +
        #                                         datetime.timedelta(
        #                                             365),
        #                             hours_better=True) for
        #     f in tile_scores['field_id']}

        # This is the hardest part - we need to compute the n_sci_rem for each
        # tile at this point in time
        # n_sci_rem = {}
        # for field in list(set(tile_scores['field_id'])):
        #     tgt_this_field = np.asarray([t.idn for t in rScexec(cursor, field_list=[field,])])
        #     tgt_rem_this_field = np.logical_and(
        #         np.in1d(tgt_this_field,
        #                 obs_log[obs_log['date_obs'] < tile_to_check['date_obs']]['target_id']),
        #         ~np.in1d(tgt_this_field,
        #                 obs_log[obs_log['date_obs'] > tile_to_check['date_obs']]['target_id']),
        #     )
        #     tgt_rem_this_field = np.logical_and(
        #         tgt_rem_this_field, ~np.in1d(tgt_this_field, targets_completed['target_id'])
        #     )
        #     n_sci_rem[field] = np.count_nonzero(~tgt_rem_this_field)

        # print('Tile scores field id:')
        # print(sorted(tile_scores['field_id']))
        # if prioritize_lowz:
        #     print('lowz fields:')
        #     print(sorted(lowz_fields))


        # Report on the following:
        # - The highest scoring field
        # - The highest scoring lowz field
        # - The field with the highest n_sci_rem
        # - The field with lowest hours_obs
        # - The field that was actually selected
        tile_scores.sort(order='prior_sum')
        highest_score = tile_scores[::-1][0]
        try:
            highest_score_lowz = \
                tile_scores[np.in1d(tile_scores['field_id'], lowz_fields)][::-1][0]
            prioritize_lowz = True
        except KeyError:
            prioritize_lowz = False
        highest_n_sci = tile_scores[np.argmax(tile_scores['n_sci_rem'])]
        lowest_hrs = \
        tile_scores[tile_scores['field_id'] == min(hours_obs, key=hours_obs.get)][0]

        stat_type = [
            'Highest raw score',
            'Highest n. sci. rem.',
            'Lowest rem. hours obs.',
            'Selected field',
        ]
        if prioritize_lowz:
            stat_type.append('Highest raw score (low-z)')

        stats = [
            highest_score['tile_pk'],
            highest_n_sci['tile_pk'],
            lowest_hrs['tile_pk'],
            tile_to_check['tile_pk']
        ]
        if prioritize_lowz:
            stats.append(highest_score_lowz['tile_pk'])

        scores = []
        for i in range(len(stats)):
            scores.append((tile_scores[tile_scores['tile_pk'] == stats[i]][0]['prior_sum'] *
                           tile_scores[tile_scores['tile_pk'] == stats[i]][0]['n_sci_rem']) / \
                          hours_obs[
                              tile_scores[tile_scores['tile_pk'] == stats[i]][0]['field_id']])
        if scores[3] == np.max(scores):
            pass
        else:
            no_fails += 1
            print('Possible failure at %s' % dt.strftime('%Y-%m-%d %H:%M:%S'))
