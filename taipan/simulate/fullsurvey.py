# Simulate a full tiling of the Taipan galaxy survey

import sys
import logging
import taipan.core as tp
import taipan.tiling as tl
import taipan.scheduling as ts
import simulate as tsim

from utils.tiling import retile_fields
from utils.bugfail import simulate_bugfails

import pickle

import numpy as np
import atpy
import ephem
import operator
import os
import datetime
import random

from src.resources.v0_0_1.readout.readCentroids import execute as rCexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readScience import execute as rScexec
from src.resources.v0_0_1.readout.readTileScores import execute as rTSexec
from src.resources.v0_0_1.readout.readCentroidsAffected import execute as rCAexec
from src.resources.v0_0_1.readout.readScienceTypes import execute as rSTyexec
from src.resources.v0_0_1.readout.readScienceTile import execute as rSTiexec
from src.resources.v0_0_1.readout.readScienceVisits import execute as rSVexec
from src.resources.v0_0_1.readout.readCentroidsByTarget import execute as \
    rCBTexec
import src.resources.v0_0_1.readout.readAlmanacStats as rAS

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec

from src.resources.v0_0_1.manipulate.makeScienceVisitInc import execute as mSVIexec
from src.resources.v0_0_1.manipulate.makeScienceRepeatInc import execute as mSRIexec
from src.resources.v0_0_1.manipulate.makeTilesObserved import execute as mTOexec
from src.resources.v0_0_1.manipulate.makeTilesQueued import execute as mTQexec
from src.resources.v0_0_1.manipulate.makeTargetPosn import execute as mTPexec
from src.resources.v0_0_1.manipulate.makeTilesReset import execute as mTRexec

import src.resources.v0_0_1.manipulate.makeNSciTargets as mNScT

from src.scripts.connection import get_connection

from simulate import test_redshift_success

SIMULATE_LOG_PREFIX = 'SIMULATOR: '


def sim_prepare_db(cursor, prepare_time=datetime.datetime.now(),
                   commit=True):
    """
    This initial step prepares the database for the simulation run by getting
    the fields in from the database, performing the initial tiling of fields,
    and then returning that information to the database for later use.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for interacting with the database.
    prepare_time:
        Optional; datetime.datetime instance that the database should be
        considered to be prepared at (e.g., initial tiles will have
        date_config set to this). Defaults to datetime.datetime.now().
    commit:
        Optional; Boolean value denoting whether to hard-commit changes to
        actual database. Defaults to True.

    Returns
    -------
    Nil. Database updated in place.
    """

    logger = logging.getLogger()

    try:
        with open('tiles.pobj', 'r') as tfile:
            candidate_tiles = pickle.load(tfile)
    except IOError:
        # Get the field centres in from the database
        logging.info(SIMULATE_LOG_PREFIX + 'Loading targets')
        field_tiles = rCexec(cursor)
        candidate_targets = rScexec(cursor)
        guide_targets = rGexec(cursor)
        standard_targets = rSexec(cursor)

        logging.info(SIMULATE_LOG_PREFIX+'Generating first pass of tiles')
        # TEST ONLY: Trim the tile list to 10 to test DB write-out
        # field_tiles = random.sample(field_tiles, 40)
        # Set the logging to debug for this
        old_level = logger.level
        # logger.setLevel(logging.DEBUG)
        candidate_tiles, targets_remain = \
            tl.generate_tiling_greedy_npasses(candidate_targets,
                                              standard_targets,
                                              guide_targets,
                                              1,
                                              tiles=field_tiles,
                                              repeat_targets=True,
                                              tile_unpick_method='sequential',
                                              sequential_ordering=(2, 1)
                                              )
        # logger.setLevel(old_level)
        logging.info('First tile pass complete!')

        # 'Pickle' the tiles so they don't need
        # to be regenerated later for tests
        with open('tiles.pobj', 'w') as tfile:
            pickle.dump(candidate_tiles, tfile)

    # Write the tiles to DB
    iTexec(cursor, candidate_tiles, config_time=prepare_time,
           disqualify_below_min=False)
    # Commit now in case mNScT not debugged right
    # cursor.connection.commit()

    # Compute the n_sci_rem and n_sci_obs for these tiles
    # mTPexec(cursor)
    mNScT.execute(cursor)

    if commit:
        cursor.connection.commit()

    return


def sim_dq_analysis(cursor, tiles_observed, tiles_observed_at,
                    prob_bugfail=1./10000.,
                    prob_vpec_first=0.3, prob_vpec_second=0.7,
                    prob_lowz_each=0.8):
    # -------
    # FAKE DQ/SCIENCE ANALYSIS
    # -------
    # Read in from DB
    # Get the list of science target IDs on these tiles
    target_ids = np.asarray(rSTiexec(cursor, tiles_observed))

    # Add in small probability of uncategorized 'bug failure'
    target_ids = target_ids[simulate_bugfails([True] * len(target_ids),
                                              prob=prob_bugfail)]

    if len(target_ids) > 0:
        # Get the array of target_ids with target types from the database
        target_types_db = rSTyexec(cursor, target_ids=target_ids)
        # Get an array with the number of visits and repeats of these
        visits_repeats = rSVexec(cursor, target_ids=target_ids)

        # Form an array showing the type of those targets
        # target_types = np.asarray(list(['' for _ in target_types_db]))
        # for ttype in ['is_H0_target', 'is_vpec_target', 'is_lowz_target']:
        #     target_types[
        #         np.asarray([_[ttype] is True for _ in target_types_db],
        #                    dtype=bool)
        #     ] = ttype
        # Calculate a success/failure rate for each target
        # Compute target success based on target type(s)
        # Note function needs the 'updated' value of target visits, which
        # won't be pushed to the database until after the result of this
        # function is implemented
        success_targets = test_redshift_success(target_types_db,
                                                visits_repeats['visits'] + 1,
                                                prob_vpec_first=prob_vpec_first,
                                                prob_vpec_second=
                                                prob_vpec_second,
                                                prob_lowz_each=prob_lowz_each)

        # Set relevant targets as observed successfully, all others
        # observed but unsuccessfully
        mSRIexec(cursor, target_ids[success_targets], set_done=True)
        mSVIexec(cursor, target_ids[~success_targets])

    # Mark the tiles as having been observed
    mTOexec(cursor, tiles_observed, time_obs=tiles_observed_at)

    # Reset all tiles to be unqueued (none should be, but this is just
    # a safety measure)
    mTRexec(cursor)

    return


def sim_do_night(cursor, date, date_start, date_end,
                 almanac_dict=None, dark_almanac=None,
                 save_new_almanacs=True, instant_dq=False,
                 prioritize_lowz=True,
                 commit=True):
    """
    Do a simulated 'night' of observations. This involves:
    - Determine the tiles to do tonighttar
    - 'Observe' them
    - Update the DB appropriately

    Parameters
    ----------
    cursor:
        The psycopg2 cursor for interacting with the database
    date:
        Python datetime.date object. This should be the local date that the
        night *starts* on, eg. the night of 23-24 July should be passed as
        23 July.
    date_start, date_end:
        The dates the observing run starts and ends on. These are required
        in order to compute the amount of time a certain field will remain
        observable.
    almanac_dict:
        Dictionary of taipan.scheduling.Almanac objects used for computing
        scheduling. Should be a dictionary with field IDs as keys, and values
        being either a single Almanac object, or a list of Almanac objects,
        covering the date in question. sim_do_night will calculate new/updated
        almanacs from date_start to date_end if almanacs are not passed for a
        particular field and/or date range. Defaults to None, at which point
        almanacs will be constructed for all fields over the specified date
        range.
    dark_almanac:
        As for almanac_list, but holds the dark almanacs, which simply
        specify dark or grey time on a per-datetime basis. Optional,
        defaults to None (so the necessary DarkAlmanac will be created).
    save_new_almanacs:
        Boolean value, denoting whether to save any new almanacs that are
        created by sim_do_night. Defaults to True.
    instant_dq:
        Optional Boolean value, denoting whether to immediately apply
        simulated data quality checks at the tile selection phase (effectively,
        assume instantaneous data processing; True) or not, which requires
        a re-tile of all affected fields at the end of the night (False).
        Defaults to False.
    prioritize_lowz : Boolean, optional
        Whether or not to prioritize fields with lowz targets by computing
        the hours_observable for those fields against a set end date, and
        compute all other fields against a rolling one-year end date.
        Defaults to True.
    commit:
        Boolean value, denoting whether to hard-commit the database changes made
        to the database proper. Defaults to True.

    Returns
    -------
    local_time_now:
        The local time at which observing ceased. Note that this may be well
        after the night being investigated, due to a long weather break
        occurring.

    """
    logging.info('Doing simulated observing for %s' % date.strftime('%y-%m-%d'))
    # Do some input checking
    # Date needs to be in the range of date_start and date_end
    if date < date_start or date > date_end:
        raise ValueError('date must be in the range [date_start, date_end]')

    logging.info('Checking almanacs for night %s' %
                 date.strftime('%Y-%m-%d'))
    start = datetime.datetime.now()
    # Seed an alamnac dictionary if not passed
    if almanac_dict is None:
        almanac_dict = {}

    # Nest all the almanac_dict values inside a list for consistency.
    for k, v in almanac_dict.iteritems():
        if not isinstance(v, list):
            almanac_dict[k] = [v]
            # Check that all elements of input list are instances of Almanac
            if not np.all([isinstance(a, ts.Almanac) for a in almanac_dict[k]]):
                raise ValueError('The values of almanac_dict must contain '
                                 'single Almanacs of lists of Almanacs')

    if dark_almanac is not None:
        if not isinstance(dark_almanac, ts.DarkAlmanac):
            raise ValueError('dark_almanac must be None, or an instance of '
                             'DarkAlmanac')

    # Needs to do the following:
    # Read in the tiles that are awaiting observation, along with their scores
    scores_array = rTSexec(cursor, metrics=['cw_sum', 'prior_sum', 'n_sci_rem'],
                           ignore_zeros=True)
    logging.debug('Scores array info:')
    logging.debug(scores_array)

    # Make sure we have an almanac for every field in the scores_array for the
    # correct date
    # If we don't, we'll need to make one
    # Note that, because of the way Python's scoping is set up, this will
    # permanently add the almanac to the input dictionary
    # logging.debug('Checking all necessary almanacs are present')
    # almanacs_existing = almanac_dict.keys()
    # almanacs_relevant = {row['field_id']: None for row in scores_array}
    # for row in scores_array:
    #     if row['field_id'] not in almanacs_existing:
    #         almanac_dict[row['field_id']] = [ts.Almanac(row['ra'], row['dec'],
    #                                                     date_start, date_end), ]
    #         if save_new_almanacs:
    #             almanac_dict[row['field_id']][0].save()
    #         almanacs_existing.append(row['field_id'])
    #
    #     # Now, make sure that the almanacs actually cover the correct date range
    #     # If not, replace any existing almanacs with one super Almanac for the
    #     # entire range requested
    #     try:
    #         almanacs_relevant[
    #             row['field_id']] = (a for a in almanac_dict[row['field_id']] if
    #                                 a.start_date <= date <= a.end_date).next()
    #     except KeyError:
    #         # This catches when no almanacs satisfy the condition in the
    #         # list constructor above
    #         almanac_dict[row['field_id']] = [
    #             ts.Almanac(row['ra'], row['dec'],
    #                        date_start, date_end), ]
    #         if save_new_almanacs:
    #             almanac_dict[row['field_id']][0].save()
    #         almanacs_relevant[
    #                 row['field_id']] = almanac_dict[row['field_id']][0]
    #
    # # Check that the dark almanac spans the relevant dates; if not,
    # # regenerate it
    # if dark_almanac is None or (dark_almanac.start_date > date or
    #                             dark_almanac.end_date < date):
    #     dark_almanac = ts.DarkAlmanac(date_start, end_date=date_end)
    #     if save_new_almanacs:
    #         dark_almanac.save()
    #
    # end = datetime.datetime.now()
    # delta = end - start
    # logging.info('Completed (nightly) almanac prep in %d:%2.1f' %
    #              (delta.total_seconds() / 60, delta.total_seconds() % 60.))

    logging.info('Finding first block of dark time for this evening')
    start = datetime.datetime.now()
    # Compute the times for the first block of dark time tonight
    midday = ts.utc_local_dt(datetime.datetime.combine(date,
                                                       datetime.time(12, 0, 0)))
    midday_end = ts.utc_local_dt(datetime.datetime.combine(date_end,
                                                           datetime.time(12, 0,
                                                                         0)))
    # dark_start, dark_end = dark_almanac.next_dark_period(midday,
    #                                                      limiting_dt=midday +
    #                                                      datetime.timedelta(1))
    dark_start, dark_end = rAS.next_night_period(cursor, midday,
                                                 limiting_dt=midday_end,
                                                 dark=True, grey=False)
    end = datetime.datetime.now()
    delta = end - start
    logging.info('Found first block of dark time in %d:%2.1f' %
                 (delta.total_seconds() / 60, delta.total_seconds() % 60.))

    tiles_observed = []
    tiles_observed_at = []

    while dark_start is not None:
        logging.info('Commencing observing...')
        start = datetime.datetime.now()
        logging.info('Observing over dark period %s to %s local' %
                     (ts.localize_utc_dt(dark_start).strftime(
                         '%Y-%m-%d %H:%M:%S'),
                      ts.localize_utc_dt(dark_end).strftime(
                          '%Y-%m-%d %H:%M:%S'), ))
        # ephem_time_now = dark_start
        # local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(ephem_time_now,
        #                                                    ts.EPHEM_DT_STRFMT))
        local_utc_now = dark_start
        while local_utc_now < (dark_end - datetime.timedelta(ts.POINTING_TIME)):
            # ------
            # FAKE WEATHER FAILURES
            # ------
            # For now, assume P% of all tiles are lost randomly to weather
            P = 0.25
            weather_prob = np.random.random(1)
            if weather_prob[0] < P:
                local_utc_now += datetime.timedelta(ts.POINTING_TIME)
                logging.info('*BREAK* Lost one pointing to weather, '
                             'advancing to %s' % (
                    local_utc_now.strftime('%Y-%m-%d %H:%M:%S'),
                ))
                continue

            # Get the next observing period for all fields being considered
            # field_periods = {r['field_id']: almanacs_relevant[
            #     r['field_id']
            # ].next_observable_period(
            #     local_time_now - (datetime.timedelta(
            #         almanacs_relevant[r['field_id']].resolution *
            #         60. / ts.SECONDS_PER_DAY)),
            #     datetime_to=ts.localize_utc_dt(ts.ephem_to_dt(dark_end))) for
            #                  r in scores_array}
            field_periods = {r['field_id']: rAS.next_observable_period(
                cursor, r['field_id'], local_utc_now,
                datetime_to=dark_end) for
                             r in scores_array}
            logging.debug('Next observing period for each field:')
            logging.debug(field_periods)
            logging.info('Next available field will rise at %s' %
                         (min([v[0].strftime('%Y-%m-%d %H:%M:%S') for v in
                               field_periods.itervalues() if
                               v[0] is not None]),)
                         )
            fields_available = [f for f, v in field_periods.iteritems() if
                                v[0] is not None and v[0] < dark_end]
            logging.debug('%d fields available at some point tonight' %
                          len(fields_available))

            # Get the latest scores_array
            scores_array = rTSexec(cursor, metrics=['cw_sum', 'prior_sum',
                                                    'n_sci_rem'],
                                   ignore_zeros=True)

            # Rank the available fields
            logging.info('Computing field scores')
            start = datetime.datetime.now()
            tiles_scores = {row['tile_pk']: (row['n_sci_rem'], row['prior_sum'])
                            for
                            row in scores_array if
                            row['field_id'] in fields_available}
            fields_by_tile = {row['tile_pk']: row['field_id'] for
                              row in scores_array if
                              row['field_id'] in fields_available}
            # hours_obs = {f: almanacs_relevant[f].hours_observable(
            #     local_time_now,
            #     datetime_to=midday_end,
            #     dark_almanac=dark_almanac,
            #     hours_better=True
            # ) for f in fields_by_tile.values()}
            if prioritize_lowz:
                lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                                       unobserved=True)
                hours_obs = {f: rAS.hours_observable(cursor, f, local_utc_now,
                                                     datetime_to=max(
                                                         midday_end,
                                                         local_utc_now +
                                                         datetime.timedelta(
                                                             30.)
                                                     ),
                                                     hours_better=True) for
                             f in fields_by_tile.values() if
                             f in lowz_fields}
                hours_obs += {f: rAS.hours_observable(cursor, f, local_utc_now,
                                                      datetime_to=
                                                      local_utc_now +
                                                      datetime.timedelta(365),
                                                      hours_better=True) for
                              f in fields_by_tile.values() if
                              f not in lowz_fields}
            else:
                hours_obs = {f: rAS.hours_observable(cursor, f, local_utc_now,
                                                     datetime_to=midday_end,
                                                     hours_better=True) for
                             f in fields_by_tile.values()}
            # hours_obs = {h['field_id']: h['count'] for h in
            #              rAS.hours_observable_bulk(cursor,
            #                                        fields_by_tile.values(),
            #                                        local_utc_now,
            #                                        datetime_to=midday_end,
            #                                        hours_better=True)}
            # Modulate scores by hours remaining
            tiles_scores = {t: v[0] * v[1] / hours_obs[fields_by_tile[t]] for
                            t, v in tiles_scores.iteritems()}
            logging.debug('Tiles scores: ')
            logging.debug(tiles_scores)
            end = datetime.datetime.now()
            delta = end - start
            logging.info('Computed tile scores/hours remaining in %d:%2.1f' %
                         (delta.total_seconds() / 60,
                          delta.total_seconds() % 60.))

            # 'Observe' while the remaining time in this dark period is
            # longer than one pointing (slew + obs)

            logging.info('At time %s, going to %s' % (
                local_utc_now.strftime('%Y-%m-%d %H:%M:%S'),
                dark_end.strftime('%Y-%m-%d %H:%M:%S'),
            ))


            # Select the best ranked field we can see
            try:
                # logging.debug('Next observing period for each field:')
                # logging.debug(field_periods)
                tile_to_obs = (t for t, v in sorted(tiles_scores.iteritems(),
                                                    key=lambda x: -1. * x[1]
                                                    ) if
                               field_periods[fields_by_tile[t]][0] is not
                               None and
                               field_periods[fields_by_tile[t]][1] is not
                               None and
                               field_periods[fields_by_tile[t]][0] -
                               datetime.timedelta(seconds=ts.SLEW_TIME) <
                               local_utc_now and
                               field_periods[fields_by_tile[t]][1] >
                               local_utc_now +
                               datetime.timedelta(
                                   seconds=ts.POINTING_TIME)).next()
            except StopIteration:
                # This triggers if fields will be available later tonight,
                # but none are up right now. What we do now is advance time_now
                # to the first time when any field becomes available
                local_utc_now = min([v[0] for f, v in
                                     field_periods.iteritems()
                                     if v[0] is not None and
                                     v[1] if not None and
                                     v[0] > local_utc_now])
                if local_utc_now is None:
                    logging.info('There appears to be no valid observing time '
                                 'remaining out to the end_date')
                    return

                logging.info('No fields up - advancing time to %s' %
                             local_utc_now.strftime('%Y-%m-%d %H:%M:%S'))
                # local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(
                #     ephem_time_now, ts.EPHEM_DT_STRFMT))
                continue

            # 'Observe' the field
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
            # This is the section of code that does the 'observing'
            # This is a proxy for generating an 'observing list' at the start
            # of the night; simulating live operations just turns out to be
            # easier/quicker to implement, and conceptually identical at this
            # stage

            # Set the tile to be queued
            # Note we are *not* setting the tile's date_obs value yet
            mTQexec(cursor, [tile_to_obs], time_obs=None)
            # Record the time that this was done
            tiles_observed_at.append(local_utc_now)

            if instant_dq:
                # Do the DQ analysis now
                sim_dq_analysis(cursor, [tile_to_obs], [local_utc_now])

            # Re-tile the affected areas (should be 7 tiles, modulo any areas
            # where we have deliberately added an over/underdense tiling)
            fields_to_retile = rCAexec(cursor, tile_list=[tile_to_obs])
            # Switch the logger to DEBUG
            # logger.setLevel(logging.DEBUG)
            retile_fields(cursor, fields_to_retile, tiles_per_field=1,
                          tiling_time=local_utc_now,
                          disqualify_below_min=False)
            # logger.setLevel(logging.INFO)

            # Increment time_now and move to observe the next field
            local_utc_now += datetime.timedelta(ts.POINTING_TIME)
            # local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(
            #     ephem_time_now, ts.EPHEM_DT_STRFMT))

            # Set the tile score to 0 so it's not re-observed tonight
            tiles_scores[tile_to_obs] = 0.
            tiles_observed.append(tile_to_obs)

        # When this dark period is exhausted, figure out when the next dark
        # period is tonight (if there is one)
        logging.debug('Finding next block of dark time for tonight')
        # dark_start, dark_end = dark_almanac.next_dark_period(
        #     local_utc_now + datetime.timedelta(
        #         minutes=15.),
        #     limiting_dt=midday + datetime.timedelta(1))
        dark_start, dark_end = rAS.next_night_period(cursor,
            local_utc_now + datetime.timedelta(
                minutes=15.),
            limiting_dt=midday + datetime.timedelta(1))

    try:
        _ = local_utc_now
    except (NameError, UnboundLocalError, ):
        # This means that there was no dark period at all tonight.
        # So, we need to set local_time_now to be a reasonable time the
        # following morning (say 9am local)
        local_utc_now = ts.utc_local_dt(
            datetime.datetime.combine(date + datetime.timedelta(1),
                                      datetime.time(9, 0, 0)))

    end = datetime.datetime.now()
    delta = end - start
    logging.info('Completed simulated observing in %d:%2.1f' %
                 (delta.total_seconds() / 60, delta.total_seconds() % 60.))

    # Generate a local time for end-of-night operations
    local_time_now = ts.localize_utc_dt(local_utc_now)

    # We are now done observing for the night. It is time for some
    # housekeeping

    logging.info('%d tiles were observed' % len(tiles_observed))
    logging.info('Observing done at %s local' %
                 local_time_now.strftime('%Y-%m-%d %H:%M:%S'))

    start = datetime.datetime.now()
    if len(tiles_observed) > 0 and not instant_dq:
        sim_dq_analysis(cursor, tiles_observed, tiles_observed_at)

        # Re-tile the affected fields
        # Work out which fields actually need re-tiling
        fields_to_retile = rCAexec(cursor, tile_list=tiles_observed)
        logging.info('This requires %d fields be re-tiled' %
                     len(fields_to_retile))
        # Re-tile those fields to a particular depth - usually 1
        # Note that the calls made by the tiling function automatically include
        # a re-computation of the target numbers in each field
        # logger.setLevel(logging.DEBUG)
        # Note we now need to retile fields which have is_queued=True and
        # is_observed=False, as they're no longer actually queued
        retile_fields(cursor, fields_to_retile, tiles_per_field=1,
                      tiling_time=local_utc_now,
                      disqualify_below_min=False,
                      delete_queued=True, bins=int(len(fields_to_retile)/15))
        # logger.setLevel(logging.INFO)

    end = datetime.datetime.now()
    delta = end - start
    logging.info('Completed simulated end-of-night tasks in %d:%2.1f' %
                 (delta.total_seconds() / 60, delta.total_seconds() % 60.))

    logging.info('Completed simulated observing for %s' %
                 date.strftime('%y-%m-%d'))

    if commit:
        cursor.connection.commit()

    return local_utc_now


def execute(cursor, date_start, date_end, output_loc='.', prep_db=True,
            instant_dq=False, seed=None):
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

    Returns
    -------
    Nil. Tiling outputs are written to the database (to simulate the action of
    the virtual observer). Anything generated and written out to file will end
    up in output_loc (although currently nothing is).
    """

    # Seed the random number generator
    if seed is not None:
        random.seed(seed)

    if prep_db:
        start = datetime.datetime.now()
        sim_prepare_db(cursor,
                       prepare_time=datetime.datetime.combine(
                           date_start, datetime.time(12, 0)),
                       commit=True)
        end = datetime.datetime.now()
        delta = end - start
        logging.info('Completed DB prep in %d:%2.1f' %
                     (delta.total_seconds() / 60, delta.total_seconds() % 60.))

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

    # Run the actual observing
    logging.info('Commencing observing...')
    curr_date = date_start
    while curr_date <= date_end:
        sim_do_night(cursor, curr_date, date_start, date_end,
                     almanac_dict=almanacs, dark_almanac=dark_almanac,
                     instant_dq=instant_dq,
                     commit=True)
        curr_date += datetime.timedelta(1.)
        if curr_date == datetime.date(2017,5,1):
            break

    logging.info('----------')
    logging.info('OBSERVING COMPLETE')
    logging.info('----------')


if __name__ == '__main__':

    sim_start = datetime.date(2017, 4, 1)
    sim_end = datetime.date(2018, 4, 1)
    global_start = datetime.datetime.now()

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
    logging.info('Executing fullsurvey.py as file')
    # logging.info('*** THIS IS AN INSTANT-FEEDBACK SIMULATION')

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    logging.debug('Doing execute function')
    execute(cursor, sim_start, sim_end,
            instant_dq=False,
            output_loc='.', prep_db=True)

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
