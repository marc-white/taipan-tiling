# Simulate a full tiling of the Taipan galaxy survey


"""
This module was the initial simulation to be set up.
"""

import sys
import logging
import taipan.tiling as tl
import taipan.scheduling as ts

from .utils.tiling import retile_fields
from .utils.bugfail import simulate_bugfails
from .utils.updatesci import update_science_targets

import pickle
import numpy as np
import datetime
import random
from functools import partial
import multiprocessing
from joblib import Parallel, delayed

from taipandb.resources.stable.readout.readCentroids import execute as rCexec
from taipandb.resources.stable.readout.readGuides import execute as rGexec
from taipandb.resources.stable.readout.readStandards import execute as rSexec
from taipandb.resources.stable.readout.readScience import execute as rScexec
from taipandb.resources.stable.readout.readSkies import execute as rSkexec
from taipandb.resources.stable.readout.readTileScores import execute as rTSexec
from taipandb.resources.stable.readout.readCentroidsAffected import execute as rCAexec
from taipandb.resources.stable.readout.readScienceTypes import execute as rSTyexec
from taipandb.resources.stable.readout.readScienceTile import execute as rSTiexec
from taipandb.resources.stable.readout.readSciencePosn import execute as rSPexec
from taipandb.resources.stable.readout.readScienceVisits import execute as rSVexec
from taipandb.resources.stable.readout.readCentroidsByTarget import execute as \
    rCBTexec
import taipandb.resources.stable.readout.readAlmanacStats as rAS

from taipandb.resources.stable.insert.insertTiles import execute as iTexec

from taipandb.resources.stable.manipulate.makeScienceVisitInc import execute as mSVIexec
from taipandb.resources.stable.manipulate.makeScienceRepeatInc import execute as mSRIexec
from taipandb.resources.stable.manipulate.makeTilesObserved import execute as mTOexec
from taipandb.resources.stable.manipulate.makeTilesQueued import execute as mTQexec
from taipandb.resources.stable.manipulate.makeTargetPosn import execute as mTPexec
from taipandb.resources.stable.manipulate.makeTilesReset import execute as mTRexec
from taipandb.resources.stable.manipulate.makeScienceTypes import execute as mScTyexec
from taipandb.resources.stable.manipulate.makeSciencePriorities import execute as mScPexec
from taipandb.resources.stable.manipulate.makeScienceDiff import execute as mSDexec
from taipandb.resources.stable.manipulate import makeObservingLog as mOL

import taipandb.resources.stable.manipulate.makeNSciTargets as mNScT

from taipandb.scripts.connection import get_connection

from .simulate import test_redshift_success
import taipan.simulate.logic as tsl

SIMULATE_LOG_PREFIX = 'SIMULATOR: '

USE_NEW_FIELDS_AVAIL = True

# HELPER FUNCITONS


def _hours_obs_reshuffle(f,
                         # cursor=get_connection().cursor(),
                        dt=None, datetime_to=None,
                         hours_better=True, airmass_delta=0.05):
    """
    Helper function for parallelizing the determination of hours observable

    Parameters
    ----------
    f : :obj:`int`
        Field ID
    dt : :obj:`datetime.datetime`
        Current datetime
    datetime_to : :obj:`datetime.datetime`
        Datetime to consider to for the computation of hours available.
    hours_better : :obj:`bool`
        See documentation for :any:`readAlmanacStats.hours_observable`.
    airmass_delta : obj:`float`
        See documentation for :any:`readAlmanacStats.hours_observable`.

    Returns
    -------
    hrs: :obj:`float`
        The number of hours that this field remains observable over the
        prescribed time period.
    """
    # Multipool process needs its own cursor
    # if cursor is None:
    #     cursor = get_connection().cursor()
    with get_connection().cursor() as cursor_int:
        hrs = rAS.hours_observable(cursor_int, f, dt,
                                   datetime_to=datetime_to,
                                   hours_better=hours_better,
                                   airmass_delta=airmass_delta)
    return hrs


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
        period = rAS.next_observable_period(
            cursor_int, f, dt,
            datetime_to=per_end,
        )
    return period


def sim_prepare_db(cursor, prepare_time=datetime.datetime.now(),
                   assign_sky_fibres=True,
                   commit=True):
    """
    Perform initial database set-up at the start of the run

    This initial step prepares the database for the simulation run by getting
    the fields in from the database, performing the initial tiling of fields,
    and then returning that information to the database for later use.

    Note that this function does *not* do any target loading, almanac loading
    or related functions - these should have been done previously using the
    utitlies available in :any:`taipandb`.

    Parameters
    ----------
    cursor : :obj:`psycopg2.connection.cursor`
        psycopg2 cursor for interacting with the database.
    prepare_time: :obj:`datetime.datetime`
        Optional; datetime.datetime instance that the database should be
        considered to be prepared at (e.g., initial tiles will have
        date_config set to this). Defaults to datetime.datetime.now().
    commit: :obj:`bool`
        Optional; Boolean value denoting whether to hard-commit changes to
        actual database. Defaults to True.
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
        if assign_sky_fibres:
            sky_targets = rSkexec(cursor)

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
                                              sky_targets=sky_targets if
                                              assign_sky_fibres else None,
                                              tiles=field_tiles,
                                              repeat_targets=True,
                                              tile_unpick_method='sequential',
                                              sequential_ordering=(2, 1),
                                              multicores=7,
                                              )
        # logger.setLevel(old_level)
        logging.info('First tile pass complete!')

        # 'Pickle' the tiles so they don't need
        # to be regenerated later for tests
        with open('tiles.pobj', 'w') as tfile:
            pickle.dump(candidate_tiles, tfile)

    # Write the tiles to DB
    iTexec(cursor, candidate_tiles, config_time=prepare_time,
           disqualify_below_min=False, remove_index=True)
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
                    prob_vpec_first=0.18,
                    prob_vpec_second=0.510,
                    prob_vpec_third=0.675,
                    prob_vpec_fourth=0.845,
                    prob_lowz_each=0.8, prisci=False,
                    do_diffs=False,
                    hrs_better=None, airmass=None):
    """
    Perform simulated data analysis on simulated observations

    This function is effectively a wrapper to
    :any:`taipan.simulate.simulate.test_redshift_success`.
    It also handles all the
    necessary database options around the actual determination of the
    redshift success. The procedure is:

    - Get the target IDs of all observed targets
    - Run `taipan.simulate.test_redshift_success` on these targets
    - Record the result to the observing log first
    - Increment the visits or repeats number of the targets in the database
      as appropriate (visits increments if the observation was unsuccessful,
      repeats increments and visits goes to 0 if successful)
    - Update the science target information (e.g. priorities) based on the
      updated success/visits/repeats
    - Mark the tiles passed in as 'observed' in the database
    - Set all tiles in the database to be unqueued

    Note that this algorithm has two important assumptions:

    - Targets will only be observed once on the tile set passed in. Therefore,
      this function should be called before every attempt to re-tile
      any fields based on observational results.
    - *All* tiles marked as 'queued' in the simulator (that is, sent to be
      observed, but status currently unknown) should be passed to this function
      at once. The last step of the algorithm above (set all tiles to unqueued)
      is a safety check against errant flags, and will destroy the queued
      status of any tiles that are legitimately still queued, even if they've
      not been passed to this function.


    Parameters
    ----------
    cursor : :obj:`psycopg2.connection.cursor`
        Cursor for database communication
    tiles_observed : :obj:`list` of :any:`TaipanTile`
        The list of :any:`TaipanTile` observed
    tiles_observed_at : :obj:`list` of :obj:`datetime.datetime`
        The list of times that ``tiles_observed`` were observed at. There
        is a one-to-one correspondence between tiles and observing times.
    prob_bugfail : :obj:`float`
        The probability of any one Starbug failing (e.g. failing to find
        correct position, falling off the plate, other malfunction) for any
        one observation. Must be <= 1.
    prob_vpec_first : :obj:`float`
        Probability that a peculiar velocity target will be successfully
        observed on the first visit. Must be <= 1.
    prob_vpec_second : :obj:`float`
        Probability that a peculiar velocity target will be successfully
        observed on the second visit. Must be <= 1.
    prob_vpec_third : :obj:`float`
        Probability that a peculiar velocity target will be successfully
        observed on the second visit. Must be <= 1.
    prob_vpec_fourth : :obj:`float`
        Probability that a peculiar velocity target will be successfully
        observed on the second visit. Must be <= 1.
    prob_lowz_each : :obj:`float`
        Probability that a low redshift science target will be successfully
        observed on *any* visit. Must be <= 1.
    prisci : :obj:`bool`
        Flag denoting whether to use priority science or main survey tiling
        schema.
    do_diffs : :obj:`bool`
        Flag denoting whether to recompute target difficulties are evaluating
        observation success.
    hrs_better : :obj:`float`
        DEPRECATED. The hours_better value of the tile when observed. This
        functionality is now handled by the observing_log database table; should
        leave this as default value (:obj:`None`).
    airmass : obj:`float`
        DEPRECATED. The airmass the tile when observed. This
        functionality is now handled by the observing_log database table; should
        leave this as default value (:obj:`None`).
    """
    # -------
    # FAKE DQ/SCIENCE ANALYSIS
    # -------
    # Read in from DB
    # Get the list of science target IDs on these tiles
    target_ids = np.asarray(rSTiexec(cursor, tiles_observed))

    # Add in small probability of uncategorized 'bug failure'
    # Note that we do this here so that targets where the bug has failed *don't*
    # get their visits number incremented - after all, we'd get no data at all!
    target_ids = target_ids[simulate_bugfails([True] * len(target_ids),
                                              prob=prob_bugfail)]

    if len(target_ids) > 0:
        # Get the array of target_ids with target types from the database
        target_types_db = rSTyexec(cursor, target_ids=target_ids)
        # Get an array with the number of visits and repeats of these
        visits_repeats = rSVexec(cursor, target_ids=target_ids)

        if len(target_types_db) != len(visits_repeats):
            raise RuntimeError('Arrays of target types and visits do not '
                               'match in length!')

        # Sort these arrays by target_id to make sure they correspond with
        # one another
        target_types_db.sort(order='target_id')
        visits_repeats.sort(order='target_id')
        target_ids.sort()

        # Work out if there will be new vpec targets
        # Note that anything observed at this point will have is_vpec_target
        # set is is_full_vpec_target, so we can just copy to the other
        target_types_db['is_vpec_target'] = np.logical_or(
            target_types_db['is_full_vpec_target'],
            target_types_db['is_vpec_target']
        )

        # Calculate a success/failure rate for each target
        # Compute target success based on target type(s)
        # Note function needs the 'updated' value of target visits, which
        # won't be pushed to the database until after the result of this
        # function is implemented
        # Note that test_redshift_success preserves the list ordering
        success_targets = test_redshift_success(target_types_db,
                                                visits_repeats['visits'] + 1,
                                                prob_vpec_first=prob_vpec_first,
                                                prob_vpec_second=
                                                prob_vpec_second,
                                                prob_vpec_third=prob_vpec_third,
                                                prob_vpec_fourth=
                                                prob_vpec_fourth,
                                                prob_lowz_each=prob_lowz_each)

        # Record the observing log at this point to capture data at time
        # of assignment/observation
        mOL.execute(cursor, tiles_observed, target_ids, success_targets,
                    datetime_at=tiles_observed_at)

        # Set relevant targets as observed successfully, all others
        # observed but unsuccessfully
        mSRIexec(cursor, target_ids[success_targets], set_done=True)
        mSVIexec(cursor, target_ids[~success_targets])

        # Recompute the target priorities and types for the observed targets
        # Will need fresh visits/repeats data
        # target_types_db = rSTyexec(cursor, target_ids=target_ids)
        # target_types_new = tsl.compute_target_types(target_types_db,
        #                                             prisci=prisci)
        # mScTyexec(cursor, target_types_new['target_id'],
        #           target_types_new['is_h0_target'],
        #           target_types_new['is_vpec_target'],
        #           target_types_new['is_lowz_target'])
        # target_types_db = rSTyexec(cursor, target_ids=target_ids)
        # new_priors = tsl.compute_target_priorities_tree(target_types_db,
        #                                                 prisci=prisci)
        # mScPexec(cursor, target_types_db['target_id'], new_priors)
        # if do_diffs:
        #     # Difficulties need to be re-done after types modified
        #     # This needs to be done for the field observed, and all affected fields
        #     # mSDexec(
        #     #     cursor,
        #     #     target_list=rSPexec(cursor,
        #     #                         field_list=rCAexec(cursor,
        #     #                                            tile_list=tiles_observed)
        #     #                         )['target_id'])
        #     # However, this is quite slow
        #     # What we really need to do is do it for this tile, and all targets
        #     # within TILE_RADIUS + EXCLUSION RADIUS of the field centre
        #     # To do this, read in all targets in the affected fields, and then
        #     # restrict them to a certain distance from the tile centre
        #     # Read in targets from affected fields
        #     # Need the tile info
        #     tgts = rScexec(cursor, field_list=rCAexec(
        #         cursor, tile_list=tiles_observed)
        #                    )
        #     # Reduce the target list to those within TILE_RADIUS + EXCLUSION_RADIUS
        #     # of the observed tile
        #     tile_data = rCexec(cursor, tile_list=tiles_observed)
        #     # print tile_data
        #     # print tgts
        #     tgts = list(set(
        #         np.concatenate([tp.targets_in_range(t.ra, t.dec, tgts,
        #                                             tp.TILE_RADIUS +
        #                                             tp.FIBRE_EXCLUSION_RADIUS)
        #                         for t in tile_data])))
        #     mSDexec(cursor, target_list=[t.idn for t in tgts])

        # UDDATE 2017-03-02
        # Use the new super-utility function to do this efficiently
        update_science_targets(cursor, target_list=target_ids, do_d=do_diffs,
                               prisci=prisci)

    # Mark the tiles as having been observed
    mTOexec(cursor, tiles_observed, time_obs=tiles_observed_at,
            hrs_better=hrs_better, airmass=airmass)

    # Reset all tiles to be unqueued (none should be, but this is just
    # a safety measure)
    mTRexec(cursor)

    return


def select_best_tile(cursor, dt, per_end,
                     midday_end,
                     midday_start=None,
                     prioritize_lowz=True,
                     resolution=15.,
                     multipool_workers=multiprocessing.cpu_count(),
                     dark=True, grey=False):
    """
    Select the best tile to observe at the current datetime.

    The algorithm for determining the best tile is as follows:

    - Read in the tile score information from the database
    - Use the almanac information stored in the database to determine which
      tiles will be available for observation within the current time period.
    - For those tiles, compute an 'hours remaining' statistic from the
      almanacs database. This is the number of hours that the field will
      be observable over some period of time (which may be evaluated differently
      depending on which targets are in which fields, whether we are in
      priority or main survey operations, etc.)
    - Compute the number of targets awaiting observation on the available
      fields.
    - Compute the final tile score (see below)
    - Return the primary key of the tile to observe (or :obj:`None` if no tile
      is available).

    The final tile score is computed as:

    .. math::
        \\frac{\\textrm{metric} \\times \\textrm{no. of targets rem.} }{ \\textrm{no of hours rem.}}

    Note that:

    - There may be a priority threshold cut on the targets used to compute the
      number of targets remaining in the field;
    - The number of hours remaining for the field corresponds to the number
      of *better* (i.e. lower airmass) hours available out to some end time.
      This end time may vary depending on the survey observing strategy, and
      may be either a fixed or rolling date.

    This calculation has the effect of:

    - Prioritising fields that have a lot of targets left to go through;
    - Prioritising fields that will not be available very often in the upcoming
      months/years.

    Only tiles which are marked as unqueued and unobserved in the database will
    be considered.

    Note that the function will analyse observability at the given time with
    no interest in configure time etc. - users should therefore pass the time
    when the observation (of length :any:`taipan.scheduling.POINTING_TIME`)
    is expected to start.

    Parameters
    ----------
    cursor : :obj:`psycopg2.connection.cursor`
        For interacting with the database
    dt : :obj:`datetime.datetime`
        Current datetime (should be naive, but in UTC)
    per_end : :obj:`datetime.datetime`
        End of the time bracket (e.g. dark time ends) when tiles should
        be considered to. Used to compute field observability.
    midday_end: :obj:`datetime.datetime`
        Midday after the priority science period finishes,
        transformed to UTC
    midday_start: :obj:`datetime.datetime`
        Midday before the survey begins,
        transformed to UTC. Defaults to None.
    prioritize_lowz : :obj:`bool`
        Whether or not to prioritize fields with hurry-up targets by computing
        the hours_observable for those fields against a set end date, and
        compute all other fields against a rolling end date.
        Defaults to True.
    resolution: :obj:`float`, minutes
        Denotes the resolution of the almanacs being used. Defaults to 15.
    multipool_workers : :obj:`int`
        The number of pool workers to use in parallel computations/retrievals
        from the database. Defaults to :any:`multiprocessing.cpu_count`.
    dark, grey: :obj:`bool`
        Boolean values denoting whether to return the next observable period
        of grey time, dark time, or all night time (which corresponds to both
        dark and grey being False). Trying to pass ``dark=True`` and
        ``grey=True`` will
        raise a ValueError.

    Returns
    -------
    tile_pk: :obj:`int` or :obj:`None`
        The primary key of the tile that should be observed. Returns :obj:`None`
        if no tile is available.
    fields_available, tiles_scores, scores_array, field_periods, fields_by_tile,
    hours_obs : :any:`numpy` arrays
        Data used to compute the best tile to observe. Designed to be fed into
        the :any:`check_tile_choice` function.
    """
    # Get the latest scores_array
    logging.info('Fetching scores array...')
    scores_array = rTSexec(cursor, metrics=[
        # 'cw_sum',
        'prior_sum',
        'n_sci_rem',
    ],
                           ignore_zeros=False, unobserved_only=True,
                           ignore_empty=True
                           )
    logging.info('-- Number of waiting tiles: %d' % len(scores_array))
    scores_array.sort(order='field_id')

    logging.info('Fetching next field periods...')

    logging.info('-- Finding available fields')
    fields_available = rAS.find_fields_available(
        cursor, dt, datetime_to=per_end,
        field_list=list(scores_array['field_id']),
        resolution=resolution,
        dark=dark, grey=grey
    )
    fields_available.sort(order='field_id')

    nop_workers = 1  # Set to 1 for single-thread next_observable_period,
    # set to multipool_workers to utilize threading

    logging.info('-- Computing field periods')

    if USE_NEW_FIELDS_AVAIL:
        fields_available = rAS.get_fields_available_pointing(
            cursor, dt, minimum_airmass=2.0,
            resolution=resolution,
            pointing_time=ts.POINTING_TIME,
        )
        fields_available = fields_available['field_id']
        field_periods = None
    else:
        if nop_workers == 1:
            field_periods = {r['field_id']: rAS.next_observable_period(
                cursor, r['field_id'], dt,
                datetime_to=per_end,
            ) for
                r in
                # scores_array
                fields_available
            }
        else:
            field_periods_partial = partial(_field_period_reshuffle, dt=dt,
                                            per_end=per_end)
            pool = multiprocessing.Pool(nop_workers)
            field_periods = pool.map(field_periods_partial,
                                     fields_available['field_id'])
            pool.close()
            pool.join()
            field_periods = {fields_available['field_id'][i]: field_periods[i] for i
                             in range(len(field_periods))}

        # logging.debug('Next observing period for each field:')
        # logging.debug(field_periods)
        # logging.info('Next available field will rise at %s' %
        #              (min([v[0].strftime('%Y-%m-%d %H:%M:%S') for v in
        #                    field_periods.values() if
        #                    v[0] is not None]),)
        #              )
        fields_available = [f for f, v in field_periods.items() if
                            v[0] is not None and v[0] < per_end]
        logging.debug('%d fields available at some point tonight' %
                      len(fields_available))
        # Further trim fields_available for to account for field observability
        # at the time of observation
        logging.info('-- Trimming fields_available to now')
        fields_available = [f for f in fields_available if
                            field_periods[f][0] is not None and
                            field_periods[f][1] is not None and
                            field_periods[f][0] < dt and
                            field_periods[f][1] > dt + datetime.timedelta(
                                seconds=ts.OBS_TIME)
                            ]
    logging.info('Currently %d fields available for observation' %
                 len(fields_available))

    # Rank the available fields
    logging.info('Computing field scores')
    start = datetime.datetime.now()
    logging.info('-- Forming tile score dictonary')
    tiles_scores_raw = {row['tile_pk']: (row['n_sci_rem'], row['prior_sum'])
                        for
                        row in scores_array if
                        row['field_id'] in fields_available}
    logging.info('  -- There are now %d tiles in contention' %
                 len(tiles_scores_raw))
    fields_by_tile = {row['tile_pk']: row['field_id'] for
                      row in scores_array if
                      row['field_id'] in fields_available}

    hours_obs_stan_partial = partial(_hours_obs_reshuffle,
                                     # cursor=cursor,
                                     dt=dt,
                                     datetime_to=dt + datetime.timedelta(
                                         365. * 2.),
                                     hours_better=True,
                                     airmass_delta=0.05
                                     )

    hours_obs_lowz_partial = partial(_hours_obs_reshuffle,
                                     # cursor=cursor,
                                     dt=dt,
                                     datetime_to=max(
                                         # End of lowz period, less 6 month
                                         # buffer
                                         # midday_end
                                         # - datetime.timedelta(
                                         #     days=(7./12.)*365.)
                                         # ,
                                         # 12 months from start of survey
                                         midday_start +
                                         datetime.timedelta(days=365.),
                                         dt +
                                         datetime.
                                         timedelta(365./2.)
                                     ),
                                     hours_better=True,
                                     airmass_delta=0.05
                                     )

    fields_to_calculate = list(set(fields_by_tile.values()))
    fields_to_calculate.sort()

    logging.info('-- Calculating scores')
    if prioritize_lowz:
        lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                               unobserved=True, threshold_value=30,
                               assigned_only=True)
        lowz_of_interest = [f for f in fields_to_calculate if f in lowz_fields]
        stan_fields = [f for f in fields_to_calculate if f not in
                       lowz_of_interest]

        if multipool_workers > 1:

            pool = multiprocessing.Pool(multipool_workers)
            hrs = pool.map(hours_obs_lowz_partial, lowz_of_interest)
            pool.close()
            pool.join()
            hours_obs_lowz = {lowz_of_interest[i]: hrs[i] for i in
                              range(len(lowz_of_interest))}

            pool = multiprocessing.Pool(multipool_workers)
            hrs = pool.map(hours_obs_stan_partial, stan_fields)
            pool.close()
            pool.join()
            hours_obs_oth = {stan_fields[i]: hrs[i] for i in
                             range(len(stan_fields))}

            # hrs = Parallel(n_jobs=multipool_workers, backend='multiprocessing')(
            #     delayed(hours_obs_reshuffle)(f, cursor=cursor,
            #                                  dt=dt,
            #                                  datetime_to=max(
            #                                      midday_end
            #                                      - datetime.timedelta(
            #                                          days=(7. / 12.) * 365.)
            #                                      ,
            #                                      dt +
            #                                      datetime.
            #                                      timedelta(365. / 2.)
            #                                  ),
            #                                  hours_better=True,
            #                                  airmass_delta=0.05)
            #     for f in lowz_of_interest
            # )
            # hours_obs_lowz = {lowz_of_interest[i]: hrs[i] for i in
            #                   range(len(lowz_of_interest))}
            #
            # hrs = Parallel(n_jobs=multipool_workers, backend='multiprocessing')(
            #     delayed(hours_obs_reshuffle)(f, cursor=cursor,
            #                                  dt=dt,
            #                                  datetime_to=dt + datetime.timedelta(
            #                                      365. * 2.),
            #                                  hours_better=True,
            #                                  airmass_delta=0.05)
            #     for f in stan_fields
            # )
            # hours_obs_oth = {stan_fields[i]: hrs[i] for i in
            #                  range(len(stan_fields))}
        else:
            # OLD, LINEAR SCHEMA
            hours_obs_lowz = {f: rAS.hours_observable(cursor, f,
                                                      dt,
                                                      datetime_to=max(
                                                          # End of lowz period,
                                                          # less 6 month
                                                          # buffer
                                                          # midday_end
                                                          # - datetime.timedelta(
                                                          #     days=(7./12.)*365.)
                                                          # ,
                                                          # 12 months from start of survey
                                                          midday_start +
                                                          datetime.timedelta(
                                                              days=365.),
                                                          dt +
                                                          datetime.
                                                          timedelta(365./2.)
                                                      ),
                                                      hours_better=True,
                                                      airmass_delta=0.05) for
                              f in list(set(fields_by_tile.values())) if
                              f in lowz_fields}
            hours_obs_oth = {f: rAS.hours_observable(cursor, f,
                                                     dt,
                                                     datetime_to=
                                                     dt +
                                                     datetime.timedelta(
                                                         365.*2.),
                                                     hours_better=True,
                                                     airmass_delta=0.05) for
                             f in list(set(fields_by_tile.values())) if
                             f not in lowz_fields}
            # hours_obs = dict(hours_obs_lowz, **hours_obs_oth)hours

        hours_obs = hours_obs_lowz.copy()
        hours_obs.update(hours_obs_oth)
    else:
        if multipool_workers > 1:

            pool = multiprocessing.Pool(processes=multipool_workers)
            hrs = pool.map(hours_obs_stan_partial, fields_to_calculate)
            pool.close()
            pool.join()

            # hrs = Parallel(n_jobs=multipool_workers, backend='multiprocessing')(
            #     delayed(hours_obs_reshuffle)(f, cursor=cursor,
            #                                  dt=dt,
            #                                  datetime_to=dt + datetime.timedelta(
            #                                      365. * 2.),
            #                                  hours_better=True,
            #                                  airmass_delta=0.05)
            #     for f in fields_to_calculate
            # )
        else:
            # OLD LINEAR SCHEMA
            hours_obs = {f: rAS.hours_observable(cursor, f, dt,
                                                 datetime_to=
                                                 dt +
                                                 datetime.timedelta(365*2),
                                                 hours_better=True,
                                                 airmass_delta=0.05) for
                         f in fields_by_tile.values()}

        hours_obs = {fields_to_calculate[i]: hrs[i] for i in
                     range(len(fields_to_calculate))}

    # logging.debug('Hours_observable:')
    # logging.debug(hours_obs)

    # Need to replace any points where hours_obs=0 with the almanac resolution;
    # otherwise, 0 hours fields will be forcibly observed, even if their score
    # does not warrant it
    for f in hours_obs.keys():
        if hours_obs[f] < (resolution / 60.):
            hours_obs[f] = resolution / 60.

    tiles_scores = {t: (v[0] * v[1] / hours_obs[fields_by_tile[t]]) for
                    t, v in tiles_scores_raw.items()}
    # logging.debug('Tiles scores: ')
    # logging.debug(tiles_scores)
    end = datetime.datetime.now()
    delta = end - start
    logging.info('Computed tile scores/hours remaining in %d:%2.1f' %
                 (delta.total_seconds() / 60,
                  delta.total_seconds() % 60.))

    # 'Observe' while the remaining time in this dark period is
    # longer than one pointing (slew + obs)

    logging.info('At time %s, considering til %s' % (
        dt.strftime('%Y-%m-%d %H:%M:%S'),
        per_end.strftime('%Y-%m-%d %H:%M:%S'),
    ))

    # Select the best ranked field we can see
    try:
        # logging.debug('Next observing period for each field:')
        # logging.debug(field_periods)
        # Generator pattern
        # tile_to_obs = (t for t, v in sorted(tiles_scores.items(),
        #                                     key=lambda x: -1. * x[1]
        #                                     ) if
        #                field_periods[fields_by_tile[t]][0] is not
        #                None and
        #                field_periods[fields_by_tile[t]][1] is not
        #                None and
        #                field_periods[fields_by_tile[t]][0] -
        #                datetime.timedelta(seconds=ts.SLEW_TIME) <
        #                local_utc_now and
        #                field_periods[fields_by_tile[t]][1] >
        #                local_utc_now +
        #                datetime.timedelta(
        #                    seconds=ts.POINTING_TIME)).next()
        # Full dict pattern
        # tile_to_obs = {t: v for t, v in tiles_scores.items() if
        #                field_periods[fields_by_tile[t]][0] is not
        #                None and
        #                field_periods[fields_by_tile[t]][1] is not
        #                None and
        #                field_periods[fields_by_tile[t]][0] -
        #                datetime.timedelta(seconds=ts.SLEW_TIME) <
        #                local_utc_now and
        #                field_periods[fields_by_tile[t]][1] >
        #                local_utc_now +
        #                datetime.timedelta(
        #                    seconds=ts.POINTING_TIME)}
        # tile_to_obs = max(tile_to_obs, key=tile_to_obs.get)
        # Structured array pattern
        tile_to_obs = np.asarray(
            [(t, v,)
             for t, v in tiles_scores.items()
             # Redundant - time restriction moved to creation of
             # fields_available
             # if
             # field_periods[
             #     fields_by_tile[t]][0] is not
             # None and
             # field_periods[
             #     fields_by_tile[t]][1] is not
             # None and
             # field_periods[
             #     fields_by_tile[t]][0] <
             # dt and
             # field_periods[fields_by_tile[t]][1] >
             # dt +
             # datetime.timedelta(
             #     seconds=ts.OBS_TIME)
             ],
            dtype={
                'names': ['tile_pk', 'final_score'],
                'formats': ['int', 'float64'],
            }
        )
        # tile_to_obs.sort(order='final_score')
        # tile_to_obs = tile_to_obs['tile_pk'][-1]
        best_tile_i = np.argmax(tile_to_obs['final_score'])
        tile_to_obs = tile_to_obs['tile_pk'][best_tile_i]
    # except IndexError:
    except ValueError:
        tile_to_obs = None

    return tile_to_obs, fields_available, tiles_scores, scores_array, \
           field_periods, fields_by_tile, hours_obs


def check_tile_choice(cursor, dt, tile_to_obs, fields_available, tiles_scores,
                      scores_array, field_periods, fields_by_tile, hours_obs,
                      abort=False):
    """
    Do some simple checking of the selection made by :any:`select_best_tile`.

    This function uses the data generated by :any:`select_best_tile`, but uses
    a different algorithm to confirm that the best tile was indeed selected.
    This function should not be required in production.

    Note that the algorithm used by :any:`check_tile_choice` has not been
    extensively tested; hence, we only claim that it is capable of detecting
    if the chosen tile is *likely* to be correct.

    Parameters
    ----------
    cursor : :obj:`psycopg2.connection.cursor`
        For interacting with the database
    dt : :obj:`datetime.datetime`
        Current datetime (should be naive, but in UTC)
    tile_to_obs : :obj:`int`
        The PK of the tile that has been selected
    scores_array, field_periods, fields_by_tile, hours_obs : numpy arrays
        Arrays of numpy data. These match those arrays output by
        :any:`select_best_tile`.
    abort : :obj:`bool`, optional
        Boolean value denoting whether to abort the run if a failure is
        detected. If true, diagnostic information will be dumped to the logging
        system, the cursor will commit to the database, and then sys.exit() will
        be called. Defaults to False.

    Returns
    -------
    likely_correct : Boolean
        Returns True if the tile choice is likely to be correct, False if not.

    """
    row_chosen = scores_array[scores_array['tile_pk'] == tile_to_obs][0]
    ref_score = row_chosen['prior_sum'] * row_chosen['n_sci_rem']
    ref_score /= hours_obs[row_chosen['field_id']]

    # fields_available is those fields available this dark period - we
    # need a list of fields available *now*
    fields_actually_available = [f for f in fields_available
                                 if field_periods[f][0] <=
                                 dt and
                                 field_periods[f][1] >= dt +
                                 datetime.timedelta(seconds=
                                                    ts.OBS_TIME)]

    # Find the tile with the highest raw score (prior_sum), compute its
    # adjusted score, and check against that chosen
    scores_array.sort(order='prior_sum')
    highest_score = scores_array[np.in1d(scores_array['field_id'],
                                         fields_actually_available)][-1]
    highest_calib_score = highest_score['prior_sum'] * highest_score[
        'n_sci_rem']
    highest_calib_score /= hours_obs[highest_score['field_id']]
    if highest_calib_score > ref_score:
        logging.warning('TILE SELECTION FAILURE')
        logging.warning('Tile %d was selected (score %f)' % (tile_to_obs,
                                                             ref_score,))
        logging.warning('Found tile %d, score %f (highest raw score)' %
                        (highest_score['tile_pk'], highest_calib_score))
        if abort:
            sys.exit()
        return False

    # Find the tile with the highest n_sci_rem score (prior_sum), compute its
    # adjusted score, and check against that chosen
    scores_array.sort(order='n_sci_rem')
    highest_score = scores_array[np.in1d(scores_array['field_id'],
                                         fields_actually_available)][-1]
    highest_calib_score = highest_score['prior_sum'] * highest_score[
        'n_sci_rem']
    highest_calib_score /= hours_obs[highest_score['field_id']]
    if highest_calib_score > ref_score:
        logging.warning('TILE SELECTION FAILURE')
        logging.warning(
            'Tile %d was selected (score %f)' % (tile_to_obs,
                                                 ref_score,))
        logging.warning('Found tile %d, score %f (highest n_sci_rem)' %
                        (highest_score['tile_pk'], highest_calib_score))
        if abort:
            sys.exit()
        return False

    # Find the tile with the lowest hours_rem, compute its adjusted score,
    # then check against that chosen
    min_hrs = min([v for v in hours_obs.values()])
    min_hrs_field = [f for f, v in hours_obs.items() if v == min_hrs][0]
    for row in scores_array[scores_array['field_id'] == min_hrs_field]:
        highest_calib_score = row['prior_sum'] * row['n_sci_rem']
        highest_calib_score /= hours_obs[row['field_id']]
        if highest_calib_score > ref_score:
            logging.warning('TILE SELECTION FAILURE')
            logging.warning(
                'Tile %d was selected (score %f)' % (tile_to_obs,
                                                     ref_score,))
            logging.warning('Found tile %d, score %f (lowest hours_obs)' %
                            (highest_score['tile_pk'], highest_calib_score))
            if abort:
                cursor.connection.commit()
                sys.exit()
            return False

    # All good to here, so return True
    return True


def sim_do_night(cursor, date, date_start, date_end,
                 almanac_dict=None, dark_almanac=None,
                 save_new_almanacs=True,
                 instant_dq=False,
                 # prisci=True,
                 check_almanacs=True,
                 commit=True, kill_time=None,
                 prisci_end=None,
                 priority_function=tsl.compute_target_priorities_tree):
    """
    Do a simulated 'night' of observations.

    This involves:

    - Determine the first block of available time for observing (currently
      hardcoded to be 'dark' time), and advance time to then.
    - During the block of available time, until the end of the block is reached:

      - Determine the tiles to do next;
      - 'Observe' it;
      - Update the DB appropriately;
      - Advance time by :any:`taipan.scheduling.POINTING_TIME` and repeat.

    - Repeat the procedure until the end of the night is reached.

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
    prisci : Boolean, optional
        Whether or not to prioritize fields with lowz targets by computing
        the hours_observable for those fields against a set end date, and
        compute all other fields against a rolling one-year end date.
        Defaults to True.
    check_almanacs : Boolean
        Optional; denotes whether the algorithm should check if there is
        siufficient Almanac data in the database to perform the simulation.
        This is simply done by making sure the maximum database table is
        12 months after the current date. If enough data is not found, an
        extra 12 months will be generated after maximum database date detected.
        Note that this will cost several hours of simulation time.
    commit:
        Boolean value, denoting whether to hard-commit the database changes made
        to the database proper. Defaults to True.
    kill_time: datetime.datetime object, optional
        A pre-determined time at which to 'kill' a simulation. Used for
        debugging purposes. Defaults to None.
    prisci_end : datetime.timedelta object, optional
        Denotes for how long after the start of the survey that lowz targets
        should be prioritized. Defaults to None, and which point lowz fields
        will always be prioritized (if prioritize_lowz=True).

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

    # Compute the times for the first block of dark time tonight
    midday_start = ts.utc_local_dt(datetime.datetime.combine(date_start,
                                                             datetime.time(12,
                                                                           0,
                                                                           0)))
    midday = ts.utc_local_dt(datetime.datetime.combine(date,
                                                       datetime.time(12, 0,
                                                                     0)))
    midday_end = ts.utc_local_dt(datetime.datetime.combine(date_end,
                                                           datetime.time(12,
                                                                         0,
                                                                         0)))

    if prisci_end is not None:
        # Change prisci_end to the actual end datetime
        prisci_end = midday_start + prisci_end
        # Compare with midday today to work out if prisci=True or False
        prisci = prisci_end > midday
        logging.info('Prisci status for %s: %s' %
                     (date.strftime('%Y-%m-%d'), bool(prisci))
                     )

    if check_almanacs:
        logging.info('Checking almanacs for night %s' %
                     date.strftime('%Y-%m-%d'))
        almanac_end = rAS.check_almanac_finish(cursor)
        if almanac_end - midday < datetime.timedelta(365.):
            logging.warning('WARNING - almanacs do not hold enough data to '
                            'compute this night correctly')
            sys.exit()

    # Needs to do the following:
    # Read in the tiles that are awaiting observation, along with their scores
    scores_array = rTSexec(cursor, metrics=['cw_sum', 'prior_sum', 'n_sci_rem'],
                           ignore_zeros=False)
    logging.debug('Scores array info:')
    logging.debug(scores_array)

    logging.info('Finding first block of dark time for this evening')
    start = datetime.datetime.now()

    dark_start, dark_end = rAS.next_night_period(cursor, midday,
                                                 limiting_dt=
                                                 midday + datetime.timedelta(1),
                                                 dark=True, grey=False,
                                                 field_id=scores_array[
                                                     'field_id'][0])
    end = datetime.datetime.now()
    delta = end - start
    logging.info('Found first block of dark time in %d:%2.1f' %
                 (delta.total_seconds() / 60, delta.total_seconds() % 60.))

    tiles_observed = []
    tiles_observed_at = []
    tiles_observed_hrs_better = []
    tiles_observed_airmass = []

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
            # Do kill-time logic
            if kill_time is not None:
                if local_utc_now > kill_time:
                    cursor.connection.commit()
                    logging.warning('KILL TIME REACHED - %s' %
                                    kill_time.strftime('%Y-%m-%d %H:%M:%S'))
                    sys.exit()

            # ------
            # FAKE WEATHER FAILURES
            # ------
            # Superseded by night losses in execute
            # P = 0.25
            # weather_prob = np.random.random(1)
            # if weather_prob[0] < P:
            #     local_utc_now += datetime.timedelta(ts.POINTING_TIME)
            #     logging.info('*BREAK* Lost one pointing to weather, '
            #                  'advancing to %s' % (4
            #         local_utc_now.strftime('%Y-%m-%d %H:%M:%S'),
            #     ))
            #     continue

            if prisci_end is not None:
                prioritize_lowz_today = prisci and (local_utc_now <
                                                    prisci_end)
                midday_end_prior = prisci_end
            else:
                prioritize_lowz_today = prisci
                midday_end_prior = midday_end
            # Pick the best tile
            tile_to_obs, fields_available, tiles_scores, scores_array, \
            field_periods, fields_by_tile, hours_obs = select_best_tile(
                cursor, local_utc_now,
                dark_end, midday_end_prior,
                prioritize_lowz=prioritize_lowz_today,
                midday_start=midday_start,
            )
            if tile_to_obs is None:
                # This triggers if fields will be available later tonight,
                # but none are up right now. What we do now is advance time_now
                # to the first time when any field becomes available
                if USE_NEW_FIELDS_AVAIL:
                    local_utc_now = rAS.next_time_available(
                        cursor, local_utc_now,
                        end_dt=dark_end, minimum_airmass=2.0,
                        resolution=15.,
                    )
                else:
                    local_utc_now = min([v[0] for f, v in
                                         field_periods.items()
                                         if v[0] is not None and
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
            # else:
            #     _ = check_tile_choice(cursor, local_utc_now, tile_to_obs,
            #                           fields_available, tiles_scores,
            #                           scores_array, field_periods,
            #                           fields_by_tile, hours_obs,
            #                           abort=False)

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
            tiles_observed_hrs_better.append(
                hours_obs[fields_by_tile[tile_to_obs]]
            )
            tiles_observed_airmass.append(rAS.get_airmass(
                cursor, fields_by_tile[tile_to_obs], local_utc_now
            )['airmass'][0])

            if instant_dq:
                # Do the DQ analysis now
                sim_dq_analysis(cursor, [tile_to_obs], [local_utc_now],
                                do_diffs=False, prisci=prisci,
                                hrs_better=[tiles_observed_hrs_better[-1], ],
                                airmass=[tiles_observed_airmass[-1], ])

            # Re-tile the affected areas (should be 7 tiles, modulo any areas
            # where we have deliberately added an over/underdense tiling)
            fields_to_retile = rCAexec(cursor, tile_list=[tile_to_obs])
            # Re-do the priorities just to be sure
            update_science_targets(cursor, field_list=fields_to_retile,
                                   do_tp=True, do_d=False,
                                   priority_function=priority_function,
                                   prisci=prisci)
            logging.info('Retiling fields %s' % ', '.join(str(i) for i in
                                                          fields_to_retile))
            # Switch the logger to DEBUG
            # logger.setLevel(logging.DEBUG)
            retile_fields(cursor, fields_to_retile, tiles_per_field=1,
                          tiling_time=local_utc_now,
                          disqualify_below_min=False,
                          multicores=4,
                          bins=len(fields_to_retile)/20 + 1
                          # prisci=prioritize_lowz_today,
                          )
            # logger.setLevel(logging.INFO)
            # Test use only
            # if fields_by_tile[tile_to_obs] == 1592:
            #     logging.warning('Reached field 1592 - ABORT ABORT ABORT')
            #     sys.exit()

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
                                                     local_utc_now +
                                                     datetime.timedelta(
                                                         ts.POINTING_TIME),
                                                     limiting_dt=
                                                     midday +
                                                     datetime.timedelta(1))

    try:
        _ = local_utc_now
    except (NameError, UnboundLocalError, ):
        # This means that there was no dark period at all tonight.
        # So, we need to set local_time_now to be a reasonable time the
        # following morning (say 10am local)
        local_utc_now = ts.utc_local_dt(
            datetime.datetime.combine(date + datetime.timedelta(1),
                                      datetime.time(10, 0, 0)))

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
        sim_dq_analysis(cursor, tiles_observed, tiles_observed_at,
                        do_diffs=False, prisci=prisci,
                        hrs_better=tiles_observed_hrs_better,
                        airmass=tiles_observed_airmass)

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
                      delete_queued=True, bins=int(len(fields_to_retile)/15),
                      multicores=4)
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
            instant_dq=False, seed=None, kill_time=None,
            prior_lowz_end=None, weather_loss=0.4):
    """
    Execute the simulation.

    .. warning::
        This function encapsulates the initial simulation scheme considered.
        It should now be considered redundant, and other simulation packages
        (e.g. :any:`fullsurvey_baseline`) should be constructed & run instead.

    Tiling outputs are written to the database (to simulate the action of
    the virtual observer). Anything generated and written out to file will end
    up in output_loc (although currently nothing is).

    Parameters
    ----------
    cursor: :obj:`psycopg2.connection.cursor`
        psycopg2 cursor for communicating with the database.
    output_loc: :obj:`str`
        String providing the path for placing the output plotting images.
        Defaults to '.' (ie. the present working directory). Directory must
        already exist.
    date_start: :obj:`datetime.date`
        The start date of the simulated observing period. Should be a
        datetime.date instance.
    date_end: :obj:`datetime.date`
        The final day of the simulated observing period. Should be a
        datetime.date instance.
    prep_db: :obj:`bool`
        Boolean value, denoting whether or not to invoke the sim_prepare_db
        function before beginning the simulation. Defaults to True.
    instant_dq: :obj:`bool`
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
    prior_lowz_end : :obj:`datetime.timedelta`, optional
        Denotes for how long after the start of the survey that lowz targets
        should be prioritized. Defaults to None, and which point lowz fields
        will always be prioritized (if prioritize_lowz=True).
    weather_loss: :obj:`float`, in the range :math:`[0, 1)`
        Percentage of nights lost to weather every calendar year. The nights
        to be lost will be computed at the start of each calendar year, to
        ensure exactly 40% of nights are lost per calendar year (or part
        thereof).

    Returns
    -------
    :obj:`None`
    """

    # Seed the random number generator
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if prep_db:
        start = datetime.datetime.now()
        sim_prepare_db(cursor,
                       prepare_time=ts.utc_local_dt(
                           datetime.datetime.combine(
                               date_start, datetime.time(12, 0))),
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
    #                      for (k, v) in almanacs.items()}
    # logging.info('Saving almanacs to disc...')
    # for k in [k for (k, v) in almanacs_existing.items() if not v]:
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
        if weather_fails[curr_date] > weather_fail_thresh:
            sim_do_night(cursor, curr_date, date_start, date_end,
                         almanac_dict=almanacs, dark_almanac=dark_almanac,
                         instant_dq=instant_dq, check_almanacs=False,
                         commit=True, kill_time=kill_time,
                         prisci_end=prior_lowz_end)
        else:
            logging.info('WEATHER LOSS: Lost %s to weather' %
                         curr_date.strftime('%Y-%m-%d'))
        curr_date += datetime.timedelta(1.)
        if curr_date not in weather_fails.keys():
            curr_date = date_start
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

    sim_start = datetime.date(2017, 9, 1)
    sim_end = datetime.date(2022, 6, 1)
    global_start = datetime.datetime.now()
    prior_lowz_end = datetime.timedelta(days=365.)

    kill_time = None
    # kill_time = datetime.datetime(2017, 7, 23, 9, 25, 0)

    # Override the sys.excepthook behaviour to log any errors
    # http://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def excepthook_override(exctype, value, tb):
        logging.error(
            'My Error Information\nType: %s\nValue: %s\nTraceback: %s' %
            (exctype, value, tb, ))
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
    logging.info('Executing fullsurvey.py as file')
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
            output_loc='.', prep_db=True, kill_time=kill_time,
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
