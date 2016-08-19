# Simulate a full tiling of the Taipan galaxy survey

import sys
import logging
import taipan.core as tp
import taipan.tiling as tl
import taipan.scheduling as ts
import simulate as tsim

from utils.tiling import retile_fields

import pickle

import numpy as np
import atpy
import ephem
import operator
import os
import datetime

from src.resources.v0_0_1.readout.readCentroids import execute as rCexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readScience import execute as rScexec
from src.resources.v0_0_1.readout.readTileScores import execute as rTSexec
from src.resources.v0_0_1.readout.readCentroidsAffected import execute as rCAexec
from src.resources.v0_0_1.readout.readScienceTypes import execute as rSTyexec
from src.resources.v0_0_1.readout.readScienceTile import execute as rSTiexec
from src.resources.v0_0_1.readout.readScienceVisits import execute as rSVexec

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec

from src.resources.v0_0_1.manipulate.makeScienceVisitInc import execute as mSVIexec
from src.resources.v0_0_1.manipulate.makeScienceRepeatInc import execute as mSRIexec
from src.resources.v0_0_1.manipulate.makeTilesObserved import execute as mTOexec

import src.resources.v0_0_1.manipulate.makeNSciTargets as mNScT

from src.scripts.connection import get_connection

from simulate import test_redshift_success

SIMULATE_LOG_PREFIX = 'SIMULATOR: '


def sim_prepare_db(cursor):
    """
    This initial step prepares the database for the simulation run by getting
    the fields in from the database, performing the initial tiling of fields,
    and then returning that information to the database for later use.

    Parameters
    ----------
    cursor

    Returns
    -------

    """

    # Ge the field centres in from the database
    logging.info(SIMULATE_LOG_PREFIX+'Loading targets')
    field_tiles = rCexec(cursor)
    candidate_targets = rScexec(cursor)
    guide_targets = rGexec(cursor)
    standard_targets = rSexec(cursor)

    logging.info(SIMULATE_LOG_PREFIX+'Generating first pass of tiles')
    # TEST ONLY: Trim the tile list to 10 to test DB write-out
    # field_tiles = random.sample(field_tiles, 40)
    candidate_tiles = tl.generate_tiling_greedy_npasses(candidate_targets,
                                                        standard_targets,
                                                        guide_targets,
                                                        1,
                                                        tiles=field_tiles,
                                                        )
    logging.info('First tile pass complete!')

    # 'Pickle' the tiles so they don't need to be regenerated later for tests
    with open('tiles.pobj', 'w') as tfile:
        pickle.dump(candidate_tiles, tfile)

    # Write the tiles to DB
    iTexec(cursor, candidate_tiles)

    # Compute the n_sci_rem and n_sci_obs for these tiles
    mTR.execute(cursor)

    return


def sim_do_night(cursor, date, date_start, date_end,
                 almanac_dict=None, dark_almanac=None,
                 save_new_almanacs=True):
    """
    Do a simulated 'night' of observations. This involves:
    - Determine the tiles to do tonight
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

    Returns
    -------
    Nil. All actions are internal or apply to the database.

    """
    logging.info('Doing simulated observing for %s' % date.strftime('%y-%m-%d'))
    # Do some input checking
    # Date needs to be in the range of date_start and date_end
    if date < date_start or date > date_end:
        raise ValueError('date must be in the range [date_start, date_end]')

    logging.info('Starting observing for night %s' %
                 date.strftime('%Y-%m-%d'))

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
    scores_array = rTSexec(cursor, metrics=['cw_sum', 'n_sci_rem'])
    logging.debug('Scores array info:')
    logging.debug(scores_array)

    # Make sure we have an almanac for every field in the scores_array for the
    # correct date
    # If we don't, we'll need to make one
    # Note that, because of the way Python's scoping is set up, this will
    # permanently add the almanac to the input dictionary
    logging.debug('Checking all necessary almanacs are present')
    almanacs_existing = almanac_dict.keys()
    almanacs_relevant = {row['field_id']: None for row in scores_array}
    for row in scores_array:
        if row['field_id'] not in almanacs_existing:
            almanac_dict[row['field_id']] = [ts.Almanac(row['ra'], row['dec'],
                                                        date_start, date_end), ]
            if save_new_almanacs:
                almanac_dict[row['field_id']][0].save()
            almanacs_existing.append(row['field_id'])

        # Now, make sure that the almanacs actually cover the correct date range
        # If not, replace any existing almanacs with one super Almanac for the
        # entire range requested
        try:
            almanacs_relevant[
                row['field_id']] = (a for a in almanac_dict[row['field_id']] if
                                    a.start_date <= date <= a.end_date).next()
        except KeyError:
            # This catches when no almanacs satisfy the condition in the
            # list constructor above
            almanac_dict[row['field_id']] = [
                ts.Almanac(row['ra'], row['dec'],
                           date_start, date_end), ]
            if save_new_almanacs:
                almanac_dict[row['field_id']][0].save()
            almanacs_relevant[
                    row['field_id']] = almanac_dict[row['field_id']][0]

    # Check that the dark almanac spans the relevant dates; if not,
    # regenerate it
    if dark_almanac is None or (dark_almanac.start_date > date or
                                dark_almanac.end_date < date):
        dark_almanac = ts.DarkAlmanac(date_start, end_date=date_end)
        if save_new_almanacs:
            dark_almanac.save()

    logging.debug('Finding first block of dark time for this evening')
    # Compute the times for the first block of dark time tonight
    midday = datetime.datetime.combine(date, datetime.time(12, 0, 0))
    midday_end = datetime.datetime.combine(date_end, datetime.time(12, 0, 0))
    dark_start, dark_end = dark_almanac.next_dark_period(midday,
                                                         limiting_dt=midday +
                                                         datetime.timedelta(1))

    tiles_observed = []

    while dark_start is not None:
        logging.info('Observing over dark period %5.3f to %5.3f' %
                     (dark_start, dark_end, ))
        ephem_time_now = dark_start
        local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(ephem_time_now,
                                                           ts.EPHEM_DT_STRFMT))

        # Get the next observing period for all fields being considered
        field_periods = {r['field_id']: almanacs_relevant[
            r['field_id']
        ].next_observable_period(
            local_time_now - (datetime.timedelta(
                almanacs_relevant[r['field_id']].resolution *
                60. / ts.SECONDS_PER_DAY)),
            datetime_to=ts.localize_utc_dt(ts.ephem_to_dt(dark_end))) for
                         r in scores_array}
        logging.debug('Next observing period for each field:')
        logging.debug(field_periods)
        logging.info('Next available field will rise at %5.3f' %
                     (min([v[0] for v in field_periods.itervalues() if
                           v[0] is not None]), )
                     )
        fields_available = [f for f, v in field_periods.iteritems() if
                            v[0] is not None and v[0] < dark_end]
        logging.debug('%d fields available at some point tonight' %
                      len(fields_available))

        # Rank the available fields
        logging.debug('Computing field scores')
        tiles_scores = {row['tile_pk']: (row['n_sci_rem'], row['cw_sum']) for
                        row in scores_array if
                        row['field_id'] in fields_available}
        fields_by_tile = {row['tile_pk']: row['field_id'] for
                          row in scores_array if
                          row['field_id'] in fields_available}
        hours_obs = {f: almanacs_relevant[f].hours_observable(
            local_time_now,
            datetime_to=midday_end,
            dark_almanac=dark_almanac,
            hours_better=True
        ) for f in fields_by_tile.values()}
        # Modulate scores by hours remaining
        tiles_scores = {t: v[0] * v[1] / hours_obs[fields_by_tile[t]] for
                        t, v in tiles_scores.iteritems()}
        logging.debug('Tiles scores: ')
        logging.debug(tiles_scores)

        # 'Observe' while the remaining time in this dark period is
        # longer than one pointing (slew + obs)
        logging.info('Commencing observing...')
        while ephem_time_now < (dark_end - ts.POINTING_TIME):
            logging.info('At time %5.3f, going to %5.3f' % (
                ephem_time_now, dark_end,
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
                               ts.SLEW_TIME <
                               ephem_time_now and
                               field_periods[fields_by_tile[t]][1] >
                               ephem_time_now + ts.POINTING_TIME).next()
            except StopIteration:
                # This triggers if fields will be available later tonight,
                # but none are up right now. What we do now is advance time_now
                # to the first time when any field becomes available
                ephem_time_now = min([v[0] for f, v in
                                      field_periods.iteritems()
                                      if v[0] is not None and
                                      v[1] if not None and
                                      v[0] > ephem_time_now])
                if ephem_time_now is None:
                    logging.info('There appears to be no valid observing time '
                                 'remaining out to the end_date')
                    return

                logging.info('No fields up - advancing time to %5.3f' %
                              ephem_time_now)
                local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(
                    ephem_time_now, ts.EPHEM_DT_STRFMT))
                continue

            # 'Observe' the field
            logging.info('Observing tile %d (score: %.1f), field %d at '
                         'time %5.3f, RA %3.1f, DEC %2.1f' %
                          (tile_to_obs, tiles_scores[tile_to_obs],
                           fields_by_tile[tile_to_obs], ephem_time_now,
                           [x['ra'] for x in scores_array if
                            x['tile_pk'] == tile_to_obs][0],
                           [x['dec'] for x in scores_array if
                            x['tile_pk'] == tile_to_obs][0],
                           ))
            # This is the section of code that does the 'observing'

            # Read in from DB
            # Get the list of science target IDs on this tile
            target_ids = np.asarray(rSTiexec(cursor, tile_to_obs))
            # Get the array of target_ids with target types from the database
            target_types_db = rSTyexec(cursor, target_ids=target_ids)
            # Get an array with the number of visits and repeats of these
            visits_repeats = rSVexec(cursor, target_ids=target_ids)

            # Form an array showing the type of those targets
            target_types = np.asarray(list(['' for _ in target_types_db]))
            for ttype in ['is_H0_target', 'is_vpec_target', 'is_lowz_target']:
                target_types[
                    np.asarray([_[ttype] is True for _ in target_types_db],
                               dtype=bool)
                ] = ttype
            # Calculate a success/failure rate for each target
            success_targets = test_redshift_success(target_types,
                                                    visits_repeats['visits'] +
                                                    1)  # Function needs
                                                        # incremented visits
                                                        # values

            # Set relevant targets as observed successfully, all others
            # observed but unsuccessfully
            mSRIexec(cursor, target_ids[success_targets], set_done=True)
            mSVIexec(cursor, target_ids[~success_targets])

            # Mark the tile as having been observed
            mTOexec(cursor, [tile_to_obs])

            # Set the tile score to 0 so it's not re-observed tonight
            tiles_scores[tile_to_obs] = 0.
            tiles_observed.append(tile_to_obs)

            # Increment time_now and move to observe the next field
            ephem_time_now += ts.POINTING_TIME
            local_time_now = ts.localize_utc_dt(ts.ephem_to_dt(
                ephem_time_now, ts.EPHEM_DT_STRFMT))

        # When this dark period is exhausted, figure out when the next dark
        # period is tonight (if there is one)
        logging.debug('Finding next block of dark time for tonight')
        dark_start, dark_end = dark_almanac.next_dark_period(
            local_time_now + datetime.timedelta(
                seconds=dark_almanac.resolution * 60.),
            limiting_dt=midday + datetime.timedelta(1))

    # We are now done observing for the night. It is time for some
    # housekeeping
    if len(tiles_observed) > 0:
        # Re-tile the affected fields
        # Work out which fields actually need re-tiling
        logging.info('%d tiles were observed' % len(tiles_observed))
        fields_to_retile = rCAexec(cursor, tile_list=tiles_observed)
        logging.info('This requires %d fields be re-tiled' %
                     len(fields_to_retile))
        # Re-tile those fields to a particular depth - usually 1
        # Note that the calls made by the tiling function automatically include
        # a re-computation of the target numbers in each field
        retile_fields(cursor, fields_to_retile, tiles_per_field=1)

    logging.info('Completed simulated observing for %s' %
                 date.strftime('%y-%m-%d'))


def execute(cursor, date_start, date_end, output_loc='.'):
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

    Returns
    -------
    Nil. Tiling outputs are written to the database (to simulate the action of
    the virtual observer), and plots are generated and placed in the output
    folder.
    """

    # This is just a rough scaffold to show the steps the code will need to
    # take

    # construct_league_table()
    # read_league_table()
    #
    # generate_initial_tiles()
    # write_tiles_to_db() # This creates the league table
    #
    # # DO EITHER:
    # date_curr = date_start
    # while date_curr < date_end:
    #     observe_night() # This will select & 'observe' tiles,
    #     # return the tiles observed
    #     manipulate_targets() # Update flags on successfully observe targets
    #     retile_fields () # Retile the affected fields
    #     curr_date += 1 # day
    #
    # # OR DO THIS INSTEAD:
    # observe_timeframe(date_start, date_end)
    # # This function will handle all of the above, but re-tile after each
    # # observation (and do all necessary DB interaction). This will be faster,
    # # as all the target handling can be done internally without
    # # reading/writing DB, *but* the function that do that then won't be
    # # prototyped
    #
    # read_in_observed_tiles()
    # generate_outputs()

    # TODO: Add check to skip this step if tables already exist
    # Currently dummied out with an is False
    if False:
        sim_prepare_db(cursor)

    fields = rCexec(cursor)
    # Construct the almanacs required
    logging.info('Constructing dark almanac...')
    dark_almanac = ts.DarkAlmanac(date_start, end_date=date_end,
                                  resolution=15.)
    dark_almanac.save()
    logging.info('Constructing field almanacs...')
    almanacs = {field.field_id: ts.Almanac(field.ra, field.dec, date_start,
                                           end_date=date_end, resolution=15.,
                                           minimum_airmass=2)
                for field in fields}
    # Work out which of the field almanacs already exist on disk
    files_on_disk = os.listdir()
    almanacs_existing = {k: v.generate_file_name() in files_on_disk
                         for (k, v) in almanacs.iteritems()}
    logging.info('Saving almanacs to disc...')
    for k in [k for (k, v) in almanacs_existing.iteritems() if v]:
        almanacs[k].save()

    return


if __name__ == '__main__':
    # Set the logging to write to terminal
    logging.info('Executing fullsurvey.py as file')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Get a cursor
    # TODO: Correct package imports & references
    logging.debug('Getting connection')
    conn = get_connection()
    cursor = conn.cursor()
    # Execute the simulation based on command-line arguments
    logging.debug('Doing scripts execute function')
    execute(cursor, None, None)