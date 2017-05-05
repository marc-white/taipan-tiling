# Utility functions for the simulator to ease tiling operations

import datetime
import logging

import taipan.tiling as tl

from src.resources.v0_0_1.readout.readScience import execute as rScexec
from src.resources.v0_0_1.readout.readGuides import execute as rGexec
from src.resources.v0_0_1.readout.readStandards import execute as rSexec
from src.resources.v0_0_1.readout.readCentroids import execute as rCexec
from src.resources.v0_0_1.readout.readCentroidsAffected import execute as \
    rCAexec

from src.resources.v0_0_1.insert.insertTiles import execute as iTexec

from src.resources.v0_0_1.delete.deleteTiles import execute as dTexec

from taipan.simulate.utils.updatesci import update_science_targets
from taipan.core import compute_target_difficulties

def retile_fields(cursor, field_list, tiles_per_field=1,
                  tiling_time=datetime.datetime.now(),
                  disqualify_below_min=True, restrict_targets=True,
                  delete_queued=False, bins=1,
                  do_priorities=True):
    """
    Re-tile the fields passed.

    Parameters
    ----------
    cursor:
        A psycopg2 cursor for communicating with the database
    field_list:
        A list of fields to be re-tiled. Should be a list of field IDs.
    tiles_per_field:
        Optional int, denoting how many tiles to generate per field. Defaults
        to 1.
    tiling_time:
        Optional; time to record as the tiling (i.e. configuration) time for
        the new tiles. Defaults to datetime.datetime.now().
    disqualify_below_min:
        Optional Boolean, passed on through to
        TaipanTile.calculate_tile_score(). Sets tile scores to 0 if tiles do
        not meet minimum numbers of guides and/or standards assigned. Defaults
        to True.
    restrict_targets:
        Optional Boolean, denoting whether to restrict the read-in
        science_targets on the database side (rather than only during tiling).
        This provides factor 5-10 speed increases when only re-tiling small
        regions of sky. Defaults to True.
    delete_queued:
        Optional Boolean, denoting whether to delete any tiles marked as
        'queued', but not 'observed'. Defaults to False (i.e. such tiles will
        not be deleted).
    bins:
        Optional integer, denoting how many bins to split re-tiling into.
        Tiling appears to be optimally efficient for 10-20 tiles per batch.
        Defaults to 1 (i.e. tiling will be done in one batch).

    Returns
    -------
    Nil. The following will occur:
    - New tiles will be generated in local memory;
    - Redundant tiles will be eliminated from the database;
    - New tiles will be pushed into the database.
    """
    bins = int(bins)

    if len(field_list) == 0:
        logging.debug('No fields passed to utils.tiling - no tiling done')
        return
    # else:
    #     logging.info('retile_fields has received the following list of fields '
    #                  'to retile: %s' % (', '.join(str(f) for f in field_list)))

    logging.debug('Retiling fields w/ recorded datetime %s' % (
        tiling_time.strftime('%y-%m-%d %H:%M:%S'),
    ))

    # Eliminate the redundant tiles from the DB
    if delete_queued:
        dTexec(cursor, field_list=field_list, obs_status=False)
    else:
        dTexec(cursor, field_list=field_list, obs_status=False,
               queue_status=False)

    for k in range(bins):

        sub_field_list = field_list[k * len(field_list) / bins:
                                    min((k + 1) * len(field_list) / bins,
                                        len(field_list))]
        # logging.info('This list has been trimmed to: %s' %
        #              (', '.join(str(f) for f in sub_field_list)))

        if restrict_targets:
            fields_w_targets = rCAexec(cursor, field_list=sub_field_list)
        else:
            fields_w_targets = None

        # Get the required targets from the database
        candidate_targets = rScexec(cursor, unobserved=False,
                                    # unassigned=True,
                                    unqueued=True,
                                    # Read in rCA fields for diff. calc.
                                    field_list=fields_w_targets)
        # New 170505 - recompute target diff. here, instead of in DB
        compute_target_difficulties(candidate_targets)

        guide_targets = rGexec(cursor, field_list=fields_w_targets)
        standard_targets = rSexec(cursor, field_list=fields_w_targets)
        fields_to_tile = rCexec(cursor, field_ids=sub_field_list)
        # logging.info('retile_fields is tiling the following fields together:'
        #              ' %s' %
        #              (', '.join(str(t.field_id) for t in fields_to_tile)))

        # Execute a re-tile of the affected fields to the required depth
        tile_list, targets_after_tile = \
            tl.generate_tiling_greedy_npasses(candidate_targets,
                                              standard_targets,
                                              guide_targets, tiles_per_field,
                                              tiles=fields_to_tile,
                                              sequential_ordering=(2, 1),
                                              recompute_difficulty=False,
                                              repeat_targets=True,
                                              # Already checks tile radius
                                              )

        # Write the new tiles back to the database
        iTexec(cursor, tile_list, config_time=tiling_time,
               disqualify_below_min=disqualify_below_min)

    return
