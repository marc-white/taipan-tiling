# Utility to do updating of science types, priorities and difficulties

from src.resources.v0_0_1.manipulate import makeScienceUpdate as mScU
from src.resources.v0_0_1.readout import readCentroidsAffected as rCA
from src.resources.v0_0_1.readout import readScience as rSc
from src.resources.v0_0_1.readout import readSciencePosn as rScP
from src.resources.v0_0_1.readout import readScienceTypes as rScTy

import taipan.simulate.logic as tsl
from taipan.simulate.simulate import test_redshift_success
from taipan.core import compute_target_difficulties

import numpy as np


def update_science_targets(cursor,
                           target_list=None, field_list=None,
                           do_tp=True, do_d=True,
                           prisci=False):
    """
    Compute updates to science types, priorities & difficulties and
    re-insert into DB.
    Moving this all into a single method prevents unnecessary duplication
    of DB reads/writes.

    Parameters
    ----------
    cursor : psycopg2.connection.cursor object
    target_list : list of ints
        List of target IDs to consider. Defaults to None, at which point all
        targets will be considered.
    field_list : list of ints
        List of field_ids to consider. Defaults to None, at which point all
        fields will be considered. If both target_list and field_list are
        supplied, field_list will be ignored.
    do_tp : Boolean, defaults to True
        Boolean denoting whether to compute types & priorities (True) or not
        (False). Defaults to True.
    do_d
        Boolean denoting whether to compute difficulties1 (True) or not
        (False). Defaults to True.

    Returns
    -------
    Nil. Database is updated in place.
    """

    # Input checking
    if target_list is not None:
        target_list = list(target_list)
        if field_list is not None:
            field_list = None
    elif field_list is not None:
        field_list = list(field_list)
        target_list = list(
            rScP.execute(cursor, field_list=field_list)['target_id'])
        field_list = None

    # Read in the input target information
    target_info_array = rScTy.execute(cursor, target_ids=target_list)

    if len(target_info_array) > 0:
        if do_tp:
            # Recompute the priorities
            priors_temp = tsl.compute_target_priorities_tree(
                target_info_array, prisci=prisci
            )
            target_info_array['priority'] = priors_temp
            temp = tsl.compute_target_types(
                target_info_array, prisci=prisci
            )
            # TODO Needs to be implemented properly
            # Anything newly-identified as vpec needs to have its
            # success re-evaluated
            # target_info_array[np.logical_and(
            #     temp['is_vpec_target'],
            #     ~target_info_array['is_vpec_target']
            # )]['success'] = test_redshift_success(
            #     target_info_array[np.logical_and(
            #         temp['is_vpec_target'],
            #         ~target_info_array['is_vpec_target']
            #     )])
            for t in ['is_h0_target', 'is_vpec_target', 'is_lowz_target']:
                target_info_array[t] = temp[t]

        if do_d:
            # Read in the science targets as TaipanTarget objects
            # Note field_list will have been set to None if target_list is not
            # None
            # targets_for_diff = rSc.execute(
            #     cursor, target_ids=target_list)
            # print targets_for_diff
            if target_list is not None:
                affected_fields = rCA.execute(
                    cursor, field_list=list(rScP.execute(cursor,
                                                         target_list=
                                                         target_list
                                                         )['field_id'])
                )
                targets_for_comp = rSc.execute(
                    cursor, field_list=list(affected_fields)
                )
                # print targets_for_comp
                targets_for_diff = [tgt for tgt in targets_for_comp if
                                    tgt.idn in target_list]
                # print len(targets_for_diff)
                # print len(targets_for_comp)
                compute_target_difficulties([tgt for tgt in targets_for_comp if
                                             tgt.idn in target_list],
                                            full_target_list=targets_for_comp)
            else:
                targets_for_diff = rSc.execute(
                    cursor)
                compute_target_difficulties(targets_for_diff)

            # Re-insert the new difficulties into the target_info_array
            # Note that we can't guarantee that the ordering of targets_for_diff
            # and target_info_array matches
            target_info_array.sort(order='target_id')
            targets_for_diff = np.array(
                [(t.idn, t.difficulty) for t in targets_for_diff],
                dtype={
                    'names': ['target_id', 'difficulty'],
                    'formats': ['i8', 'i8']
                }
            )
            targets_for_diff.sort(order='target_id')
            target_info_array['difficulty'] = targets_for_diff['difficulty']

        # Write the result back to the DB
        mScU.execute(cursor, target_info_array)



