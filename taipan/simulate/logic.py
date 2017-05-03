# These are simulator functions related to implement the Jan 2017
# survey logic, as designed by (primarily) Ned Taylor
# Some of these functions are directy transferrable to live survey ops

import numpy as np
import logging

LRG_GICOL_SELECTION_LIMIT = 1.4
NIR_JKCOL_SELECTION_LIMIT = 1.0

NORTHERN_DEC_LIMIT = 13.5
FOREGROUND_EBV_LIMIT = 1. / 3.1
GALACTIC_LATITUDE_LIMIT = 10.
JBRIGHT_MAGNITUDE_LIMIT = 14.2
ALLSKY_MAGNITUDE_LIMIT = 15.35

def _set_priority_array_values(priorities, bool_arrays, priority):
    """
    INTERNAL HELPER FUNCTION - automates the generation of statements to
    update the priorities array used in compute_target_priorities

    Parameters
    ----------
    priorities : np.array, single dimension, dtype int
        The priorities array to be updated
    bool_arrays : iterable of arrays containing booleans
        Priorities will be updated at array indices where ALL bool_arrays are
        True at said index.
    priority : int
        Priority value to set

    Returns
    -------
    Nil. priorities is updated in-situ.
    """
    # Input testing
    if ~np.all([len(b) == len(priorities) for b in bool_arrays]):
        raise ValueError('All members of bool_arrays must be of the same '
                         'length as the priorities array')

    # Form the combined boolean mask
    bool_arrays = list(bool_arrays)
    bool_mask = bool_arrays[0].copy()
    for b in bool_arrays[1:]:
        bool_mask = np.logical_and(bool_mask, b)

    # Update the priorities
    priorities[bool_mask] = priority

    return


def compute_target_types(target_info_array, prisci=False):
    """
    Compute target types (is_h0_target, is_vpec_target, is_lowz_target)
    for the given targets.

    Parameters
    ----------
    target_info_array: numpy.array object (structured)
        A numpy structured array containing the target information, one row for
        each target.
    default_priority : int
        A default priority value to use for targets which don't satisfy any
        of the given criteria. Effectively a minimum priority. Defaults to 20.
    prisci : Boolean, default False
        Whether or not we are in the 'priority science' period. Defaults to
        False.

    Returns
    -------
    tgt_types : numpy.array object (structured)
        A numpy structured array with columns 'target_id', 'is_h0_target',
        'is_lowz_target' and 'is_vpec_target'.
    """

    # Initialise the return array
    tgt_types = np.empty(len(target_info_array['target_id']),
        dtype={
            'names': ['target_id', 'is_h0_target', 'is_vpec_target',
                      'is_lowz_target'],
            'formats': ['i8', bool, bool, bool]
        }
    )
    tgt_types['target_id'] = target_info_array['target_id']
    tgt_types['is_h0_target'] = np.zeros(tgt_types['target_id'].shape
                                         ).astype(bool)
    tgt_types['is_vpec_target'] = np.zeros(tgt_types['target_id'].shape
                                           ).astype(bool)
    tgt_types['is_lowz_target'] = np.zeros(tgt_types['target_id'].shape
                                           ).astype(bool)

    # Work out which of the targets are vpec
    tgt_types['is_vpec_target'][
        np.logical_or(
            target_info_array['is_prisci_vpec_target'],
            np.logical_and(
                np.logical_or(
                    target_info_array['success'],
                    target_info_array['observations'] > 0),
                target_info_array['is_full_vpec_target']
            )
        )
    ] = True


    # Work out which of the targets are H0
    # Compute the priorities of the targets
    priorities = compute_target_priorities_tree(target_info_array,
                                                prisci=prisci)
    tgt_types['is_h0_target'][
        np.logical_and(priorities >= 60,
                       priorities <= 79)
    ] = True

    # Send back the original is_lowz_target
    tgt_types['is_lowz_target'] = target_info_array['is_lowz_target']

    # Work out which of the targets are lowz
    # Currently, none (doesn't really affect anything under current logic)

    return tgt_types


def compute_target_priorities_tree(target_info_array, default_priority=0,
                                   prisci=False):
        # Initialize the priorities array with the default value
    logging.info('Computing target priorities with prisci: %s' % str(prisci))
    priorities = np.zeros(target_info_array['target_id'].shape).astype('i')
    priorities += int(default_priority)

    is_in_survey = in_survey_footprint(target_info_array)

    # __________________________________________________________________________
    #
    #                                        all sky BAO target priorities block
    # __________________________________________________________________________

    # is_Jselected = np.logical_and(is_in_survey, target_info_array['is_nir'])
    # is_Jselected = is_in_survey & ( target_info_array[ 'is_nir' ] == True )
    Jmag = target_info_array['mag_j']
    is_Jselected = np.logical_and(is_in_survey,
                                  Jmag < ALLSKY_MAGNITUDE_LIMIT)
    JminusK = target_info_array['col_jk']  # [ 'jkcol' ]
    no_good_redshift = (target_info_array['success'] == False)

    # print np.sort( JminusK[ is_Jselected ] )

    num_visits = target_info_array['observations']

    # ___________________ BAO targets not yet observed, prioritised by J-K color

    is_unobs_Jsel = np.logical_and(is_Jselected,
                                   num_visits == 0)
    is_unobs_Jsel = np.logical_and(is_unobs_Jsel,
                                   no_good_redshift)
    # is_unobs_Jsel = is_Jselected & ( num_visits == 0 )

    priorities[np.logical_and(is_unobs_Jsel,
                              JminusK > 1.4)] = 79
    priorities[np.logical_and(is_unobs_Jsel,
                              np.logical_and(1.3 <= JminusK,
                                             JminusK < 1.4))] = 78
    priorities[np.logical_and(is_unobs_Jsel,
                              np.logical_and(1.2 <= JminusK,
                                             JminusK < 1.3))] = 77
    # Original code
    # priorities[ is_unobs_Jsel &                      ( JminusK > 1.4 ) ] = 79
    # priorities[ is_unobs_Jsel & ( 1.3 <= JminusK ) & ( JminusK < 1.4 ) ] = 78
    # priorities[ is_unobs_Jsel & ( 1.2 <= JminusK ) & ( JminusK < 1.3 ) ] = 77

    # Original code
    priorities[np.logical_and(is_unobs_Jsel,
                              np.logical_and(1.2 > JminusK,
                                             target_info_array['mag_j']
                                             <= JBRIGHT_MAGNITUDE_LIMIT)
                              )] = 76
    priorities[np.logical_and(is_unobs_Jsel,
                              np.logical_and(1.2 > JminusK,
                                             target_info_array['mag_j']
                                             > JBRIGHT_MAGNITUDE_LIMIT)
                              )] = 75

    # -- DEPRECATED --
    # priorities[np.logical_and(is_unobs_Jsel,
    #                           1.2 <= JminusK)] = 75
    # Original code
    # priorities[ is_unobs_Jsel & ( 1.2 <= JminusK )                     ] = 75


    # ___________________________ BAO targets to be reobserved at lower priority

    no_good_redshift = (target_info_array['success'] == False)

    is_noz_Jsel = np.logical_and(is_Jselected,
                                 no_good_redshift)
    is_noz_Jsel = np.logical_and(is_noz_Jsel, num_visits > 1)
    # Original code
    # is_noz_Jsel = ( is_Jselected & no_good_redshift & ( num_visits > 1 ) )

    is_noz_Jsel_red = np.logical_and(is_noz_Jsel, JminusK > 1.2)
    # Original code
    # is_noz_Jsel_red  = is_noz_Jsel & ( JminusK > 1.2 )

    priorities[np.logical_and(is_noz_Jsel_red, num_visits == 1)] = 69
    priorities[np.logical_and(is_noz_Jsel_red, num_visits == 2)] = 68
    priorities[np.logical_and(is_noz_Jsel_red, num_visits == 3)] = 67
    # Original code
    priorities[is_noz_Jsel_red & (num_visits == 1)] = 69
    priorities[is_noz_Jsel_red & (num_visits == 2)] = 68
    priorities[is_noz_Jsel_red & (num_visits == 3)] = 67

    is_noz_Jsel_blue = np.logical_and(is_noz_Jsel, JminusK <= 1.2)
    # Original code
    # is_noz_Jsel_blue = is_noz_Jsel & ( JminusK <= 1.2 )

    priorities[np.logical_and(is_noz_Jsel_blue, num_visits == 1)] = 66
    priorities[np.logical_and(is_noz_Jsel_blue, num_visits == 2)] = 65
    # Original code
    # priorities[ is_noz_Jsel_blue & ( num_visits == 1 ) ] = 66
    # priorities[ is_noz_Jsel_blue & ( num_visits == 2 ) ] = 65

    is_lrg_selected = np.logical_and(is_in_survey, is_Jselected == False)
    is_lrg_selected = np.logical_and(is_lrg_selected,
                                     target_info_array['is_lrg'])
    # Original code
    # is_lrg_selected = ( is_in_survey & ( is_Jselected == False )
    #                     & target_info_array[ 'is_lrg' ] )

    is_noz_lrg = np.logical_and(is_lrg_selected,
                                no_good_redshift)
    # Original code
    # is_noz_lrg = is_lrg_selected & no_good_redshift

    if prisci:  # _________________ in prisci period, only condsider KiDS fields

        is_in_census_region = in_census_region(target_info_array)

    else:  # ____________________ and whole sky once full survey is underway

        is_in_census_region = is_in_survey

    if not prisci:  # _________ add LRG selection after priority science period

        priorities[np.logical_and(is_noz_lrg, num_visits == 0)] = 74
        priorities[np.logical_and(is_noz_lrg, num_visits == 1)] = 73
        priorities[np.logical_and(is_noz_lrg, num_visits == 2)] = 72
        priorities[np.logical_and(is_noz_lrg, num_visits == 3)] = 71
        priorities[np.logical_and(is_noz_lrg, num_visits == 4)] = 70
        # Original code
        # priorities[ is_noz_lrg & ( num_visits == 0 ) ] = 74
        # priorities[ is_noz_lrg & ( num_visits == 1 ) ] = 73
        # priorities[ is_noz_lrg & ( num_visits == 2 ) ] = 72
        # priorities[ is_noz_lrg & ( num_visits == 3 ) ] = 71
        # priorities[ is_noz_lrg & ( num_visits == 4 ) ] = 70

    else:

        is_noz_lrg = np.logical_and(is_noz_lrg, is_in_census_region)

        priorities[np.logical_and(is_noz_lrg, num_visits == 0)] = 44
        priorities[np.logical_and(is_noz_lrg, num_visits == 1)] = 43
        priorities[np.logical_and(is_noz_lrg, num_visits == 2)] = 42
        priorities[np.logical_and(is_noz_lrg, num_visits == 3)] = 41
        priorities[np.logical_and(is_noz_lrg, num_visits == 4)] = 40

    # __________________________________________________________________________
    #
    #                                                 galaxy census target block
    # __________________________________________________________________________

    # _______________________________ NOTE: these priorities override BAO values

    is_iband_selected = target_info_array['is_iband']

    is_live_census_target = np.logical_and(is_in_census_region,
                                           is_iband_selected)
    is_live_census_target = np.logical_and(is_live_census_target,
                                           no_good_redshift)
    # Original code
    # is_live_census_target = (
    #     is_in_census_region & is_iband_selected & no_good_redshift )

    # _______________________ census targets prioritised according to num_visits

    num_visits = target_info_array['observations']

    priorities[np.logical_and(is_live_census_target, num_visits == 0)] = 99
    priorities[np.logical_and(is_live_census_target, num_visits == 1)] = 98
    priorities[np.logical_and(is_live_census_target, num_visits == 2)] = 97
    priorities[np.logical_and(is_live_census_target, num_visits == 3)] = 96
    priorities[np.logical_and(is_live_census_target, num_visits == 4)] = 95
    # Original code
    # priorities[ is_live_census_target & ( num_visits == 0 ) ] = 99
    # priorities[ is_live_census_target & ( num_visits == 1 ) ] = 98
    # priorities[ is_live_census_target & ( num_visits == 2 ) ] = 97
    # priorities[ is_live_census_target & ( num_visits == 3 ) ] = 96
    # priorities[ is_live_census_target & ( num_visits == 4 ) ] = 95

    retired = np.logical_and(is_live_census_target, num_visits >= 5)
    # Original code
    # retired = is_live_census_target & ( num_visits >= 5 )
    priorities[retired] = 10

    # __________________________________________________________________________

    #                                                  all sky vpec target block
    # __________________________________________________________________________

    # __________________________ NOTE: vpec priorities override all other values

    is_6df_preselected = target_info_array['is_prisci_vpec_target']
    is_found_in_taipan = np.logical_and(
        target_info_array['is_full_vpec_target'],
        target_info_array['observations'] > 0
    )
    # Original code
    #     is_found_in_taipan = ( target_info_array[ 'is_full_vpec_target' ]
    # #                           & ( target_info_array[ 'observations' ] > 0 ) )
    #                            & ( target_info_array[ 'visits' ] > 0 ) )
    is_live_vpec_target = np.logical_or(is_6df_preselected,
                                        is_found_in_taipan)
    is_live_vpec_target = np.logical_and(is_live_vpec_target,
                                         target_info_array['success'] == False)
    is_live_vpec_target = np.logical_and(is_live_vpec_target,
                                         is_in_survey)
    # Original code
    # is_live_vpec_target = ( ( is_6df_preselected | is_found_in_taipan )
    #                         & ( target_info_array[ 'success' ] == False ) )

    # ________________________________ targets prioritised according to redshift

    zspec = target_info_array['zspec']

    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.00 <= zspec, zspec < 0.01))] = 89
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.01 <= zspec, zspec < 0.02))] = 88
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.02 <= zspec, zspec < 0.03))] = 87
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.03 <= zspec, zspec < 0.04))] = 86
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.04 <= zspec, zspec < 0.05))] = 85
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.05 <= zspec, zspec < 0.06))] = 84
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.06 <= zspec, zspec < 0.07))] = 83
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.07 <= zspec, zspec < 0.08))] = 82
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.08 <= zspec, zspec < 0.09))] = 81
    priorities[np.logical_and(is_live_vpec_target,
                              np.logical_and(0.09 <= zspec, zspec < 0.10))] = 80
    # Original code
    # priorities[ is_live_vpec_target & (0.00 <= zspec ) & ( zspec < 0.01 ) ] = 89
    # priorities[ is_live_vpec_target & (0.01 <= zspec ) & ( zspec < 0.02 ) ] = 88
    # priorities[ is_live_vpec_target & (0.02 <= zspec ) & ( zspec < 0.03 ) ] = 87
    # priorities[ is_live_vpec_target & (0.03 <= zspec ) & ( zspec < 0.04 ) ] = 86
    # priorities[ is_live_vpec_target & (0.04 <= zspec ) & ( zspec < 0.05 ) ] = 85
    # priorities[ is_live_vpec_target & (0.05 <= zspec ) & ( zspec < 0.06 ) ] = 84
    # priorities[ is_live_vpec_target & (0.06 <= zspec ) & ( zspec < 0.07 ) ] = 83
    # priorities[ is_live_vpec_target & (0.07 <= zspec ) & ( zspec < 0.08 ) ] = 82
    # priorities[ is_live_vpec_target & (0.08 <= zspec ) & ( zspec < 0.09 ) ] = 81
    # priorities[ is_live_vpec_target & (0.09 <= zspec ) & ( zspec < 0.10 ) ] = 80

    retired = np.logical_and(is_live_vpec_target, num_visits >= 5)
    # Original code
    retired = is_live_vpec_target & (num_visits >= 5)
    priorities[retired] = 10

    return priorities


def compute_target_priorities_tree_deprecated(
        target_info_array, default_priority=0,
        prisci=False):
    """
    Compute priority values for a list of targets.

    This function is different from the _percase calculation in that it's
    closer to Ned Taylor's priority decision tree, which will (hopefully)
    compute faster.

    The input to this function should be the output of a call to the TaipanDB
    function readScienceTypes.execute(cursor, *args, **kwargs). However, this is
    not being hard-coded here, to keep database and simulator operations
    separated.

    Parameters
    ----------
    target_info_array: numpy.array object (structured)
        A numpy structured array containing the target information, one row for
        each target.
    default_priority : int
        A default priority value to use for targets which don't satisfy any
        of the given criteria. Effectively a minimum priority. Defaults to 0.
    prisci : Boolean, default False
        Whether or not we are in the 'priority science' period. Defaults to
        False.

    Returns
    -------
    priorities: list of ints
        A list of priorities, corresponding to the target_info_array provided.
    """

    default_priority = int(default_priority)

    # Initialize the priorities array with the default value
    priorities = np.zeros(target_info_array['target_id'].shape).astype('i')
    priorities += default_priority

    if prisci:
        # Priority science-period logic
        in_census_region = np.logical_or(
            # KiDS-N
            np.logical_or(
                np.logical_and(
                    np.logical_and(156. < target_info_array['ra'],
                                   target_info_array['ra'] <= 225.),
                    np.logical_and(-5. < target_info_array['dec'],
                                   target_info_array['dec'] < 4.),
                ),
                np.logical_and(
                    np.logical_and(225. < target_info_array['ra'],
                                   target_info_array['ra'] < 238.),
                    np.logical_and(-3. < target_info_array['dec'],
                                   target_info_array['dec'] < 4.)
                )
            ),
            # KiDS-S
            np.logical_and(np.logical_or(target_info_array['ra'] > 329.5,
                                         target_info_array['ra'] < 53.5),
                           np.logical_and(-35.6 < target_info_array['dec'],
                                          target_info_array['dec'] < -25.7)
        )
        )
        # IN CENSUS REGION
        # i-band selected
        in_census_region_iband = np.logical_and(in_census_region,
                                                target_info_array['is_iband'])
        # No success yet
        # This means we should use visits, not observations
        priorities[np.logical_and(in_census_region_iband,
                                  np.logical_and(~target_info_array['success'],
                                                 target_info_array['visits'] <
                                                 5))
        ] = 99 - target_info_array[np.logical_and(in_census_region_iband,
                                  np.logical_and(~target_info_array['success'],
                                                 target_info_array['visits'] <
                                                 5))
        ]['visits']

        priorities[np.logical_and(in_census_region_iband,
                                  np.logical_and(~target_info_array['success'],
                                                 target_info_array['visits'] >=
                                                 5))
        ] = 50

        # 'done' vpec target in census region
        priorities[np.logical_and(
            in_census_region_iband,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT was 80s assigned to all vpecs regardless of
            # status
            target_info_array['is_vpec_target']
        )] = 89 - np.clip(100. * target_info_array[np.logical_and(
            in_census_region_iband,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT was 80s assigned to all vpecs regardless of
            # status
            target_info_array['is_vpec_target']
        )]['zspec'], 0, 9).astype('i')

        # 'done' lowz targets in census region
        priorities[np.logical_and(
            in_census_region_iband,
            np.logical_and(target_info_array['is_lowz_target'],
                           target_info_array['success'])
        )] = 0

        # Other 'done' i-band targets in census region
        priorities[np.logical_and(
            np.logical_and(
                in_census_region_iband,
                target_info_array['success']
            ),
            np.logical_and(
                ~target_info_array['is_vpec_target'],
                ~target_info_array['is_lowz_target']
            )
        )] = 0

        # 'done' vpec target in census region
        priorities[np.logical_and(
            in_census_region_iband,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT was 80s assigned to all vpecs regardless of
            # status
            target_info_array['is_vpec_target']
        )] = 89 - np.clip(100. * target_info_array[np.logical_and(
            in_census_region_iband,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT was 80s assigned to all vpecs regardless of
            # status
            target_info_array['is_vpec_target']
        )]['zspec'], 0, 9).astype('i')

        # Remaining non-iband targets in region are fillers
        priorities[np.logical_and(
            in_census_region,
            ~in_census_region_iband
        )] = 0

        # OUTSIDE CENSUS REGION
        # NIR-selected
        out_census_region_nir = np.logical_and(~in_census_region,
                                               target_info_array['is_nir'])
        # Standard incomplete target defns.
        # Because these are incomplete, we should use visits, not observations
        # nir_priority_defs
        priorities[np.logical_and(
            np.logical_and(
                out_census_region_nir,
                ~target_info_array['success']
            ),
            target_info_array['visits'] < 1
        )] = 79 - np.clip(5 - (target_info_array[np.logical_and(
            np.logical_and(
                out_census_region_nir,
                ~target_info_array['success']
            ),
            target_info_array['visits'] < 1
        )]['col_jk'] - 1.0) * 10, 0, 4).astype('i')

        priorities[np.logical_and(
            np.logical_and(
                out_census_region_nir,
                ~target_info_array['success']
            ),
            target_info_array['visits'] >= 1
        )] = 0

        priorities[np.logical_and(
            np.logical_and(target_info_array['col_jk'] < 1.2,
                           np.logical_and(out_census_region_nir,
                                          ~target_info_array['success'])),
            np.logical_and(
                target_info_array['visits'] >= 1,
                target_info_array['visits'] < 3
            )
        )] = 67 - target_info_array[np.logical_and(
            np.logical_and(target_info_array['col_jk'] < 1.2,
                           np.logical_and(out_census_region_nir,
                                          ~target_info_array['success'])),
            np.logical_and(
                target_info_array['visits'] >= 1,
                target_info_array['visits'] < 3
            )
        )]['visits']

        priorities[np.logical_and(
            np.logical_and(
                out_census_region_nir,
                ~target_info_array['success']
            ),
            np.logical_and(
                target_info_array['visits'] >= 1,
                np.logical_and(
                    target_info_array['visits'] < 4,
                    target_info_array['col_jk'] > 1.2
                )
            )
        )] = 70 - target_info_array[np.logical_and(
            np.logical_and(
                out_census_region_nir,
                ~target_info_array['success']
            ),
            np.logical_and(
                target_info_array['visits'] >= 1,
                np.logical_and(
                    target_info_array['visits'] < 4,
                    target_info_array['col_jk'] > 1.2
                )
            )
        )]['visits']

        # 'done' vpec targets in the area
        priorities[np.logical_and(
            out_census_region_nir,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT wants this done regardless of success status
            target_info_array['is_vpec_target'],
        )] = 89 - np.clip(10. * target_info_array[np.logical_and(
            out_census_region_nir,
            # np.logical_and(target_info_array['is_vpec_target'],
            #                target_info_array['success'])
            # CHANGE 170322 - ENT wants this done regardless of success status
            target_info_array['is_vpec_target'],
        )]['zspec'], 0, 9).astype('i')

        # Other targets in region
        priorities[np.logical_and(
            out_census_region_nir,
            np.logical_and(~target_info_array['is_vpec_target'],
                           target_info_array['success'])
        )] = 0

        # Targets not NIR selected outside census region
        priorities[np.logical_and(
            ~in_census_region,
            ~out_census_region_nir
        )] = 0

    else:
        # Full survey period logic
        # i-band selected, no redshift yet
        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_and(~target_info_array['success'],
                           target_info_array['visits'] < 5)
        )] = 99 - target_info_array[np.logical_and(
            target_info_array['is_iband'],
            np.logical_and(~target_info_array['success'],
                           target_info_array['visits'] < 5)
        )]['visits']

        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_and(~target_info_array['success'],
                           target_info_array['visits'] >= 5)
        )] = 50

        # iband selected, redshift, vpec target
        # CHANGE 170322 - Ensure prisci vpec targets remain in the 80s as well
        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_or(
                np.logical_and(
                    target_info_array['success'],
                    target_info_array['is_vpec_target']
                ),
                target_info_array['is_prisci_vpec_target']
            )
        )] = 89 - np.clip(target_info_array[np.logical_and(
            target_info_array['is_iband'],
            np.logical_or(
                np.logical_and(
                    target_info_array['success'],
                    target_info_array['is_vpec_target']
                ),
                target_info_array['is_prisci_vpec_target']
            )
        )]['zspec'], 0, 9).astype('i')

        # iband selected, redshift, lowz target
        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_and(
                target_info_array['success'],
                target_info_array['is_lowz_target']
            )
        )] = 0

        # Remaining i-band selected targets
        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_and(
                target_info_array['success'],
                np.logical_and(~target_info_array['is_lowz_target'],
                               ~target_info_array['is_vpec_target'])
            )
        )] = 0

        # LRG selected, no redshift
        # lrg_priority_defs
        priorities[np.logical_and(
            target_info_array['is_lrg'],
            np.logical_and(~target_info_array['success'],
                           target_info_array['visits'] < 1)
        )] = 74 - np.clip(5 - (target_info_array[np.logical_and(
            target_info_array['is_lrg'],
            np.logical_and(~target_info_array['success'],
                           target_info_array['visits'] < 1)
        )]['col_gi'] - 1.4) * 10, 0, 4).astype('i')

        priorities[np.logical_and(
            np.logical_and(target_info_array['is_lrg'],
                           target_info_array['col_gi'] >
                           LRG_GICOL_SELECTION_LIMIT),
            np.logical_and(~target_info_array['success'],
                           np.logical_and(0 < target_info_array['visits'],
                                          target_info_array['visits'] < 6))
        )] = 65 - target_info_array[np.logical_and(
            np.logical_and(target_info_array['is_lrg'],
                           target_info_array['col_gi'] >
                           LRG_GICOL_SELECTION_LIMIT),
            np.logical_and(~target_info_array['success'],
                           np.logical_and(0 < target_info_array['visits'],
                                          target_info_array['visits'] < 6))
        )]['visits']

        priorities[np.logical_and(
            target_info_array['is_lrg'],
            np.logical_and(~target_info_array['success'],
                           np.logical_or(target_info_array['visits'] > 6,
                                         target_info_array['col_gi'] <=
                                         LRG_GICOL_SELECTION_LIMIT)
                           )
        )] = 0

        # LRG selected, redshift
        priorities[np.logical_and(
            target_info_array['is_lrg'],
            target_info_array['success']
        )] = 0

        # No particular selection, filler
        priorities[np.logical_and(
            ~target_info_array['is_iband'],
            ~target_info_array['is_lrg']
        )] = 0

        # iband selected, redshift, vpec target
        # CHANGE 170322 - Ensure prisci vpec targets remain in the 80s as well
        priorities[np.logical_and(
            target_info_array['is_iband'],
            np.logical_or(
                np.logical_and(
                    target_info_array['success'],
                    target_info_array['is_vpec_target']
                ),
                target_info_array['is_prisci_vpec_target']
            )
        )] = 89 - np.clip(100.* target_info_array[np.logical_and(
            target_info_array['is_iband'],
            np.logical_or(
                np.logical_and(
                    target_info_array['success'],
                    target_info_array['is_vpec_target']
                ),
                target_info_array['is_prisci_vpec_target']
            )
        )]['zspec'], 0, 9).astype('i')

    # For all epochs, if target is outside survey footprint, set to 0
    priorities[~np.logical_and(
        np.logical_and(
            np.abs(target_info_array['glat']) > GALACTIC_LATITUDE_LIMIT,
            target_info_array['dec'] < NORTHERN_DEC_LIMIT
        ),
        target_info_array['ebv'] < FOREGROUND_EBV_LIMIT
    )] = 0

    return priorities


def compute_target_priorities_percase(target_info_array, default_priority=20, ):
    """
    Compute priority values for a list of targets.

    The input to this function should be the output of a call to the TaipanDB
    function readScience.execute(cursor, *args, **kwargs). However, this is
    not being hard-coded here, to keep database and simulator operations
    separated.

    Parameters
    ----------
    target_info_array: numpy.array object (structured)
        A numpy structured array containing the target information, one row for
        each target.
    default_priority : int
        A default priority value to use for targets which don't satisfy any
        of the given criteria. Effectively a minimum priority. Defaults to 20.

    Returns
    -------
    priorities: list of ints
        A list of priorities, corresponding to the target_info_array provided.
    """

    raise UserWarning('You are using compute_target_priorities_percase - this '
                      'is not up-to-date!')

    default_priority = int(default_priority)

    # Initialize the priorities array with the default value
    priorities = np.zeros(target_info_array['target_id'].shape)
    priorities += default_priority

    # We now move through the priority bands from lowest to highest,
    # updating the priorities where applicable

    # 50-59: i-selected & low-z targets needing more spectra
    # NOT FULLY IMPLEMENTED FOR SIMULATOR
    # This line will set any vpec targets without zspec (i.e. ones
    # identified during the survey) to have an intermediate priority
    # Those targets with a known zpec will get set further down the
    # priority cascade
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                               ],
                               55)

    # 60-69: BAO-centric targets w/ at least one observation
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['observations'] == 5,
                               ], 60)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['observations'] == 4,
                               ], 61)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['observations'] == 3,
                               ], 62)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['observations'] == 2,
                               ], 63)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['observations'] == 1,
                               ], 64)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   # target_info_array['col_jk'] >= 1.,
                                   target_info_array['col_jk'] < 1.2,
                                   target_info_array['observations'] == 2,
                               ], 65)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   # target_info_array['col_jk'] >= 1.,
                                   target_info_array['col_jk'] < 1.2,
                                   target_info_array['observations'] == 1,
                               ], 66)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.2,
                                   target_info_array['observations'] == 3,
                               ], 67)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.2,
                                   target_info_array['observations'] == 2,
                               ], 68)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.2,
                                   target_info_array['observations'] == 1,
                               ], 69)

    # 70-79: BAO-centric targets without observations
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.4,
                                   target_info_array['col_gi'] < 1.5,
                               ], 70)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.5,
                                   target_info_array['col_gi'] < 1.6,
                               ], 71)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.6,
                                   target_info_array['col_gi'] < 1.7,
                               ], 72)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.7,
                                   target_info_array['col_gi'] < 1.8,
                               ], 73)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_lrg'],
                                   target_info_array['col_gi'] >= 1.8,
                               ], 74)
    # _set_priority_array_values(priorities,
    #                            [
    #                                target_info_array['is_nir'],
    #                                target_info_array['col_jk'] >= 1.0,
    #                                target_info_array['col_jk'] < 1.1,
    #                            ], 75)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   # target_info_array['col_jk'] >= 1.1,
                                   target_info_array['col_jk'] < 1.2,
                               ], 76)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.2,
                                   target_info_array['col_jk'] < 1.3,
                               ], 77)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.3,
                                   target_info_array['col_jk'] < 1.4,
                               ], 78)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_nir'],
                                   target_info_array['col_jk'] >= 1.4,
                               ], 79)

    # 80-89: prior-identified vpec targets
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.09,
                                   target_info_array['zspec'] < 0.1,
                               ], 80)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.08,
                                   target_info_array['zspec'] < 0.09,
                               ], 81)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.08,
                                   target_info_array['zspec'] < 0.07,
                               ], 82)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.07,
                                   target_info_array['zspec'] < 0.06,
                               ], 83)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.06,
                                   target_info_array['zspec'] < 0.05,
                               ], 84)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.05,
                                   target_info_array['zspec'] < 0.04,
                               ], 85)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.04,
                                   target_info_array['zspec'] < 0.03,
                               ], 86)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.03,
                                   target_info_array['zspec'] < 0.02,
                               ], 87)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.02,
                                   target_info_array['zspec'] < 0.01,
                               ], 88)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_vpec_target'],
                                   target_info_array['zspec'] >= 0.01,
                                   target_info_array['zspec'] < 0.,
                               ], 89)

    # i-band selection, where completeness is a priority
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_iband'],
                                   target_info_array['observations'] == 4,
                               ], 95)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_iband'],
                                   target_info_array['observations'] == 3,
                               ], 96)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_iband'],
                                   target_info_array['observations'] == 2,
                               ], 97)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_iband'],
                                   target_info_array['observations'] == 1,
                               ], 98)
    _set_priority_array_values(priorities,
                               [
                                   target_info_array['is_iband'],
                                   target_info_array['observations'] == 0,
                               ], 99)

    return priorities


# -----
# HELPER FUNCTIONS

def in_survey_footprint( target ):
    glat = target[ 'glat' ]
    ebv = target[ 'ebv' ]
    dec = target[ 'dec' ]

    result = np.logical_and(np.abs( glat ) > GALACTIC_LATITUDE_LIMIT,
                            ebv < FOREGROUND_EBV_LIMIT)
    result = np.logical_and(result,
                            dec < NORTHERN_DEC_LIMIT)
    return result
    # Original code
    # return ( ( np.abs( glat ) > GALACTIC_LATITUDE_LIMIT )
    #          & ( ebv < FOREGROUND_EBV_LIMIT )
    #          & ( dec < NORTHERN_DEC_LIMIT )
    #     )

def in_kids_north( target ):
    ra, dec = target['ra'], target['dec']
    result1 = np.logical_and(156. < ra, ra <= 225.)
    result1 = np.logical_and(result1, -5. < dec)
    result1 = np.logical_and(result1, dec < +4.)

    result2 = np.logical_and(225. < ra , ra < 238.)
    result2 = np.logical_and(result2, -3. < dec)
    result2 = np.logical_and(result2, dec < +4.)

    return np.logical_or(result1, result2)
    # Original code
    # return ( ( (156. < ra) & (ra <= 225.) & (-5. < dec) & (dec < +4.) )
    #          | ( (225. < ra ) & (ra < 238.) & (-3. < dec ) & (dec < +4.) ) )

def in_kids_south( target, ra=None, dec=None ):
    ra, dec = target['ra'], target['dec']
    result = np.logical_or(329.5 < ra,
                           ra < 53.5)
    result = np.logical_and(result,
                            -35.6 < dec)
    result = np.logical_and(result,
                            dec < -25.7)
    return result
    # Original code
    # return ( ((329.5 < ra) | (ra < 53.5)) & (-35.6 < dec) & (dec < -25.7) )

def in_census_region( target ):
    return np.logical_or(in_kids_north(target),
                         in_kids_south(target))
    # Original code
    # return ( in_kids_north( target ) | in_kids_south( target ) )
