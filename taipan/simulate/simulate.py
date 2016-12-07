import numpy as np
import logging


def test_redshift_success(target_types_db, num_visits,
                          prob_vpec_first=0.3,
                          prob_vpec_second=0.7,
                          prob_lowz_each=0.8):
    """
    FOR TEST USE ONLY

    Simulates the effects of needing repeat visits of certain targets
    by randomly designating a target as being 'satisfied' or not. The
    probability of this occurring is a function of the priority and number
    of visits.

    Written by Ned Taylor.

    Parameters
    ----------
    target_types_db:
        A list of target_types. These correspond to the database column names.
    num_visits:
        A corresponding list of the number of times each target has been
        observed already this repeat. Note that these values must have been
        updated for the current observation pass *before* test_redshift_success
        is invoked. We *strongly* recommend that this be done in memory for this
        function, and then the results of this function inform what to write
        permanently back to the database.
    prob_vpec_first:
        The probability that a peculiar velocity target will be successfully
        observed on the first pass. Float, 0.0 to 1.0 inclusive.
    prob_vpec_second:
        The probability that a peculiar velocity target will be successfully
        observed after two passes. Note that this is a *cumulative* probability,
        such that P(success in 1 || success in 2) = prob_vpec_second. Float,
        0.0 to 1.0 inclusive.
    prob_lowz_each:
        The probability that a low-redshift target will be successfully observed
        on any pass. Float, 0.0 to 1.0 inclusive.

    Returns
    -------
    is_satisfied:
        A list of Boolean values denoting whether each target should be
        considered 'satisfied' (i.e. we have enough observations of it).

    """

    # Input checking
    # Make sure we have lists, not single values
    try:
        _ = target_types_db[0]
    except TypeError:
        target_types_db = [target_types_db, ]
    try:
        _ = num_visits[0]
    except TypeError:
        num_visits = [num_visits, ]

    # Check that the length of the two lists match
    if len(target_types_db) != len(num_visits):
        raise ValueError('Lists of target priorities and num_visits must '
                         'be of equal length')
    # Make sure num_visits has been incremented before calling this function
    # Note that this will only catch non-visited targets which have not been
    # incremented - otherwise, we can't be sure if it's been done or not
    if np.any(num_visits <= 0):
        raise RuntimeError('must increment num_visits before calling '
                           'test_success!')

    if prob_vpec_first < 0 or prob_vpec_first > 1:
        raise ValueError('prob_vpec_first must be in the range [0,1]')
    if prob_vpec_second < 0 or prob_vpec_second > 1:
        raise ValueError('prob_vpec_second must be in the range [0,1]')
    if prob_lowz_each < 0 or prob_lowz_each > 1:
        raise ValueError('prob_lowz_each must be in the range [0,1]')

    # Split input table into lists
    target_ids = target_types_db['target_id']
    is_H0 = target_types_db['is_h0_target']
    is_vpec = target_types_db['is_vpec_target']
    is_lowz = target_types_db['is_lowz_target']

    score = np.random.rand(target_ids.size).reshape(target_ids.shape)
    # score is a random number between 0 and 1
    # redshift success is when this score is less than Pr(success)
    prob = np.ones(target_types_db.shape)
    # H0 targets will only be observed once; ie. success is guaranteed
    # Therefore, we can just leave prob as 1 for these targets
    # is_vpec = (target_types_db == 'is_vpec_target')
    # success for 20% vpec targets on first visit
    prob = np.where(np.logical_and(is_vpec, num_visits == 1),
                    [prob_vpec_first] * len(prob), prob)
    # success for 70% of vpec targets after two visits
    prob = np.where(np.logical_and(is_vpec, num_visits == 2),
    #                 (
    #     prob_vpec_second - prob_vpec_first
    # ) / (1. - prob_vpec_first),
                    [prob_vpec_second] * len(prob),
                    prob)
    # success for 100% of vpec targets after two visits
    prob = np.where(np.logical_and(is_vpec, num_visits >= 3), 1., prob)
    # is_lowz = (target_types_db == 'is_lowz_target')
    # 80% success for lowz targets on each pass
    prob = np.where(is_lowz, np.minimum(0.8, prob), prob)
    # redshift success is when this score is less than Pr(success)
    success = score < prob
    logging.debug('Out of %d observed targets, %d successes, %d rejections' %
                  (len(target_types_db), np.count_nonzero(success),
                   np.count_nonzero(~success)))
    return success
