import numpy as np


def test_redshift_success(target_types, num_visits):
    """
    FOR TEST USE ONLY

    Simulates the effects of needing repeat visits of certain targets
    by randomly designating a target as being 'satisfied' or not. The
    probability of this occurring is a function of the priority and number
    of visits.

    Written by Ned Taylor.

    Parameters
    ----------
    target_types:
        A list of target_types. These correspond to the database column names.
    num_visits:
        A corresponding list of the number of times each target has been
        observed already. Note that these values must have been updated for the
        current observation pass *before* test_redshift_success is invoked

    Returns
    -------
    is_satisfied:
        A list of Boolean values denoting whether each target should be
        considered 'satisfied' (i.e. we have enough observations of it).

    """

    # Input checking
    # Make sure we have lists, not single values
    try:
        burn = target_types[0]
    except TypeError:
        target_types = [target_types, ]
    try:
        burn = num_visits[0]
    except TypeError:
        num_visits = [num_visits, ]

    # Check that the length of the two lists match
    if len(target_types) != len(num_visits):
        raise ValueError('Lists of target priorities and num_visits must '
                         'be of equal length')
    # Make sure num_visits has been incremented before calling this function
    # Note that this will only catch non-visited targets which have not been
    # incremented - otherwise, we can't be sure if it's been done or not
    if np.any(num_visits <= 0):
        raise RuntimeError('must increment num_visits before calling '
                           'test_success!')

    score = np.random.rand(target_types.size).reshape(target_types.shape)
    # score is a random number between 0 and 1
    # redshift success is when this score is less than Pr(success)
    prob = np.ones(target_types.shape)
    # H0 targets will only be observed once; ie. success is guaranteed
    # Therefore, we can just leave prob as 1 for these targets
    is_vpec = (target_types == 'is_vpec')
    # success for 20% vpec targets on first visit
    prob = np.where(is_vpec & (num_visits == 1), 0.2, prob)
    # success for 70% of vpec targets after two visits
    prob = np.where(is_vpec & (num_visits == 2), (0.7-0.2)/(1.-0.2), prob)
    # success for 100% of vpec targets after two visits
    prob = np.where(is_vpec & (num_visits == 3), 1., prob)
    is_lowz = (target_types == 'is_lowz')
    # 80% success for lowz targets on each pass
    prob = np.where(is_lowz, 0.8, prob)
    # redshift success is when this score is less than Pr(success)
    return score < prob
