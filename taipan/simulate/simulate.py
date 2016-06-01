import numpy as np


def test_redshift_success(priority, num_visits):
    """
    FOR TEST USE ONLY

    Simulates the effects of needing repeat visits of certain targets
    by randomly designating a target as being 'satisfied' or not. The
    probability of this occurring is a function of the priority and number
    of visits.

    Written by Ned Taylor.

    Parameters
    ----------
    priority:
        The priority of the target.
    num_visits:
        The number of times to target has been observed already. Note that this
        must be updated *before* invoking this function.

    Returns
    -------
    is_satisfied:
        A Boolean value denoting whether this target should be considered
        'satisfied' (i.e. we have enough observations of it).

    """

    if np.any(num_visits <= 0):
        print 'must increment num_visits before calling test_success!' 
    score = np.random.rand(priority.size).reshape(priority.shape)
    # score is a random number between 0 and 1
    # redshift success is when this score is less than Pr(success)
    prob = np.ones(priority.shape)
    # H0 targets will only be observed once; ie. success is guaranteed
    is_vpec = (priority == 4) | (priority > 7)
    # success for 20% vpec targets on first visit
    prob = np.where(is_vpec & (num_visits == 1), 0.2, prob)
    # success for 70% of vpec targets after two visits
    prob = np.where(is_vpec & (num_visits == 2), (0.7-0.2)/(1.-0.2), prob)
    # success for 100% of vpec targets after two visits
    prob = np.where(is_vpec & (num_visits == 3), 1., prob)
    is_lowz = (priority == 9)
    # 80% success for lowz targets on 
    prob = np.where(is_lowz, 0.8, prob)
    # redshift success is when this score is less than Pr(success)
    return score < prob
