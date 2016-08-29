# Simulate failure of starbugs

import numpy as np


def simulate_bugfails(bug_list, prob=1./10000.):
    """
    Simulate a probability of random bug failures during simulations.

    Parameters
    ----------
    bug_list:
        A list corresponding to each current success status of each bug (True
        for successful, False for unsuccessful). Must contain only Booleans
    prob:
        The probability of any particular bug failing on a particular
        observations. Defaults to 1/10000. Must be a float in the range [0,1).

    Returns
    -------
    success_list:
        A list of Booleans, where False denotes a bug failure. This should
        be convolved against the list of redshift_successes to achieve a
        total observing success.
    """

    # Input testing
    if prob <= 0 or prob >= 1.:
        raise ValueError('prob must be in the range [0,1).')

    # Generate a list of random floats
    success_or_fail = np.logical_and(np.random.rand(len(bug_list)) > prob,
                                     bug_list)
    return success_or_fail
