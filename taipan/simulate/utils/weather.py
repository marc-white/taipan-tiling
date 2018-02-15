# Utility functions for simulating weather interruptions

import datetime
import numpy as np
from scipy.stats import truncnorm


def value_from_trunc_norm(loc, scale, trim_low, trim_high):
    """
    INTERNAL HELPER FUNCTION

    Get a value from a PDF described by a truncated normal.

    Parameters
    ----------
    loc:
        The central location of the truncated Gaussian.
    scale:
        The width (i.e. std) scaling of the truncated Gaussian.
    trim_low, trim_high:
        The lower and upper bounds of the trimmed Gaussian.

    Returns
    -------
    rv:
        A random variate (i.e. randomly generated value) from the
        truncated Gaussian PDF.
    """
    # Determine the trim limits in the altered Gaussian space
    a = (trim_low - loc) / scale
    b = (trim_high - loc) / scale

    # Generate a random variate from the function
    rv = truncnorm.rvs(a, b, loc=loc, scale=scale, size=1)[0]

    # Return the value
    return rv


# def simulate_weather_interrupt(datetime, weather_function, **kwargs):
#     """
#     Simulate a weather interruption using a pre-defined, date-specific
#     weather function. This function acts as a wrapper for those functions.
#
#     Parameters
#     ----------
#     date:
#         The current datetime. This is important for some of the weather
#         functions, which will return success of failure with a probability
#         dependent on the date.
#     **kwargs
#
#     Returns
#     -------
#     weather_fail:
#         A Boolean value denoting if a weather fail has been triggered.
#     datetime_to:
#         A datetime at which the weather block can be considered complete. Will
#         be None if weather_fail is False.
#     """
#     pass


def wf_flat(dt, prob=0.005, fail_mean=120., fail_width=120., **kwargs):
    """
    Weather function which returns a flat probability of a weather failure,
    regardless of the current datetime.
    Parameters
    ----------
    prob:
        Float probability of a weather failure in the range [0,1). Defaults
        to 0.005 (i.e. 0.5% at any given time interval).

    Returns
    -------
    wf:
        Boolean value denoting whether a weather failure has occurred.
    """
    # Input testing
    if prob <= 0 or prob >= 1.:
        raise ValueError('prob must be in the range [0,1).')

    wf = np.random.rand(1)[0] < prob

    if wf:
        # Let's work out the time when the weather fail will end
        len_mins = value_from_trunc_norm(fail_mean, fail_width, 10., 20000.)
        time_to_resume = dt + datetime.timedelta(seconds=60.*len_mins)
        return wf, time_to_resume

    return wf, None
