# Utility functions for simulating weather interruptions

import datetime
import numpy as np


def simulate_weather_interrupt(datetime, weather_function, **kwargs):
    """
    Simulate a weather interruption using a pre-defined, date-specific
    weather function. This function acts as a wrapper for those functions.

    Parameters
    ----------
    date:
        The current datetime. This is important for some of the weather
        functions, which will return success of failure with a probability
        dependent on the date.
    Further keyword arguments may be passed, depending on which weather
    function is invoked.

    Returns
    -------
    weather_fail:
        A Boolean value denoting if a weather fail has been triggered.
    datetime_to:
        A datetime at which the weather block can be considered complete. Will
        be None if weather_fail is False.
    """


def wf_flat(datetime, prob=0.01):
    """
    Weather function which returns a flat probability of a weather failure,
    regardless of the current datetime.
    Parameters
    ----------
    prob:
        Float probability of a weather failure in the range [0,1). Defaults
        to 0.01 (i.e. 1% at any given time interval).

    Returns
    -------
    wf:
        Boolean value denoting whether a weather failure has occurred.
    """
    wf = np.random.rand(1)[0] < prob
    return wf