# Functions for plotting up the results of a tiling simulation

import numpy as np

from src.resources.v0_0_1.readout import readTileObservingInfo as rTOI
from src.resources.v0_0_1.readout import readAlmanacStats as rAS
from ...scheduling import localize_utc_dt, utc_local_dt, POINTING_TIME

import matplotlib
import datetime

MIDDAY = datetime.time(12,0)


def tiles_per_night(cursor, start_date=None, end_date=None,
                    pylab_mode=False, output_loc='.',
                    output_name='tiles_per_night',
                    output_fmt='png'):
    """
    Plot a histogram of the number of tiles observed per night. Histogram is
    binned on a 'nightly' basis.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for communicating with the database.
    start_date, end_date:
        Optional datetime.date objects, denoting the start and end dates to
        consider when producing the plot. Both default to None, at which point
        these values will be calculated from the observed tiles stored in the
        database.
    pylab_mode:
        Optional Boolean, denoting whether to run in interactive Pylab mode
        (True) or just write out to file (False). Defaults to False.
    output_loc:
        Optional string, denoting where to write output to in the event
        the function is not run in Pylab mode. Defaults to '.' (i.e. the present
        working directory).
    output_name:
        Optional string, denoting the name of the output image. Defaults to
        'tiles_per_night'.
    output_fmt:
        Optional string, denoting the format of the output image. Defaults to
        'png'. Should be a valid matplotlib format specifier.

    Returns
    -------
    fig:
        A matplotlib.Figure instance referencing the figure made. In addition,
        either an image file will be written to disk (pylab_mode=False), or
        an interactive Pylab window will be opened/updated (pylab_mode=True).
    """

    # Read the necessary data in from the database
    # It's easiest just to read in all the tiles, and filter on date later -
    # we either need all this information to calculate start and end dates,
    # or we're plotting them all anyway
    obs_tile_info = rTOI.execute(cursor)

    # Compute the start and/or end dates if necessary
    if start_date is None:
        # Compute a start_date for plotting
        start_date = localize_utc_dt(np.min(obs_tile_info['date_obs'])).date()
    else:
        obs_tile_info = obs_tile_info[np.asarray(
            map(lambda x: localize_utc_dt(x).date(),
                obs_tile_info['date_obs'])) >= start_date]
    if end_date is None:
        # Compute an end_date for plotting
        end_date = localize_utc_dt(np.max(obs_tile_info['date_obs'])).date()
    else:
        # Filter the tile_list
        obs_tile_info = obs_tile_info[np.asarray(
            map(lambda x: localize_utc_dt(x).date(),
                obs_tile_info['date_obs'])) >= start_date]

    # Compute the number of hours available for observation during each night
    nights = [start_date + datetime.timedelta(n) for n in
              range((end_date - start_date).days)]
    hrs_dark = [rAS.hours_observable(cursor, 1,  # field_id - dummy value
                                     utc_local_dt(
                                         datetime.datetime.combine(night,
                                                                   MIDDAY)),
                                     utc_local_dt(
                                         datetime.datetime.combine(night +
                                                                   datetime.
                                                                   timedelta(1),
                                                                   MIDDAY)),
                                     hours_better=False,
                                     minimum_airmass=10.0) for
                night in nights]

    # Actually plot
    if pylab_mode:
        matplotlib.pyplot.clf()
        fig = matplotlib.pyplot.gcf()
    else:
        fig = matplotlib.pyplot.Figure()

    ax = fig.add_subplot(111)
    ax.set_title('Tiles observed: %s to %s' % (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
    ))
    ax.set_xlabel('Date')
    ax.set_ylabel('No. tiles observed')

    ax.hist(obs_tile_info['date_obs'], bins=(end_date - start_date).days,
            label='Tiles observed')

    ax.plot(np.asarray(nights) + datetime.timedelta(0.5),  # centre on midnight
            hrs_dark,
            'r-', lw=0.7, label='Dark hours')
    ax.plot(np.asarray(nights) + datetime.timedelta(0.5),  # centre on midnight
            np.asarray(hrs_dark) / (POINTING_TIME * 24.),
            'g-', lw=0.7, label='Possible tiles per night')

    leg = ax.legend(fontsize=7)

    if not pylab_mode:
        fig.savefig('%s/%s.%s' % (output_loc, output_name, output_fmt),
                    fmt=output_fmt)
    else:
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    return fig
