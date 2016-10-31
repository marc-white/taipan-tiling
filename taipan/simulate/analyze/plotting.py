# Functions for plotting up the results of a tiling simulation

import numpy as np

from src.resources.v0_0_1.readout import readTileObservingInfo as rTOI
from src.resources.v0_0_1.readout import readAlmanacStats as rAS
from src.resources.v0_0_1.readout import readCentroids as rC
from src.resources.v0_0_1.readout import readScienceObservingInfo as rSOI

from ..utils.allskymap import AllSkyMap

from ...scheduling import localize_utc_dt, utc_local_dt, POINTING_TIME, \
    UKST_TELESCOPE
from ...core import TILE_RADIUS

import matplotlib
import matplotlib.pyplot
from matplotlib.patches import Circle
# from mpl_toolkits.basemap import Basemap
import datetime
import time

MIDDAY = datetime.time(12,0)


def plot_tiles_per_night(cursor, start_date=None, end_date=None,
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
        fig = matplotlib.pyplot.figure()

    ax = fig.add_subplot(111)
    ax.set_title('Tiles observed: %s to %s' % (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
    ))
    ax.set_xlabel('Date')
    ax.set_ylabel('No. tiles observed')

    def date_vec(x):
        return x.date()
    date_vec = np.vectorize(date_vec)

    ax.hist(obs_tile_info[np.logical_and(
        date_vec(obs_tile_info['date_obs']) >= start_date,
        date_vec(obs_tile_info['date_obs']) <= end_date)]['date_obs'],
            bins=len(nights),
            range=(utc_local_dt(datetime.datetime.combine(start_date, MIDDAY)),
                   utc_local_dt(datetime.datetime.combine(end_date, MIDDAY))),
            label='Tiles observed')

    # ax.plot(np.asarray(nights) + datetime.timedelta(0.5),  # centre on midnight
    #         hrs_dark,
    #         'r-', lw=0.7, label='Dark hours')
    ax.plot(np.asarray([utc_local_dt(datetime.datetime.combine(night, MIDDAY))
                        for night in nights]) +
            datetime.timedelta(0.5),  # centre on midnight
            np.asarray(hrs_dark) / (POINTING_TIME * 24.),
            'g-', lw=0.7, label='Possible tiles per night')

    leg = ax.legend(fontsize=12)

    if not pylab_mode:
        fig.savefig('%s/%s.%s' % (output_loc, output_name, output_fmt),
                    fmt=output_fmt)
    else:
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    return fig


def plot_timedelta_histogram(cursor, pylab_mode=False,
                             output_loc='.',
                             output_name='timedelta_hist',
                             output_fmt='png'):
    """
    Plot a histogram of the unique times between related tile observations
    Parameters
    ----------
    cursor:
        psycopg2 cursor for communicating with the database.
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
        The matplotlib.Figure instance that the figure was drawn into. An image
        file will also be written to disk if pylab_mode=False.
    """

    # Read in the tiles data
    obs_tile_info = rTOI.execute(cursor)
    obs_tile_info.sort(order='date_obs')

    # Generate the timedelta array
    timedeltas = obs_tile_info['date_obs'][1:] - obs_tile_info['date_obs'][:-1]
    timedeltas = map(lambda x: int(x.total_seconds() / 60.), timedeltas)

    # Actually plot
    if pylab_mode:
        matplotlib.pyplot.clf()
        fig = matplotlib.pyplot.gcf()
    else:
        fig = matplotlib.pyplot.figure()

    ax = fig.add_subplot(111)
    ax.set_title('Time between observations:')
    ax.set_xlabel('Time (mins)')
    ax.set_ylabel('No. differences')
    ax.set_yscale('log')

    ax.hist(timedeltas, bins=len(set(timedeltas)),
            label='Differences')

    if not pylab_mode:
        fig.savefig('%s/%s.%s' % (output_loc, output_name, output_fmt),
                    fmt=output_fmt)
    else:
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    return fig


def plot_position_histogram(cursor, start_date=None, end_date=None,
                            rabins=24, decbins=10, pylab_mode=False,
                            output_loc='.',
                            output_name='tiles_per_night',
                            output_fmt='png'):
    """
    Plot a histogram of the number of tiles observed at RA/Dec positions.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for communicating with the database.
    start_date, end_date:
        Optional datetime.date objects, denoting the start and end dates to
        consider when producing the plot. Both default to None, at which point
        these values will be calculated from the observed tiles stored in the
        database.
    rabins, decbins:
        Optional integers, denoting how many histogram bins to split the RA and
        Dec positions of the tile centroids into. Defaults to 24 and 10
        respectively, which will split the full survey area into 10 degree bins
        in both RA and Dec.
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
        The matplotlib.Figure instance that the figure was drawn into. An image
        file will also be written to disk if pylab_mode=False.
    """
    # Input checking
    rabins = int(rabins)
    decbins = int(decbins)

    # Actually plot
    if pylab_mode:
        matplotlib.pyplot.clf()
        fig = matplotlib.pyplot.gcf()
    else:
        fig = matplotlib.pyplot.figure()

    # Read in the tiles data
    obs_tile_info = rTOI.execute(cursor)

    # Construct the 2D histogram and plot
    H, nx, ny = np.histogram2d(
        np.radians((obs_tile_info['ra'] - 180.) % 360 - 180.),
        np.radians(obs_tile_info['dec']),
        bins=[rabins, decbins])

    # axhist = fig.add_subplot(222,
    #                          projection='mollweide',
    #                          )
    axhist = matplotlib.pyplot.subplot2grid((3,3), (0,1), colspan=2,
                                            rowspan=2,
                                            projection='mollweide')
    axhist.grid(True)
    # axhist.set_xlim(0., 360.)
    # axhist.set_ylim(-90., 20.)
    Y, X = np.meshgrid(ny, nx)
    hist2d = axhist.pcolor(X, Y, H,
                           cmap='hot')
    histcb = fig.colorbar(hist2d, ax=axhist,
                          # use_gridspec=True
                          )
    histcb.set_label('Tiles observed')

    # Histogram of RA
    axra = matplotlib.pyplot.subplot2grid((3,3), (2, 1), colspan=2)
    histra = axra.hist(
        (obs_tile_info['ra'] - 180.) % 360 - 180., bins=rabins)
    axra.set_xlim((-180., 180.))
    axra.set_xlabel('Right ascension (deg)')
    axra.set_ylabel('Tiles observed')

    # Histogram of Dec
    axdec = matplotlib.pyplot.subplot2grid((3, 3), (0, 0), rowspan=2)
    histdec = axdec.hist(
        obs_tile_info['dec'],
        bins=rabins,
        orientation='horizontal')
    axdec.set_ylim((-90., 90.))
    axdec.set_ylabel('Declination')
    axdec.set_xlabel('Tiles observed')

    if not pylab_mode:
        fig.savefig('%s/%s.%s' % (output_loc, output_name, output_fmt),
                    fmt=output_fmt)
    else:
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    return fig


def plot_observing_sequence(cursor, start_date=None, end_date=None,
                            rotate=True, pylab_mode=False,
                            output_loc='.',
                            output_prefix='obs_seq',
                            output_fmt='png'):
    """
    Plot an observing sequence diagnostic chart.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for interacting with the results database
    start_datetime, end_datetime:
        Start and end datetimes in *UTC*. Both default to None, so these values
        will be computed from the observed tiles in the database.
    rotate:
        Optional Boolean, denoting whether to rotate the plot such that the
        most accessible hour angle at that time is centred. Defaults to True.
    pylab_mode:
        Optional Boolean, denoting whether to run in interactive Pylab mode
        (True) or just write out to file (False). Defaults to False.
    output_loc:
        Optional string, denoting where to write output to in the event
        the function is not run in Pylab mode. Defaults to '.' (i.e. the present
        working directory).
    output_prefix:
        Optional string, denoting the name of the output image. Defaults to
        'tiles_per_night'.
    output_fmt:
        Optional string, denoting the format of the output image. Defaults to
        'png'. Should be a valid matplotlib format specifier.


    Returns
    -------
     fig:
        A matplotlib.Figure instance referencing the figure instance used.
        In addition, either an image file will be written to disk
        (pylab_mode=False), or an interactive Pylab window will be
        opened/updated (pylab_mode=True).
    """

    # Let's get the information for all the observed tiles to start with1
    obs_tile_info = rTOI.execute(cursor)
    if len(obs_tile_info) == 0:
        # No tiles observed - abort
        return

    # Pull in all the science targets for plotting
    # sci_targets = rSc.execute(cursor)

    # Get the science observing info
    targets_db, targets_tiles, targets_completed = rSOI.execute(cursor)

    # Read in the tile observing info
    obs_tile_info = rTOI.execute(cursor)
    dates = map(lambda x: x.date(), obs_tile_info['date_obs'])
    dates = np.asarray(dates)
    # print dates
    if start_date is not None:
        obs_tile_info = obs_tile_info[dates >= start_date]
    if end_date is not None:
        obs_tile_info = obs_tile_info[dates <= end_date]
    obs_tile_info.sort(order='date_obs')
    # obs_tile_info = obs_tile_info[:20]

    # Prepare the Figure instance
    if pylab_mode:
        matplotlib.pyplot.clf()
        fig = matplotlib.pyplot.gcf()
    else:
        fig = matplotlib.pyplot.figure()
        fig.set_size_inches((9, 6))

    # ax1 = fig.add_subplot(
    #     111,
    #     projection='mollweide'
    # )
    # ax1.grid(True)

    mapplot = AllSkyMap(projection='moll', lon_0=180.)
    limb = mapplot.drawmapboundary(fill_color='white')
    mapplot.drawparallels(np.arange(-75, 76, 15), linewidth=0.5, dashes=[1, 2],
                      labels=[1, 0, 0, 0], fontsize=9)
    mapplot.drawmeridians(np.arange(30, 331, 30), linewidth=0.5, dashes=[1, 2])

    # m = Basemap(projection='moll', lon_0=0., ax=ax1)
    # m.drawparallels(np.arange(-90., 90., 15.), labels=range(-90, 90, 15))
    # m.drawmeridians(np.arange(0., 360., 30.))
    # # Hack - basemap needs the list sorted by longitude
    # targets_db.sort(order='ra')
    # m.scatter(list((targets_db['ra']-180.) % 360 - 180.),
    #           list(targets_db['dec']), latlon=True,
    #           s=1, edgecolors=None, c='black', alpha=0.05)

    # p_tgts_all = ax1.scatter(np.radians((targets_db['ra'] - 180.) % 360 - 180.),
    #                          np.radians(targets_db['dec']),
    #                          1, 'k', edgecolors='none',
    #                          rasterized=True, alpha=0.02)

    # To save computational power, plot the targets as a fine-grained
    # 2d histogram
    # H, nx, ny = np.histogram2d(np.radians((targets_db['ra'] - 180.) %
    #                                       360 - 180.),
    #                            np.radians(targets_db['dec']),
    #                            bins=[360, 90 - int(np.min(targets_db['dec']))])
    # Y, X = np.meshgrid(ny, nx)
    # ax1.pcolor(X, Y, H, cmap='Greys', vmax=np.max(H)/4.0)

    H, nx, ny = np.histogram2d(targets_db['ra'],
                               targets_db['dec'],
                               bins=[360, 90 - int(np.min(targets_db['dec']))])
    Y, X = np.meshgrid(ny, nx)
    mapplot.pcolor(X, Y, H, cmap='Greys', vmax=np.max(H) / 4.0,
                   latlon=True,
                   )

    # Get and plot the field centroids
    cents = rC.execute(cursor)
    # for c in cents:
    #     ax1.add_patch(Circle(
    #         (np.radians((c.ra - 180.) % 360 - 180.), np.radians(c.dec)),
    #         radius=np.radians(TILE_RADIUS / 3600.), facecolor='none',
    #         edgecolor='r',
    #         ls='--', lw=0.7
    #     ))

    # Plot the observed tiles in sequence
    new_seq = True
    moves_this_seq = 0
    geos_this_seq = []
    z_marker = None
    curr_tissot = None
    for i in range(len(obs_tile_info)):
        if i > 0:
            t_p = obs_tile_info[i-1]
        t = obs_tile_info[i]
        # ax1.set_title(t['date_obs'].strftime('%Y-%m-%d %H:%M:%S'))
        # ax1.add_patch(Circle(
        #     (np.radians((t['ra'] - 180.) % 360 - 180.), np.radians(t['dec'])),
        #     radius=np.radians(TILE_RADIUS / 3600.), facecolor='red',
        #     edgecolor='none', lw=0.7, alpha=0.1
        # ))
        if curr_tissot:
            for _ in curr_tissot[::-1]:
                k = curr_tissot.pop(-1)
                k.remove()
            mapplot.tissot(t_p['ra'], t_p['dec'], TILE_RADIUS / 3600., 100,
                           facecolor='red',
                           edgecolor='none', lw=0.7, alpha=0.1)
        curr_tissot = mapplot.tissot(t['ra'], t['dec'], TILE_RADIUS / 3600.,
                                     100,
                                     facecolor='yellow',
                                     edgecolor='none', lw=0.7, alpha=1.)

        # Compute and plot the zenith position, removing the old one if known
        UKST_TELESCOPE.date = t['date_obs']
        z_ra, z_dec = map(np.degrees,
                          UKST_TELESCOPE.radec_of(0.0, np.radians(90.)))
        if z_marker:
            pass
            z_marker.remove()
        z_marker = mapplot.scatter([z_ra]*3, [z_dec]*3, s=30, marker='x',
                                   edgecolors='none', facecolors='green',
                                   latlon=True, zorder=10)

        if i > 0 and (obs_tile_info[i]['date_obs'] -
                          obs_tile_info[i-1]['date_obs'] <
                          datetime.timedelta(minutes=60)):
            new_seq = False
            # ax1.add_line(matplotlib.lines.Line2D(
            #     np.radians((np.asarray([obs_tile_info[i-1]['ra'],
            #                             obs_tile_info[i]['ra']]) - 180.) %
            #                360 - 180.),
            #     np.radians(np.asarray([obs_tile_info[i-1]['dec'],
            #                            obs_tile_info[i]['dec']])),
            #     color='cyan', alpha=0.5, linewidth=3
            # ))
            g = mapplot.geodesic(obs_tile_info[i-1]['ra'],
                             obs_tile_info[i-1]['dec'],
                             obs_tile_info[i]['ra'], obs_tile_info[i]['dec'],
                             color='cyan', alpha=0.5, linewidth=3)
            moves_this_seq += len(g)
            geos_this_seq += g
        else:
            # print geos_this_seq
            new_seq = True
            for _ in geos_this_seq[::-1]:
            # for i in range(moves_this_seq)[::-1]:
            #     del mapplot.ax.lines[-1]
                _ = geos_this_seq.pop(-1)
                _.remove()
                # l = geos_this_seq.pop(i)
                # del l
            moves_this_seq = 0
            geos_this_seq = []

        matplotlib.pyplot.title('%s' %
                                t['date_obs'].strftime('%Y-%m-%d %H:%M:%S'))

        if not pylab_mode:
            # matplotlib.pyplot.draw()
            fig.savefig('%s/%s-%s.%s' % (
                output_loc,
                output_prefix,
                t['date_obs'].strftime('%Y-%m-%d-%H-%M-%S'),
                output_fmt
            ), fmt=output_fmt, dpi=300)

    if pylab_mode:
        matplotlib.pyplot.show()
        matplotlib.pyplot.draw()
    else:
        # Save the figure
        pass

    return fig
