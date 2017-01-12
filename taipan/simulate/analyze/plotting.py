# Functions for plotting up the results of a tiling simulation

import numpy as np

from src.resources.v0_0_1.readout import readTileObservingInfo as rTOI
from src.resources.v0_0_1.readout import readAlmanacStats as rAS
from src.resources.v0_0_1.readout import readCentroids as rC
from src.resources.v0_0_1.readout import readFibrePosns as rFP
from src.resources.v0_0_1.readout import readScienceObservingInfo as rSOI
from src.resources.v0_0_1.readout.readCentroidsByTarget import execute as \
    rCBTexec
from src.resources.v0_0_1.readout.readObservingLog import execute as rOLexec
from src.resources.v0_0_1.readout.readScienceVisits import execute as rScVexec
from src.resources.v0_0_1.readout.readScienceTypes import execute as rScTexec
from src.resources.v0_0_1.readout.readSciencePosn import execute as rScPexec
from src.resources.v0_0_1.readout.readScienceDates import execute as rScDexec

from ..utils.allskymap import AllSkyMap

from ...scheduling import localize_utc_dt, utc_local_dt, POINTING_TIME, \
    UKST_TELESCOPE
from ...core import TILE_RADIUS, BUGPOS_MM, dist_points, PATROL_RADIUS, \
    ARCSEC_PER_MM

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, colors
from matplotlib.patches import Circle
from mpl_toolkits.basemap import Basemap
import datetime
import time
import sys
import logging

MIDDAY = datetime.time(12,0)

EARTH_RADIUS = 6371000 # m


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
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()

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
        plt.draw()
        plt.show()

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
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()

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
        plt.draw()
        plt.show()

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
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()

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
    axhist = plt.subplot2grid((3,3), (0,1), colspan=2,
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
    axra = plt.subplot2grid((3,3), (2, 1), colspan=2)
    histra = axra.hist(
        (obs_tile_info['ra'] - 180.) % 360 - 180., bins=rabins)
    axra.set_xlim((-180., 180.))
    axra.set_xlabel('Right ascension (deg)')
    axra.set_ylabel('Tiles observed')

    # Histogram of Dec
    axdec = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
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
        plt.draw()
        plt.show()

    return fig


def plot_observing_sequence(cursor, start_date=None, end_date=None,
                            rotate=True,
                            pylab_mode=False,
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
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
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

    targets_priority = np.logical_or(targets_db['is_h0_target'],
                                     targets_db['is_vpec_target'])
    targets_priority = np.logical_or(targets_priority,
                                     targets_db['is_lowz_target'])

    H, nx, ny = np.histogram2d(targets_db[targets_priority]['ra'],
                               targets_db[targets_priority]['dec'],
                               bins=[360, 90 - int(
                                   np.min(targets_db[
                                              targets_priority]['dec']))])
    Y, X = np.meshgrid(ny, nx)
    mapplot.pcolor(X, Y, H, cmap='Greys', vmax=np.max(H),
                   latlon=True,
                   )

    # Plot the KiDS region
    kids_region = [
        [329.5, 53.5, 53.5, 329.5, 329.5],
        [-35.6, -35.6, -25.7, -25.7, -35.6]
    ]

    for i in range(len(kids_region[0]) - 1):
        mapplot.geodesic(kids_region[0][i], kids_region[1][i],
                         kids_region[0][i+1], kids_region[1][i+1],
                         color='limegreen', lw=1.2, ls='--')

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

        plt.title('%s' %
                                t['date_obs'].strftime('%Y-%m-%d %H:%M:%S'))

        if not pylab_mode:
            # plt.draw()
            fig.savefig('%s/%s-%s.%s' % (
                output_loc,
                output_prefix,
                t['date_obs'].strftime('%Y-%m-%d-%H-%M-%S'),
                output_fmt
            ), fmt=output_fmt, dpi=300)

    if pylab_mode:
        plt.show()
        plt.draw()
    else:
        # Save the figure
        pass

    return fig


def plot_target_completeness_time(cursor,
                                  start_time=None,
                                  end_time=None,
                                  pylab_mode=False,
                                  by_type=True,
                                  output_loc='.',
                                  output_prefix='completeness',
                                  output_fmt='png',
                                  ):
    """
    Plot the completeness of target types as a function of time.
    Parameters
    ----------
    cursor : psycopg2 cursor
        Cursor for interacting with the database
    start_time, end_time : datetime.datetime instances, optional
        UTC datetimes specifying start and end times for the plot. Both default
        to None, such that the full observing time range in the database will
        be plotted.
    pylab_mode : Boolean, optional
        Denotes whether to run in interactive pylab mode (True) or write output
        to disk (False). Defaults to False.
    by_type: Boolean, optional
        Denotes whether to group targets by their science type (True), or by
        their priority (False). Defaults to True. Note that grouping by
        science type will create some overlap, as targets may have multiple
        science types.
    output_loc, output_prefix, output_fmt: strings
        Strings denoting the location to write output (relative or absolute,
        defaults to '.'),
        the prefix of the output files (defaults to 'hrs_bet'), and the format
        of the output (defaults to 'png').

    Returns
    -------
    fig : plt.Figure
        The Figure instance the plot was made on.
    """

    lincols = [
        'black',
        'green',
        'red',
        'gold',
        'grey',
        'orange',
        'purple',
        'darkgreen',
        'cyan',
    ]

    # Read in the observing log from the database, and order it by date_obs
    obs_log = rOLexec(cursor)

    # Read in the visits/repeats information
    target_status = rScVexec(cursor)

    if by_type:
        groups = ['is_h0_target', 'is_vpec_target', 'is_lowz_target']
    else:
        groups = list(set(obs_log['priority']))

    # Work out the start and end date if not given
    if not start_time:
        start_time = np.min(obs_log['date_obs'])
    if not end_time:
        end_time = np.max(obs_log['date_obs'])

    xpts = []
    ypts_started = {g: [] for g in groups}
    ypts_done = {g: [] for g in groups}
    if by_type:
        ypts_started['Filler'] = []
        ypts_done['Filler'] = []

    target_visit_mask = np.in1d(obs_log['target_id'],
                                target_status[np.logical_or(
                                    target_status['visits'] > 0,
                                    target_status['repeats'] > 0,
                                    )]['target_id'])
    target_done_mask = np.in1d(obs_log['target_id'],
                               target_status[
                                   target_status['repeats'] > 0]['target_id'])

    # Make a count of the number of targets in each group
    sci_types = rScTexec(cursor)
    sci_types_count = {}
    if by_type:
        sci_types_count = {group: np.count_nonzero(sci_types[group]) for
                           group in groups}
        sci_types_count['Filler'] = np.count_nonzero(
            ~(np.logical_or(np.logical_or(sci_types['is_h0_target'],
                                          sci_types['is_lowz_target'],),
                            sci_types['is_vpec_target']))
        )
    else:
        sci_types_count = {group:
                               np.count_nonzero(sci_types['priority'] == group)
                           for group in groups}
    print sci_types_count

    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
        fig.set_size_inches((9, 6))
    # Prepare the axes
    ax = fig.add_subplot(111)
    ax.set_xlim(start_time, end_time)
    # ax.set_ylim(0, int(1.1 * np.count_nonzero(target_done_mask)))
    months = mdates.MonthLocator()
    mth_fmt = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mth_fmt)

    curr_time = start_time
    leg = None
    while curr_time <= end_time:
        print curr_time
        xpts.append(curr_time)

        if leg:
            leg.remove()
        for l in ax.lines[::]:
            l.remove()

        for group in groups:
            # Count the number of targets that have been started, and the
            # number that have completed
            if by_type:
                started = np.count_nonzero(np.unique(
                    obs_log[
                        np.logical_and(
                            np.logical_and(
                                target_visit_mask,
                                obs_log[group]
                            ),
                            obs_log['date_obs'] <= curr_time
                        )
                    ]['target_id']))
                done_ids = np.unique(obs_log[
                                         np.logical_and(target_done_mask,
                                                        obs_log[group])
                                     ]['target_id'])
                # print done_ids
                done = np.count_nonzero(~np.in1d(
                    done_ids,
                    obs_log[obs_log['date_obs'] > curr_time]['target_id']
                ))
            else:
                started = np.count_nonzero(np.unique(
                    obs_log[
                        np.logical_and(
                            np.logical_and(
                                target_visit_mask,
                                obs_log['priority'] == group
                            ),
                            obs_log['date_obs'] <= curr_time
                        )
                    ]['target_id']))
                done_ids = np.unique(obs_log[
                                         np.logical_and(target_done_mask,
                                                        obs_log[group])
                                     ]['target_id'])
                # print done_ids
                done = np.count_nonzero(~np.in1d(
                    done_ids,
                    obs_log[obs_log['date_obs'] > curr_time]['target_id']
                ))
            ypts_started[group].append(started)
            ypts_done[group].append(done)
        # For the special case of 'filler', we need to do this manually, outside
        # the groups loop
        if by_type:
            started = np.count_nonzero(np.unique(
                obs_log[
                    np.logical_and(
                        np.logical_and(
                            target_visit_mask,
                            ~np.logical_or(
                                np.logical_or(obs_log['is_h0_target'],
                                              obs_log['is_vpec_target']),
                                obs_log['is_lowz_target']
                            )
                        ),
                        obs_log['date_obs'] <= curr_time
                    )
                ]['target_id']))
            done_ids = np.unique(obs_log[
                                     np.logical_and(target_done_mask,
                                                    ~np.logical_or(
                                                        np.logical_or(
                                                            obs_log[
                                                                'is_h0_target'],
                                                            obs_log[
                                                                'is_vpec_target']),
                                                        obs_log['is_lowz_target']
                                                    ))
                                 ]['target_id'])
            done = np.count_nonzero(~np.in1d(
                done_ids,
                obs_log[obs_log['date_obs'] > curr_time]['target_id']
            ))
            ypts_started['Filler'].append(started)
            ypts_done['Filler'].append(done)

        for i in range(len(groups)):
            ax.plot(xpts, ypts_started[groups[i]], '--', color=lincols[i])
            ax.plot(xpts, ypts_done[groups[i]], '-', color=lincols[i],
                    label=str(groups[i]))
            ax.plot(ax.get_xlim(), [sci_types_count[group]]*2, ':',
                    color=lincols[i])
        if by_type:
            ax.plot(xpts, ypts_started['Filler'], '--',
                    color=lincols[len(groups)])
            ax.plot(xpts, ypts_done['Filler'], '-',
                    color=lincols[len(groups)],
                    label='Filler')
            # ax.plot(ax.get_xlim(), [sci_types_count['Filler']] * 2, ':',
            #         color=lincols[len(groups)])

        leg = ax.legend()

        fig.autofmt_xdate()

        # print '%s, %s: %5d started, %5d done' % (str(group),
        #                       curr_time.strftime('%Y-%m-%d %H:%M'), started,
        #                                          done)

        if not pylab_mode:
            # plt.draw()
            fig.savefig('%s/%s-%s.%s' % (
                output_loc,
                output_prefix,
                curr_time.strftime('%Y-%m-%d-%H-%M-%S'),
                output_fmt
            ), fmt=output_fmt, dpi=300)

        curr_time += np.min(
            obs_log[obs_log['date_obs'] > curr_time]['date_obs']
        ) - curr_time

        # sys.exit()

    return fig


def plot_hours_remain_analysis(cursor,
                               start_time,
                               end_time,
                               rotate=True,
                               pylab_mode=False,
                               output_loc='.',
                               output_prefix='hrs_bet',
                               output_fmt='png',
                               prioritize_lowz=True):
    """
    Plot an analysis of the hours_remaining parameter for each field in the
    survey.

    Parameters
    ----------
    cursor : psycopg2 cursor
        For communicating with the database.
    start_time, end_time: datetime.datetime instances
        Limiting start and end times for analysis. Both default to None,
        at which point a value error will be raised
    rotate: can't remember what this does
    pylab_mode : Boolean, optional
        Denotes whether to run in interactive pylab mode (True) or write output
        to disk (False). Defaults to False.
    output_loc, output_prefix, output_fmt: strings
        Strings denoting the location to write output (relative or absolute,
        defaults to '.'),
        the prefix of the output files (defaults to 'hrs_bet'), and the format
        of the output (defaults to 'png').
    prioritize_lowz : Boolean, optional
        Whether or not to prioritize fields with lowz targets by computing
        the hours_observable for those fields against a set end date, and
        compute all other fields against a rolling one-year end date.
        Defaults to True.

    Returns
    -------
    fig: plt.Figure instance
        The figure instance the information was plotted to.
    """
    # Input checking
    if start_time is None or end_time is None:
        raise ValueError('Must specify both a start_date and end_date')

    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
        fig.set_size_inches((9, 6))

    # ax1 = fig.add_subplot(
    #     111,
    #     projection='mollweide'
    # )
    # ax1.grid(True)

    mapplot = AllSkyMap(projection='moll', lon_0=0.)
    limb = mapplot.drawmapboundary(fill_color='white')
    mapplot.drawparallels(np.arange(-75, 76, 15), linewidth=0.5, dashes=[1, 2],
                          labels=[1, 0, 0, 0], fontsize=9)
    mapplot.drawmeridians(np.arange(30, 331, 30), linewidth=0.5, dashes=[1, 2])

    # Plot the KiDS region
    kids_region = [
        [329.5, 53.5, 53.5, 329.5, 329.5],
        [-35.6, -35.6, -25.7, -25.7, -35.6]
    ]

    for i in range(len(kids_region[0]) - 1):
        mapplot.geodesic(kids_region[0][i], kids_region[1][i],
                         kids_region[0][i + 1], kids_region[1][i + 1],
                         color='limegreen', lw=1.2, ls='--', zorder=20)

    # Read in the field information
    fields = rC.execute(cursor)
    # Turn this into dicts for each of plotting
    fields_ra = [t.ra for t in fields]
    fields_dec = [t.dec for t in fields]
    fields = [t.field_id for t in fields]

    for i in range(len(fields_ra)):
        if abs(fields_ra[i] - 0.) < 1e-5:
            fields_ra[i] = 0.1
        if abs(fields_dec[i] - 0) < 1e-5:
            fields_dec[i] = 0.1

    # print fields_ra[:30]
    # print fields_dec[:30]
    # print fields[:30]

    local_utc_now = start_time

    z_marker = None

    while local_utc_now <= end_time:
        print local_utc_now.strftime('%Y-%m-%d %H:%M:%S')
        dark_start, dark_end = rAS.next_night_period(cursor, local_utc_now,
                                                     limiting_dt=local_utc_now +
                                                     datetime.timedelta(1),
                                                     dark=True, grey=False)

        field_periods = {r: rAS.next_observable_period(
            cursor, r, local_utc_now,
            datetime_to=dark_end) for
                         r in fields}
        # logging.debug('Next observing period for each field:')
        # logging.debug(field_periods)
        # logging.info('Next available field will rise at %s' %
        #              (min([v[0].strftime('%Y-%m-%d %H:%M:%S') for v in
        #                    field_periods.itervalues() if
        #                    v[0] is not None]),)
        #              )
        fields_available = [f for f, v in field_periods.iteritems() if
                            v[0] is not None and v[0] < dark_end]
        # logging.debug('%d fields available at some point tonight' %
        #               len(fields_available))

        # Compute the hours_remaining information
        if prioritize_lowz:
            lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                                   unobserved=True)
            hours_obs_lowz = {f: rAS.hours_observable(cursor, f,
                                                      local_utc_now,
                                                      datetime_to=max(
                                                          end_time,
                                                          local_utc_now +
                                                          datetime.
                                                          timedelta(30.)
                                                      ),
                                                      hours_better=True) for
                              f in fields if
                              f in lowz_fields}
            hours_obs_oth = {f: rAS.hours_observable(cursor, f,
                                                     local_utc_now,
                                                     datetime_to=
                                                     local_utc_now +
                                                     datetime.timedelta(
                                                         365),
                                                     hours_better=True) for
                             f in fields}
            hours_obs = dict(hours_obs_lowz, **hours_obs_oth)
        else:
            hours_obs = {f: rAS.hours_observable(cursor, f, local_utc_now,
                                                 datetime_to=end_time,
                                                 hours_better=True) for
                         f in fields}



        # Compute and plot the zenith position, removing the old one if known
        UKST_TELESCOPE.date = local_utc_now
        z_ra, z_dec = map(np.degrees,
                          UKST_TELESCOPE.radec_of(0.0, np.radians(90.)))
        if z_marker:
            z_marker.remove()
        z_marker = mapplot.scatter([z_ra] * 3, [z_dec] * 3, s=50,
                                   marker='x',
                                   edgecolors='none', facecolors='black',
                                   latlon=True, zorder=11)

        # for f in hours_obs.keys():
        #     x = mapplot.tissot(fields_ra[f], fields_dec[f], TILE_RADIUS/3600.,
        #                        50, lw=0., edgecolor='none',
        #                        facecolor=cm.jet(max(hours_obs[f] / 600., 1.)))
        #     polys = polys + x
        dots = []
        step = 11
        for k in range(0, len(fields_ra), step):
            # print k
            try:
                dots = mapplot.scatter(
                    # [90.]*3, [-45.]*3,
                    # fields_ra[30:33], fields_dec[30:33],
                    fields_ra[k:k+step], fields_dec[k:k+step],
                    s=30, marker='o',
                    edgecolors='none',
                    facecolors=
                    # 'red',
                    # cm.jet,
                    # 'red',
                    [hours_obs[k] for k in range(k, k+step)
                     if k in hours_obs.keys()],
                    cmap=cm.jet,
                    vmin=0.0, vmax=1200.,
                    latlon=True,
                    zorder=8,
                )
            except ValueError:
                pass

        fig.suptitle(local_utc_now.strftime('%Y-%m-%d %H:%M:%S'))

        if pylab_mode:
            plt.show()
            plt.draw()
        else:
            fig.savefig('%s/%s-%s.%s' % (
                output_loc,
                output_prefix,
                local_utc_now.strftime('%Y-%m-%d-%H-%M-%S'),
                output_fmt
            ), fmt=output_fmt, dpi=300)

        local_utc_now += datetime.timedelta(minutes=15)


def plot_fibre_stretch(cursor,
                       pylab_mode=False,
                       output_loc='.',
                       output_prefix='fibre_stretch',
                       output_fmt='png',
                       ):
    """
    Compute and plot the fibre 'stretch' (i.e. the distance the fibres had
    to travel from their home position) for this observation set.

    Parameters
    ----------
    cursor : psycopg2 cursor
        cursor for communication with the database
    pylab_mode : Boolean, optional
        Denotes whether to run in interactive pylab mode (True) or write output
        to disk (False). Defaults to False.
    output_loc, output_prefix, output_fmt: strings
        Strings denoting the location to write output (relative or absolute,
        defaults to '.'),
        the prefix of the output files (defaults to 'hrs_bet'), and the format
        of the output (defaults to 'png').

    Returns
    -------
    fig: plt.Figure instance
        The figure instance the information was plotted to.
    """

    # Read in *all* the tiles
    # This is because anything in the DB could conceivably be observed
    fibre_posns = rFP.execute(cursor)
    logging.info('Found %d target assignments' % len(fibre_posns))
    # Read the field centroid positions
    field_cents = rC.execute(cursor)
    # Convert the field centroids into a dict for easy lookup
    field_cents = {t.field_id: t for t in field_cents}

    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
        fig.set_size_inches((22, 12))

    # Generate a dictionary to hold the stats for each fibre
    stretch = {f: [] for f in BUGPOS_MM.keys()}

    i = 0
    for row in fibre_posns:
        # Calculate the home position
        home_pos_ra, home_pos_dec = field_cents[
            row['field_id']].compute_fibre_posn(row['bug_id'])
        dist = dist_points(home_pos_ra, home_pos_dec, row['ra'], row['dec'])
        stretch[row['bug_id']].append(dist)
        i += 1
        if (i % 10000) == 0:
            logging.info('Completed calcs for %6d / %8d assignments' %
                         (i, len(fibre_posns), ))

    ax2 = fig.add_subplot(111)
    ax = ax2.twinx()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    # ax.set_zorder(1)
    ax.set_title('Fibre travel distances')
    ax.set_xlabel('Fibre no.')
    ax.set_ylabel('Dist (")')

    ax.boxplot([stretch[f] for f in sorted(BUGPOS_MM.keys())],
               positions=[f for f in sorted(BUGPOS_MM.keys())],
               whis=[2., 98.], sym='b.')

    # Show where the patrol radius limit is
    ax.plot(ax.get_xlim(), [PATROL_RADIUS]*2,
            'r--', lw=3)

    # Rotate the x-tick labels
    ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation=90.)

    ax2.set_ylabel('No. of fibre assigns')
    # ax2.set_zorder(10)
    # ax2.set_ylim(map(lambda x: x / ARCSEC_PER_MM, ax.get_ylim()))
    # ax2.set_ylabel('Dist (mm)')

    # Instead of converting to mm, let's plot the number of assignments per
    # fibre in the background
    ax2.bar([f-0.5 for f in sorted(BUGPOS_MM.keys())],
            [np.count_nonzero(fibre_posns['bug_id'] == f)
             for f in sorted(BUGPOS_MM.keys())],
            bottom=0, width=1., color='lightgrey',
            linewidth=0)

    # ax.set_zorder(ax2.get_zorder() + 1)
    # ax.patch.set_visible(False)

    plt.tight_layout()

    if pylab_mode:
        plt.show()
        plt.draw()
    else:
        fig.savefig('%s/%s.%s' % (
            output_loc,
            output_prefix,
            output_fmt
        ), fmt=output_fmt, dpi=300)

    return fig


def plot_hours_better(cursor,
                      pylab_mode=False,
                      output_loc='.',
                      output_prefix='hours_obs',
                      output_fmt='png',
                      ):
    """
    Plot the values of hours_better (i.e. observability) used during the
    simulation

    Parameters
    ----------
    cursor
    pylab_mode
    output_loc
    output_prefix
    output_fmt

    Returns
    -------

    """
    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
        fig.set_size_inches((9, 6))


    # Read in the observing data
    tile_obs_log = rTOI.execute(cursor)
    tile_obs_log.sort(order='date_obs')
    lowz_fields = rCBTexec(cursor, 'is_lowz_target', unobserved=False)

    hours_used = []
    airmass_at = []

    # Compute the data to be plotted
    for row in tile_obs_log:
        if row['field_id'] in lowz_fields:
            hb = rAS.hours_observable(cursor, row['field_id'], row['date_obs'],
                                      datetime_to=max(
                                          datetime.datetime(2018, 4, 1, 2, 0),
                                          row['date_obs'] +
                                          datetime.timedelta(30)
                                      ), hours_better=True)
        else:
            hb = rAS.hours_observable(cursor, row['field_id'], row['date_obs'],
                                      datetime_to=max(
                                          datetime.datetime(2018, 4, 1, 2, 0),
                                          row['date_obs'] +
                                          datetime.timedelta(30)
                                      ), hours_better=True)
        am = rAS.get_airmass(cursor, row['field_id'], row['date_obs'])[0][1]
        hours_used.append((row['date_obs'], row['dec'], hb,
                           am,
                           ))

    hours_used = np.array(hours_used,
                          dtype={
                              'names': ['date_obs', 'dec', 'hb',
                                        'airmass'],
                              'formats': [datetime.datetime, 'f8',
                                          'float', 'float']
                          })

    # Make the hours_better plot
    ax1 = fig.add_subplot(211)
    ax1.scatter(hours_used['date_obs'], hours_used['hb'])

    ax2 = fig.add_subplot(212)
    ax2.scatter(hours_used['dec'], hours_used['hb'])

    plt.tight_layout()

    if pylab_mode:
        plt.show()
        plt.draw()
    else:
        fig.savefig('%s/%s.%s' % (
            output_loc,
            output_prefix,
            output_fmt
        ), fmt=output_fmt, dpi=300)

    return fig


def plot_airmass(cursor,
                 pylab_mode=False,
                 output_loc='.',
                 output_prefix='hours_obs',
                 output_fmt='png',
                      ):
    """
    Plot the values of hours_better (i.e. observability) used during the
    simulation

    Parameters
    ----------
    cursor
    pylab_mode
    output_loc
    output_prefix
    output_fmt

    Returns
    -------

    """
    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        plt.clf()
        fig = plt.gcf()
    else:
        fig = plt.figure()
        fig.set_size_inches((9, 6))


    # Read in the observing data
    tile_obs_log = rTOI.execute(cursor)
    tile_obs_log.sort(order='date_obs')
    lowz_fields = rCBTexec(cursor, 'is_lowz_target', unobserved=False)

    hours_used = []
    airmass_at = []

    # Compute the data to be plotted
    for row in tile_obs_log:
        if row['field_id'] in lowz_fields:
            hb = rAS.hours_observable(cursor, row['field_id'], row['date_obs'],
                                      datetime_to=max(
                                          datetime.datetime(2018, 4, 1, 2, 0),
                                          row['date_obs'] +
                                          datetime.timedelta(30)
                                      ), hours_better=True)
        else:
            hb = rAS.hours_observable(cursor, row['field_id'], row['date_obs'],
                                      datetime_to=max(
                                          datetime.datetime(2018, 4, 1, 2, 0),
                                          row['date_obs'] +
                                          datetime.timedelta(30)
                                      ), hours_better=True)
        am = rAS.get_airmass(cursor, row['field_id'], row['date_obs'])[0][1]
        hours_used.append((row['date_obs'], row['dec'], hb,
                           am,
                           ))

    hours_used = np.array(hours_used,
                          dtype={
                              'names': ['date_obs', 'dec', 'hb',
                                        'airmass'],
                              'formats': [datetime.datetime, 'f8',
                                          'float', 'float']
                          })

    # Make the hours_better plot
    ax1 = fig.add_subplot(211)
    ax1.scatter(hours_used['date_obs'], hours_used['airmass'])

    ax2 = fig.add_subplot(212)
    ax2.scatter(hours_used['dec'], hours_used['airmass'])

    plt.tight_layout()

    if pylab_mode:
        plt.show()
        plt.draw()
    else:
        fig.savefig('%s/%s.%s' % (
            output_loc,
            output_prefix,
            output_fmt
        ), fmt=output_fmt, dpi=300)

    return fig


def plot_hours_observable(cursor, datetime_now, datetime_end,
                          hours_better=True, resolution=15.,
                          pylab_mode=False,
                          output_loc='.',
                          output_prefix='hours_obs',
                          output_fmt='png',
                          ):
    """
    Plot the value of hours_observable at all positions on the sky for a
    given datetime.
    Parameters
    ----------
    cursor
    datetime_now, datetime_end
    hours_better
    resolution
    pylab_mode
    output_loc
    output_prefix
    output_fmt

    Returns
    -------

    """
    # Prepare the plot
    # Prepare the Figure instance
    if pylab_mode:
        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches((9, 6))
    else:
        fig = plt.figure()
        fig.set_size_inches((9, 6))

    mapplot = AllSkyMap(projection='moll', lon_0=0.)
    limb = mapplot.drawmapboundary(fill_color='white')
    mapplot.drawparallels(np.arange(-75, 76, 15), linewidth=0.5, dashes=[1, 2],
                          labels=[1, 0, 0, 0], fontsize=9)
    mapplot.drawmeridians(np.arange(0, 331, 30), linewidth=0.5, dashes=[1, 2])

    # Plot the KiDS region
    kids_region = [
        [329.5, 53.5, 53.5, 329.5, 329.5],
        [-35.6, -35.6, -25.7, -25.7, -35.6]
    ]

    # for i in range(len(kids_region[0]) - 1):
    #     mapplot.geodesic(kids_region[0][i], kids_region[1][i],
    #                      kids_region[0][i + 1], kids_region[1][i + 1],
    #                      color='limegreen', lw=1.2, ls='--', zorder=20)

    # Read in the field information
    fields = rC.execute(cursor)
    lowz_fields = rCBTexec(cursor, 'is_lowz_target',
                           unobserved=False)
    hours_better_comp = []
    for field in [f.field_id for f in fields]:
        if field in lowz_fields:
            hours_better_comp.append(rAS.hours_observable(
                cursor, field, datetime_now,
                max(datetime_end, datetime_now + datetime.timedelta(30.)),
                exclude_grey_time=True, exclude_dark_time=False,
                minimum_airmass=2.0,
                hours_better=hours_better, resolution=resolution,
            ))
        else:
            hours_better_comp.append(rAS.hours_observable(
                cursor, field, datetime_now,
                datetime_now + datetime.timedelta(365.),
                exclude_grey_time=True, exclude_dark_time=False,
                minimum_airmass=2.0,
                hours_better=hours_better, resolution=resolution,
            ))

    step = 21
    i = 0
    err = 0
    while (i*step) < len(fields):
        try:
            dots = mapplot.scatter(
                # [90.]*3, [-45.]*3,
                # fields_ra[30:33], fields_dec[30:33],
                [f.ra for f in fields][i*step:(i+1)*step], [f.dec for f in fields][i*step:(i+1)*step],
                c=hours_better_comp[i*step:(i+1)*step],
                s=50, marker='o',
                edgecolors='none',
                # facecolors=
                # 'red',
                # cm.jet,
                # 'red',
                # [hours_obs[k] for k in range(k, k + step)
                #  if k in hours_obs.keys()],
                cmap=cm.jet,
                vmin=0.0, vmax=1200.,
                latlon=True,
                zorder=8,
            )
            i += 1
        except ValueError:
            i += 1
            err += 1

    print('Points lost to basemap being shit: %d' % (err * step))

    plt.tight_layout()

    if pylab_mode:
        plt.show()
        plt.draw()
    else:
        fig.savefig('%s/%s_%s.%s' % (
            output_loc,
            output_prefix,
            datetime_now.strftime('%Y-%m-%d-%H-%M'),
            output_fmt
        ), fmt=output_fmt, dpi=300)

    mapplot.colorbar(dots)
    fig.suptitle(datetime_now.strftime('%Y-%m-%d %H:%M UTC'))

    return fig


def plot_tile_information(cursor, datetime_from=None, datetime_to=None,
                          pylab_mode=False,
                          output_loc='.',
                          output_prefix='tile_stats',
                          output_fmt='png',
                          extent=10.
                          ):
    """
    Make a plot with information about a tile observation

    Parameters
    ----------
    cursor : psycopg2.cursor object
        Cursor for communication with the database.
    datetime_from, datetime_to: datetime.datetime, optional
        Bounding datetimes for the plotting to consider. Both default to None,
        which will make a plot for each observed tile.
    pylab_mode : Boolean, optional
        Whether or not to output the plot to a Pylab plotting window (True) or
        not (False). Defaults to False.
    output_loc, output_prefix, output_fmt: strings
        Strings denoting the location to write output (relative or absolute,
        defaults to '.'),
        the prefix of the output files (defaults to 'hrs_bet'), and the format
        of the output (defaults to 'png').

    Returns
    -------
    fig : matplotlib.Figure instance
        The Figure instance used for plotting.
    """

    priorities = range(0, 2) + range(3, 11)

    extent_m = np.radians(10.) * EARTH_RADIUS
    extent_m_tile = np.radians(0.5+(TILE_RADIUS/3600.)) * EARTH_RADIUS

    # Read in the tile observation data
    obs_tile_info = rTOI.execute(cursor)
    obs_tile_info.sort(order='date_obs')
    # Read in the target observation data
    obs_log = rOLexec(cursor)
    obs_log.sort(order='date_obs')
    target_types = rScTexec(cursor)
    target_types.sort(order='target_id')
    target_started, target_complete = rScDexec(cursor)
    target_started.sort(order='target_id')
    target_complete.sort(order='target_id')
    sci_pos = rScPexec(cursor)
    sci_pos.sort(order='field_id')

    fields = rC.execute(cursor)
    fields_list = [(field.field_id, field.ra, field.dec) for field in fields]
    fields = np.asarray(fields_list, dtype={
        'names': ['field_id', 'ra', 'dec'],
        'formats': ['i8', 'f8', 'f8']
    })

    # Set up the Figure instance
    if pylab_mode:
        fig = plt.gcf()
        fig.clf()
        # fig.set_size_inches((14, 10.5))
    else:
        fig = plt.figure()
        fig.set_size_inches((12.5, 9))

    if datetime_from is None:
        datetime_from = np.min(obs_tile_info['date_obs'])
    if datetime_to is None:
        datetime_to = np.max(obs_tile_info['date_obs'])

    # We now need to initialize the data arrays
    # Basically, some of the data we wish to plot are cumulative across
    # the simulator run, so we need to compute those data for any tiles
    # we're not explicitly plotting (i.e. that were observed before
    # datetime_from)
    field_visits = {f: 0 for f in fields['field_id']}
    fiber_alloc = []
    fiber_alloc_priority = []
    completeness = {priority: [] for priority in priorities}
    for tile in obs_tile_info[obs_tile_info['date_obs'] < datetime_from]:
        field_visits[tile['field_id']] += 1
        fiber_alloc.append(np.count_nonzero(
            obs_log[obs_log['tile_pk'] == tile['tile_pk']]
        ))
        fiber_alloc_priority.append(
            np.count_nonzero(np.logical_and(
                obs_log['tile_pk'] == tile['tile_pk'],
                np.logical_or(
                    obs_log['is_h0_target'],
                    np.logical_or(
                        obs_log['is_vpec_target'],
                        obs_log['is_lowz_target']
                    )))
            ))
        for priority in priorities:
            completeness[priority].append(
                np.count_nonzero(np.in1d(target_types[
                            target_types['priority'] == priority
                        ]['target_id'],
                        target_complete[target_complete['date_obs'] <=
                                        tile['date_obs']]['target_id']))
                / np.count_nonzero(target_types['priority'] == priority)
            )

    datetime_curr = datetime_from

    while datetime_curr <= datetime_to:
        # Find the first tile observed at/after datetime_curr
        try:
            tile_curr = obs_tile_info[obs_tile_info['date_obs'] >=
                                      datetime_curr][0]
            print(tile_curr['date_obs'])
            datetime_curr = tile_curr['date_obs']
            field_visits[tile_curr['field_id']] += 1
            fiber_alloc.append(np.count_nonzero(
                obs_log[obs_log['tile_pk'] == tile_curr['tile_pk']]
            ))
            fiber_alloc_priority.append(
                np.count_nonzero(np.logical_and(
                    obs_log['tile_pk'] == tile_curr['tile_pk'],
                    np.logical_or(
                        obs_log['is_h0_target'],
                        np.logical_or(
                            obs_log['is_vpec_target'],
                            obs_log['is_lowz_target']
                        )))
                ))
        except IndexError:
            # We're out of tiles - end
            datetime_curr = datetime_to + datetime.timedelta(1.)
            continue

        # Axis 1 - completeness in the region of the tile
        # Start up the basemap
        ax1 = plt.subplot2grid((2,3), (0,0))
        ax1.set_title('Target distrib.')
        m1 = Basemap(
            # llcrnrlon=tile_curr['ra'] - (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  - extent))),
            # llcrnrlat=tile_curr['dec'] - extent,
            # urcrnrlon=tile_curr['ra'] + (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  + extent))),
            # urcrnrlat=tile_curr['dec'] + extent,
            projection='cass',
            # celestial=False,
            # projection='lcc',
            lon_0=tile_curr['ra'], lat_0=tile_curr['dec'],
            width=2*extent_m, height=2*extent_m
        )
        m1.drawmeridians(np.arange(0, 360., 5.),
                         labels=[0,0,0,1] if tile_curr['dec'] > -70. else [0,1,0,0],
                         labelstyle='+/-')
        m1.drawparallels(np.arange(-90., 90., 3.), labels=[1,0,0,0],
                         labelstyle='+/-')

        incomplete = ~np.in1d(target_types['target_id'],
                              target_complete[target_complete['date_obs'] <=
                                              datetime_curr]['target_id']
                              )
        # Completed targets
        m1.scatter(
            target_types[~incomplete]['ra'], target_types[~incomplete]['dec'],
            latlon=True, s=2.5, facecolor='black', edgecolor='none',
        )
        # Filler targets
        m1.scatter(
            target_types[
                np.logical_and(incomplete,
                               np.logical_and(
                                   ~target_types['is_h0_target'],
                                   np.logical_and(
                                       ~target_types['is_vpec_target'],
                                       ~target_types['is_lowz_target']
                                   )
                               ))
            ]['ra'], target_types[
                np.logical_and(incomplete,
                               np.logical_and(
                                   ~target_types['is_h0_target'],
                                   np.logical_and(
                                       ~target_types['is_vpec_target'],
                                       ~target_types['is_lowz_target']
                                   )
                               ))
            ]['dec'],
            latlon=True, s=3, facecolor='grey', edgecolor='none',
        )
        # h0 targets
        m1.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_h0_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_h0_target']
            )]['dec'],
            latlon=True, s=3, facecolor='green', edgecolor='none',
        )
        # vpec targets
        m1.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['dec'],
            latlon=True, s=3, facecolor='red', edgecolor='none',
        )
        # lowz targets
        m1.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_lowz_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_lowz_target']
            )]['dec'],
            latlon=True, s=3, facecolor='purple', edgecolor='none',
        )

        # This tile's targets
        m1.scatter(
            obs_log[obs_log['date_obs'] == tile_curr['date_obs']]['ra'],
            obs_log[obs_log['date_obs'] == tile_curr['date_obs']]['dec'],
            latlon=True, s=8, facecolor='yellow', edgecolor='none',
        )
        # Tile boundary
        m1.tissot(tile_curr['ra'], tile_curr['dec'], TILE_RADIUS/3600., 20,
                  facecolor='none', edgecolor='blue', lw=2)
        # Tile centres
        m1.scatter(
            fields['ra'], fields['dec'], latlon=True, s=30, marker='s',
            edgecolor='black', facecolor='white', linewidths=2,
        )

        # Axis 2 - No. visits in the area
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        ax2.set_title('No. visits')
        m2 = Basemap(
            # llcrnrlon=tile_curr['ra'] - (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  - extent))),
            # llcrnrlat=tile_curr['dec'] - extent,
            # urcrnrlon=tile_curr['ra'] + (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  + extent))),
            # urcrnrlat=tile_curr['dec'] + extent,
            projection='cass',
            # celestial=False,
            # projection='lcc',
            lon_0=tile_curr['ra'], lat_0=tile_curr['dec'],
            width=2 * extent_m, height=2 * extent_m
        )
        m2.drawmeridians(np.arange(0, 360., 5.),
                         labels=[0, 0, 0, 1] if tile_curr['dec']>-70 else [0,1,0,0],
                         labelstyle='+/-')
        m2.drawparallels(np.arange(-90., 90., 3.), labels=[1, 0, 0, 0],
                         labelstyle='+/-')

        for tile in obs_tile_info[obs_tile_info['date_obs'] <= datetime_curr]:
            m2.tissot(tile['ra'], tile['dec'], TILE_RADIUS/3600., 20,
                      facecolor='red', edgecolor='none', alpha=0.05,
                      # celestial=True,
                      )

        m2.scatter(
            fields['ra'], fields['dec'], latlon=True, s=30, marker='s',
            edgecolor='black', facecolor='white', linewidths=2,
        )

        # Axis 3 - tile target distribution (close-up)
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        ax3.set_title('Tile setup')
        m3 = Basemap(
            # llcrnrlon=tile_curr['ra'] - (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  - extent))),
            # llcrnrlat=tile_curr['dec'] - extent,
            # urcrnrlon=tile_curr['ra'] + (extent /
            #                              np.cos(np.radians(
            #                                  tile_curr['dec']
            #                                  + extent))),
            # urcrnrlat=tile_curr['dec'] + extent,
            projection='cass',
            # celestial=False,
            # projection='lcc',
            lon_0=tile_curr['ra'], lat_0=tile_curr['dec'],
            width=2 * extent_m_tile,
            height=2 * extent_m_tile
        )
        m3.drawmeridians(np.arange(0, 360., 2.),
                         labels=[0, 0, 0, 1] if tile_curr['dec'] > -70 else [0,
                                                                             1,
                                                                             0,
                                                                             0],
                         labelstyle='+/-')
        m3.drawparallels(np.arange(-90., 90., 2.), labels=[1, 0, 0, 0],
                         labelstyle='+/-')

        # Completed targets
        m3.scatter(
            target_types[~incomplete]['ra'], target_types[~incomplete]['dec'],
            latlon=True, s=2.5, facecolor='black', edgecolor='none',
        )
        # Filler targets
        m3.scatter(
            target_types[
                np.logical_and(incomplete,
                               np.logical_and(
                                   ~target_types['is_h0_target'],
                                   np.logical_and(
                                       ~target_types['is_vpec_target'],
                                       ~target_types['is_lowz_target']
                                   )
                               ))
            ]['ra'], target_types[
                np.logical_and(incomplete,
                               np.logical_and(
                                   ~target_types['is_h0_target'],
                                   np.logical_and(
                                       ~target_types['is_vpec_target'],
                                       ~target_types['is_lowz_target']
                                   )
                               ))
            ]['dec'],
            latlon=True, s=3, facecolor='grey', edgecolor='none',
        )
        # h0 targets
        m3.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_h0_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_h0_target']
            )]['dec'],
            latlon=True, s=3, facecolor='green', edgecolor='none',
        )
        # vpec targets
        m3.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['dec'],
            latlon=True, s=3, facecolor='red', edgecolor='none',
        )
        # lowz targets
        m3.scatter(
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['ra'],
            target_types[np.logical_and(
                incomplete,
                target_types['is_vpec_target']
            )]['dec'],
            latlon=True, s=3, facecolor='purple', edgecolor='none',
        )

        m3.tissot(tile_curr['ra'], tile_curr['dec'], TILE_RADIUS / 3600., 50,
                  facecolor='none', edgecolor='blue', lw=2)
        # Any target which has a priority *higher* than the lowest-priority
        # target actually observed on the tile
        m3.scatter(
            target_types[np.logical_and(
                np.logical_and(
                    incomplete,
                    np.in1d(target_types['target_id'],
                            sci_pos[sci_pos['field_id'] ==
                                    tile_curr['field_id']]['target_id'])
                ),
                target_types['priority'] == max(
                    obs_log[obs_log['date_obs'] == datetime_curr]['priority']
                )
            )]['ra'],
            target_types[np.logical_and(
                np.logical_and(
                    incomplete,
                    np.in1d(target_types['target_id'],
                            sci_pos[sci_pos['field_id'] ==
                                    tile_curr['field_id']]['target_id'])
                ),
                target_types['priority'] == max(
                    obs_log[obs_log['date_obs'] == datetime_curr]['priority']
                )
            )]['dec'],
            latlon=True, s=14, facecolor='orange', edgecolor='orange',
            marker='x',
        )

        # This tile's targets
        # filler
        m3.scatter(
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.logical_and(
                                   ~np.in1d(obs_log['target_id'],
                                            target_types[target_types[
                                                'is_vpec_target']]['target_id']),
                                   np.logical_and(
                                       ~np.in1d(obs_log['target_id'],
                                                target_types[target_types[
                                                    'is_h0_target']][
                                                    'target_id']),
                                       ~np.in1d(obs_log['target_id'],
                                                target_types[target_types[
                                                    'is_lowz_target']][
                                                    'target_id'])))
                               )
            ]['ra'],
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.logical_and(
                                   ~np.in1d(obs_log['target_id'],
                                            target_types[target_types[
                                                'is_vpec_target']][
                                                'target_id']),
                                   np.logical_and(
                                       ~np.in1d(obs_log['target_id'],
                                                target_types[target_types[
                                                    'is_h0_target']][
                                                    'target_id']),
                                       ~np.in1d(obs_log['target_id'],
                                                target_types[target_types[
                                                    'is_lowz_target']][
                                                    'target_id'])))
                               )
            ]['dec'],
            latlon=True, s=20, facecolor='grey', edgecolor='black',
            marker='D',
        )
        # h0
        m3.scatter(
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_h0_target']]['target_id'])
                               )
            ]['ra'],
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_h0_target']]['target_id'])
                               )
            ]['dec'],
            latlon=True, s=20, facecolor='green', edgecolor='black',
            marker='D',
        )
        # vpec
        m3.scatter(
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_vpec_target']]['target_id'])
                               )
            ]['ra'],
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_vpec_target']]['target_id'])
                               )
            ]['dec'],
            latlon=True, s=20, facecolor='red', edgecolor='black',
            marker='D',
        )
        # lowz
        m3.scatter(
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_lowz_target']]['target_id'])
                               )
            ]['ra'],
            obs_log[
                np.logical_and(obs_log['date_obs'] == tile_curr['date_obs'],
                               np.in1d(obs_log['target_id'],
                                       target_types[target_types[
                                           'is_lowz_target']]['target_id'])
                               )
            ]['dec'],
            latlon=True, s=20, facecolor='purple', edgecolor='black',
            marker='D',
        )

        # Axis 4 - No. of fibre allocations
        ax4 = plt.subplot2grid((2,5), (1,0))
        ax4.bar(
            np.asarray(range(len(fiber_alloc))[-100:]) - 0.5,
            fiber_alloc[-100:],
            width=1., edgecolor='none', facecolor='blue', label='all tgts'
        )
        ax4.bar(
            np.asarray(range(len(fiber_alloc_priority))[-100:]) - 0.5,
            fiber_alloc_priority[-100:],
            width=1., edgecolor='none', facecolor='red', label='w/ type'
        )
        ax4.set_xlim((
            range(len(fiber_alloc))[-100:][0] - 0.5,
            range(len(fiber_alloc))[-100:][0] + 101,
        ))
        ax4.set_ylim((0, 140))
        ax4.set_title('Fibre alloc')
        ax4.set_xlabel('Observation no.')
        ax4.set_ylabel('No. fibres assigned')
        ax4.legend(loc='lower left')

        # Axis 5 - Survey progress (takes up two standard axis slots)
        ax5 = plt.subplot2grid((2,5), (1,3), colspan=2)
        ax5.set_title('Progress')
        ax5.set_ylabel('Fraction targets done')
        ax5.set_xlabel('Observation no.')
        ax5.set_ylim((0.0, 1.05))
        ax5.set_xlim(0, len(obs_tile_info))

        for priority in priorities:
            completeness[priority].append(
                np.count_nonzero(np.in1d(target_types[
                            target_types['priority'] == priority
                        ]['target_id'],
                        target_complete[target_complete['date_obs'] <=
                                        tile['date_obs']]['target_id']))
                / np.count_nonzero(target_types['priority'] == priority)
            )
            ax5.plot(
                range(len(completeness[priority])), completeness[priority],
                label='%2d (%.1f%%)' % (priority,
                                        completeness[priority][-1]*100.),
            )
        ax5.legend(loc='upper left')

        # Axis 6 - priority distribution on tile
        ax6 = plt.subplot2grid((2, 5), (1, 1))
        ax6.set_xlabel('Priority')
        ax6.set_xlim((-0.5, 10.5))
        ax6.set_ylabel('No. targets')
        ax6.set_title('Assigned prior.')
        ax6.bar(
            np.asarray(priorities) - 0.5,
            [np.count_nonzero(np.logical_and(
                obs_log['date_obs'] == datetime_curr,
                obs_log['priority'] == priority
            )) for priority in priorities],
            width=1., edgecolor='none', facecolor='black',
        )

        # Axis 7 - remaining priorities on tile
        ax7 = plt.subplot2grid((2, 5), (1, 2))
        ax7.set_xlabel('Priority')
        ax7.set_xlim((-0.5, 10.5))
        ax7.set_ylabel('No. targets')
        ax7.set_title('Remaining prior.')
        ax7.set_yscale('log')
        ax7.bar(
            np.asarray(priorities) - 0.5,
            [np.count_nonzero(np.logical_and(
                target_types[incomplete]['priority'] == priority,
                np.in1d(target_types[incomplete]['target_id'],
                        sci_pos[sci_pos['field_id'] ==
                                tile_curr['field_id']]['target_id'])
            )) for priority in priorities],
            width=1., edgecolor='none', facecolor='black',
        )

        # Final things
        fig.suptitle('%s - Tile %s - Field %s - RA %3.1f - Dec %2.1f' %
                     (datetime_curr.strftime('%Y-%m-%d %H:%M:%S'),
                      tile_curr['tile_pk'],
                      tile_curr['field_id'],
                      tile_curr['ra'],
                      tile_curr['dec'], ))

        if pylab_mode:
            plt.draw()
            plt.show()
        else:
            fig.savefig('%s/%s_%s.%s' % (
                output_loc,
                output_prefix,
                datetime_curr.strftime('%Y-%m-%d-%H-%M'),
                output_fmt
            ), fmt=output_fmt, dpi=300)

        try:
            datetime_curr = obs_tile_info[
                obs_tile_info['date_obs'] > datetime_curr]['date_obs'][0]
        except IndexError:
            datetime_curr += datetime.timedelta(1.)

    return fig


def plot_total_visits(cursor, datetime_to=None,
                      pylab_mode=False,
                      output_loc='.',
                      output_prefix='total_visits',
                      output_fmt='png'):
    """
    Plot the total number of visits to each pointing up to a certain time.

    Parameters
    ----------
    cursor : psycopg2.cursor object
        Cursor for communicating with the database.
    datetime_to : datetime.datetime instance, optional
        UTC datetime to consider til. Defaults to None, at which point all
        dates are considered.
    pylab_mode : Boolean, optional
        Whether or not to output the plot to a Pylab plotting window (True) or
        not (False). Defaults to False.
    output_loc, output_prefix, output_fmt: strings
        Strings denoting the location to write output (relative or absolute,
        defaults to '.'),
        the prefix of the output files (defaults to 'hrs_bet'), and the format
        of the output (defaults to 'png').

    Returns
    -------
    fig : matplotlib.Figure instance
        The Figure instance the plot was made in
    """

    # Set up the Figure instance
    if pylab_mode:
        fig = plt.gcf()
        fig.clf()
        # fig.set_size_inches((14, 10.5))
    else:
        fig = plt.figure()
        fig.set_size_inches((12.5, 9))

    # Read in the tile observation data
    obs_tile_info = rTOI.execute(cursor)
    obs_tile_info.sort(order='date_obs')

    if datetime_to is None:
        datetime_to = np.max(obs_tile_info['date_obs'])

    ax = fig.add_subplot(111)
    m = Basemap(
        lon_0=180., projection='moll',
    )
    m.drawmeridians(np.arange(30., 360., 30.),
                    # labels=[1, 0, 0, 0]
                    )
    m.drawparallels(np.arange(-75., 90., 15.),
                    labels=[1, 0, 0, 0])

    # jet = colors.Colormap('jet')
    # cNorm = colors.Normalize(vmin=0, vmax=20)
    # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    x = np.arange(0.0, 360.05, 2.)
    y = np.arange(-90., 20.05, 1.)
    X, Y = np.meshgrid(x, y)
    coverage_data = np.zeros(X.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            coverage_data[j, i] = np.count_nonzero(
                [dist_points(x[i]+0.5, y[j]+0.5, t['ra'], t['dec']) <
                 TILE_RADIUS for
                 t in obs_tile_info[:-1]]
            )

    pcol = m.pcolor(X, Y, coverage_data, latlon=True, cmap=cm.jet,
                    vmin=0., )
    m.colorbar(pcol)

    for field in set(obs_tile_info['field_id']):
        pass
        # this_field_obs = obs_tile_info[obs_tile_info['field_id'] == field]
        # m.tissot(
        #     this_field_obs['ra'][0], this_field_obs['dec'][0],
        #     TILE_RADIUS/3600., 20,
        #     # facecolor=scalarMap.to_rgba(len(this_field_obs))
        # )

    if pylab_mode:
        plt.draw()
        plt.show()
    else:
        fig.savefig('%s/%s.%s' % (
            output_loc,
            output_prefix,
            output_fmt
        ), fmt=output_fmt, dpi=300)

    return fig
