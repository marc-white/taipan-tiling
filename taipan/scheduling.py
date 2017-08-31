"""Taipan scheduling functionality.

This module handles the details of scheduling for the Taipan simulator/tiling
package, including the computation of visibilities, time conversions, sun/moon
positions, etc.
"""

import numpy as np
import datetime
import pytz
from collections import OrderedDict

import os
import sys
import time
import logging
import pickle
import copy

import ephem

# ______________________________________________________________________________
#
# CONSTANTS

# define twilight values for when observing can start/must end

SOLAR_TWILIGHT_HORIZON = np.radians(-10.)      # radians everywhere
""":obj:`float`, radians: Sun distance below the horizon at solar twilight"""
LUNAR_TWILIGHT_HORIZON = np.radians(0.)        # radians everywhere
""":obj:`float`, radians: Moon distance below the horizon at lunar twilight"""

SOLAR_HORIZON = SOLAR_TWILIGHT_HORIZON - np.radians(0.24)
""":obj:`float`, radians: Sun distance below the horizon at dark time"""
LUNAR_HORIZON = LUNAR_TWILIGHT_HORIZON - np.radians(0.27)
""":obj:`float`, radians: Moon distance below the horizon at dark time"""

SUN = ephem.Sun()
""":any:`ephem.Sun`: :any:`ephem` Sun object"""
MOON = ephem.Moon()
""":any:`ephem.Sun`: :any:`ephem` Moon object"""

# define and establish emph.Observer object for UKST-Taipan
#  all from https://en.wikipedia.org/wiki/Siding_Spring_Observatory
UKST_LATITUDE = -31.272231
""":obj:`float`: Latitude of the UK Schmidt Telescope"""
UKST_LONGITUDE = +149.071233
""":obj:`float`: Longitude of the UK Schmidt Telescope"""
UKST_ELEVATION = 1165  # metres
""":obj:`float`: Elevation of the UK Schmidt Telescope"""
UKST_TIMEZONE = pytz.timezone('Australia/Sydney')
""":any:`pytz.timezone`: Local timezone of the UK Schmidt Telescope"""

# Create a standard ephem Observer object for UKST
UKST_TELESCOPE = ephem.Observer()
""":any:`ephem.Observer`: :any:`ephem` Observer object representing UKST"""
UKST_TELESCOPE.lat = np.radians(UKST_LATITUDE)   # radians everywhere
UKST_TELESCOPE.lon = np.radians(UKST_LONGITUDE)  # radians everywhere
UKST_TELESCOPE.elevation = UKST_ELEVATION
UKST_TELESCOPE.temp = 10.

SECONDS_PER_DAY = 86400.
""":obj:`float`: Number of seconds per day"""

ALMANAC_RESOLUTION_MAX = 60. * 4.
""":obj:`float`: Maximum allowed resolution of :any:`Almanac` objects"""

# Constant for converting from ephem times to MJD
EPHEM_TO_MJD = 15019.5
""":obj:`float`: Conversion factor between :any:`ephem` and standard MJD 
times"""
EPHEM_DT_STRFMT = '%Y/%m/%d %H:%M:%S'
""":obj:`str`: Standard formatting string for :any:`ephem` datetimes"""

# Observing constants
SLEW_TIME = (5. * 60.) + (0.8 * 60)  # seconds, configure + calibrations
""":obj:`float`, seconds: Slew time of the UKST"""
OBS_TIME = (15. * 60.) + (1. * 60.)  # seconds, obs + readout
""":obj:`float`, seconds: Length of a Taipan observation, including readout"""
POINTING_TIME = (SLEW_TIME + OBS_TIME) / SECONDS_PER_DAY  # days
""":obj:`float`, days: Time for a total observation (slew + observe), in *days* 
"""


# ______________________________________________________________________________
# HELPER FUNCTIONS
# ______________________________________________________________________________

def get_utc_datetime(dt, tz=UKST_TIMEZONE):
    """
    Compute a UTC datetime from a naive datetime (dt) in a given timezone (tz)

    Parameters
    ----------
    dt: :any:`datetime.datetime`
        Naive datetime to be converted to UTC.
    tz: :any:`pytz.timezone`
        Timezone that the naive datetime is in. Defaults to
        :any:`UKST_TIMEZONE`.

    Returns
    -------
    dt_utc: :any:`datetime.datetime`
        A timezone-aware datetime, with tz=pytz.utc.
    """

    dt_utc = tz.localize(dt).astimezone(pytz.utc)
    return dt_utc


def get_ephem_set_rise(date, observer=UKST_TELESCOPE):
    """
    Determine the sunrise and sunset for the given date.

    Parameters
    ----------
    date: :any:`datetime.date`
        The date of the observing night. Note this is the date the night starts
        on.
    observer: :any:`ephem.Observer`
        An ephem.Observer instance holding information on the observing
        location. Defaults to :any:`UKST_TELESCOPE`.

    Returns
    -------
    sunrise, sunset: float
        The time of sunset on the given night, and the following sunrise. Note
        that the values returned are in the date standard of the ephem module -
        to convert to MJD, add :any:`EPHEM_TO_MJD`.
    """
    # Set the observer date to midday before observing starts
    observer.date = get_utc_datetime(datetime.datetime.combine(
        date, datetime.time(12, 0, 0)))
    sunset = next_sunset(observer)
    # Move the date forward to the sunset time, and ask for the next sunrise
    observer.date = sunset
    sunrise = next_sunrise(observer)
    return sunset, sunrise


def ephem_to_dt(ephem_dt, fmt=EPHEM_DT_STRFMT):
    """
    Convert an :any:`ephem` dt to a Python :any:`datetime.datetime` object.
    Note that the output datetime.datetime will be in UTC, NOT local.

    Parameters
    ----------
    ephem_dt: float
        The pyephem dt to convert.
    fmt:
        Deprecated.

    Returns
    -------
    dt: :any:`datetime.datetime`
        A Python :any:`datetime.datetime` instance. It will be timezone-naive,
        but the value will correspond to UTC.
    """
    # dt = datetime.datetime.strptime(str(ephem.Date(ephem_dt)), fmt)
    dt = ephem.Date(ephem_dt).datetime()
    return dt


def localize_utc_dt(dt, tz=UKST_TIMEZONE):
    """
    Convert a naive datetime (which is assumed to be UTC) to a local datetime

    Parameters
    ----------
    dt: :any:`datetime.datetime`
        Input datetime. Should be timezone-naive, but intended to represent
        UTC.
    tz: :any:`pytz.timezone`
        The timezone to push the datetime into.

    Returns
    -------
    dt_local: :any:`datetime.datetime`
        A local datetime. Note that dt_local will still be timezone-naive.
    """
    dt_local = pytz.utc.localize(dt).astimezone(tz).replace(tzinfo=None)
    return dt_local


def utc_local_dt(dt, tz=UKST_TIMEZONE):
    """
    Convert a naive datetime (assumed to be in timezone tz) into a
    naive UTC datetime

    Parameters
    ----------
    dt: :any:`datetime.datetime`
        The datetime to convert.
    tz: :any:`pytz.timezone`
        The 'local' timezone. Defaults to UKST_TIMEZONE.

    Returns
    -------
    dt_utc: :any:`datetime.datetime`
        A naive timezone, but cast into UTC timezone.
    """
    dt_utc = tz.localize(dt).astimezone(pytz.utc).replace(tzinfo=None)
    return dt_utc

# ______________________________________________________________________________
# CLASS DEFINITIONS
# ______________________________________________________________________________


class Almanac(object):
    """
    Object which stores observability information for a specific point on the
    sky.

    At its core, and :any:`Almanac` object is mapping of datetimes to
    sky brightness, sun and moon altitude, and other useful information. It is
    cheaper to pre-compute this information and store it rather than generate
    it on-the-fly (particularly when interfacing with :any:`TaipanDB`).
    """

    # Define class attributes
    _ra = None
    _dec = None
    _field_id = None
    _start_date = None
    _end_date = None
    _data = None
    _resolution = None
    _minimum_airmass = None
    _observer = None
    alm_file_path = None

    # Setters & getters
    @property
    def ra(self):
        """:obj:`float`: Right ascension (RA) for this Almanac

        Raises
        ------
        ValueError
            Raised if passed RA is outside the allowed range :math:`[0,360)`.
        """
        return self._ra

    @ra.setter
    def ra(self, r):
        if r is None:
            self._ra = None
            return
        r = float(r)
        if r < 0. or r >= 360.:
            raise ValueError('Almanac must have 0 <= RA < 360')
        self._ra = r

    @property
    def dec(self):
        """:obj:`float`: Declination (Dec) for this Almanac

        Raises
        ------
        ValueError
            Raised if passed Dec is outside the allowed range :math:`[-90,90]`.
        """
        return self._dec

    @dec.setter
    def dec(self, d):
        if d is None:
            self._dec = None
            return
        d = float(d)
        if d < -90. or d > 90.:
            raise ValueError('Almanac must have -90 <= RA <= 90')
        self._dec = d

    @property
    def field_id(self):
        """:obj:`int`: Field ID this :any:`Almanac` corresponds to"""
        return self._field_id

    @field_id.setter
    def field_id(self, f):
        f = int(f)
        self._field_id = f

    @property
    def start_date(self):
        """:any:`datetime.datetime`: First date in this :class:`Almanac`

        Raises
        ------
        ValueError
            Raised if ``start_date`` is not an instance of
            :any:`datetime.datetime`, or is after ``end_date``.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, d):
        if not isinstance(d, datetime.date):
            raise ValueError("start_date must be an instance of "
                             "datetime.datetime.date")
        if self.end_date is not None:
            if d > self.end_date:
                raise ValueError("Requested start_date is after the existing "
                                 "end_date for this almanac")
        self._start_date = d

    @property
    def end_date(self):
        """:any:`datetime.datetime`: Last date in this :class:`Almanac`

        Raises
        ------
        ValueError
            Raised if ``end_date`` is not an instance of
            :any:`datetime.datetime`, or is before ``start_date``.
        """
        return self._end_date

    @end_date.setter
    def end_date(self, d):
        if not isinstance(d, datetime.date):
            raise ValueError("end_date must be an instance of "
                             "datetime.datetime.date")
        if self.start_date is not None:
            if d < self.start_date:
                raise ValueError("Requested end_date is before the existing "
                                 "start_date for this almanac")
        self._end_date = d

    @property
    def data(self):
        """:any:`numpy.ndarray`: Airmass data

        Should be a two-column :any:`numpy.ndarray`, with columns ``date``
        and ``airmass``.

        Raises
        ------
        ValueError
            Raised if the passed array is not of the correct structure (or
            not an array at all)
        """
        return self._data

    @data.setter
    def data(self, a):
        if not isinstance(a, np.ndarray):
            raise ValueError("data must be a numpy array")
        try:
            _ = a['date']
            _ = a['airmass']
        except:
            raise ValueError("data must be a numpy structured array "
                             "with columns 'date' and 'airmass'")
        self._data = a

    @property
    def minimum_airmass(self):
        """:obj:`float`: minimum airmass considered 'observable'

        Although called `minimum_airmass`, it is really the maximum airmass
        that will be considered observable. The 'minimum' comes from it being
        the minimum altitude above the horizon considered observable.

        Raises
        ------
        ValueError
            Raised if ``minimum_airmass` is outside the valid range (0, 100).
            Note that the maximum practical value of airmass is :math:`~38`;
            values up to 100 are allowed as 99 is used as a special value.
        """
        return self._minimum_airmass

    @minimum_airmass.setter
    def minimum_airmass(self, a):
        if a is None:
            self._minimum_airmass = None
            return
        a = float(a)
        if a < 0:
            raise ValueError('Minimum airmass must be > 0')
        if a > 100:
            raise ValueError('Minimum airmass must be < 100 '
                             '(max. practical value is ~ 38; 99 is used '
                             'as a special value by this module')
        self._minimum_airmass = a

    @property
    def resolution(self):
        """:obj:`float`, mins: :class:`Almanac` resolution

        The resolution parameter dictates how finely-grained the date
        information stored in the :class:`Almanac` is.

        Raises
        ------
        ValueError
            Raise if `resolution` is outside the allowed range
            :math:`(0,` :any:`ALMANAC_RESOLUTION_MAX` :math:`)`.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, r):
        r = float(r)
        if r <= 0:
            raise ValueError('Resolution must be > 0 mins')
        if r > float(ALMANAC_RESOLUTION_MAX):
            raise ValueError('Resolution must be < %d mins' %
                             (ALMANAC_RESOLUTION_MAX, ))
        self._resolution = r

    @property
    def observer(self):
        """:any:`ephem.Observer`: :any:`ephem` Observer for this :any:`Almanac`

        Raises
        ------
        ValueError
            Raised if anything but an :any:`ephem.Observer` is provided
        """
        return self._observer

    @observer.setter
    def observer(self, o):
        if not isinstance(o, ephem.Observer):
            raise ValueError('observer must be an instance of the Observer '
                             'class from pyephem')
        self._observer = o

    # Helper functions
    def compute_end_date(self, observing_period):
        """
        Compute the end date for this :any:`Almanac` and set it.

        Parameters
        ----------
        observing_period : float
            The length of the observing period in days.
        """
        delta = datetime.timedelta(observing_period)
        end_date = self.start_date + delta
        self.end_date = end_date

    def get_observing_period(self):
        """
        Compute and return the observing period for this :any:`Almanac`

        Returns
        -------
        float, days
            The observing period of the :any:`Almanac` in days
        """
        return (self.end_date -
                self.start_date).total_seconds() / SECONDS_PER_DAY

    def generate_file_name(self):
        """
        Generate the file name for this :class:`Almanac`.

        To use the inbuilt functionality of this module to automatically find
        and read in :class:`Almanac` objects from file without having to
        re-compute them, it is important that they are saved to disk using
        the file name generated by this function.

        Returns
        -------
        filename : :obj:`str`
            A file name for this :class:`Almanac`, constructed from its
            parameters (i.e. dates, resolutions, sky position). The file
            extension used is ``.amn``.
        """
        filename = 'almanac_R%3.1f_D%2.1f_start%s_end%s_res%3f.amn' % (
            self.ra, self.dec, self.start_date.strftime('%y%m%d'),
            self.end_date.strftime('%y%m%d'), self.resolution,
        )
        return filename

    # Initialization
    def __init__(self, ra, dec, start_date, end_date=None,
                 observing_period=None, observer=UKST_TELESCOPE,
                 minimum_airmass=2.0, resolution=15.,
                 populate=True, alm_file_path='./'):
        """
        __init__ function for the :class:`Almanac` class.

        Assuming ``populate=True``, this function will attempt to load an
        Almanac from file by looking in `alm_file_path` for a file with a name
        matching the one generated by :any:`generate_file_name`. If found,
        it will load the data without the need for recomputation.

        Parameters
        ----------
        ra : :obj:`float`, decimal degrees
            Right ascension
        dec : :obj:`float`, decimal degrees
            Declination
        start_date : :obj:`datetime.datetime`
            Start date
        end_date : :obj:`datetime.datetime`
            End date
        observing_period : :obj:`float`, days
            Length of observing period for this :class:`Almanac`. Defaults to
            :obj:`None` (is not required if ``end_date`` is used).
        observer : :any:`ephem.Observer`
            Defaults to :any:`UKST_TELESCOPE`
        minimum_airmass : float
            Maximum airmass (i.e. minimum altitude) to be considered
            observable. Defaults to 2.0.
        resolution : float, minutes
            Resolution of the :any:`Almanac`. Defaults to 15.
        populate : bool
            Flag denoting whether ``__init__`` method should populate the data
            arrays. Defaults to try.
        alm_file_path : :obj:`str`, path
            File path where the function should look for saved almanacs.
            Defaults to the present working directory.

        Raises
        ------
        ValueError
            Raised if the user attempts to specify neither of ``end_date`` or
            ``observing_period``. If both are specified, ``end_date`` takes
            precedence.
        """
        if end_date is None and observing_period is None:
            raise ValueError("Must specify either one of end_date or "
                             "observing_period")

        self.ra = ra
        self.dec = dec
        self.start_date = start_date
        if end_date:
            self.end_date = end_date
        else:
            self.compute_end_date(observing_period)
        # self.data = np.array([], dtype=[
        #     ('date', float),
        #     ('airmass', float),
        # ])
        self.observer = observer
        self.minimum_airmass = minimum_airmass
        self.resolution = resolution
        self.alm_file_path = alm_file_path

        # See if an almanac with these properties already exists in the PWD
        if populate:
            load_success = self.load()
            if not load_success:
                # Couldn't find an almanac to load, so we need to
                # populate this one
                self.calculate_airmass()

        return

    # Save & read from disk
    # Uses pickle to seralize objects
    def save(self, filename=None, filepath=alm_file_path):
        """
        Save this :class:`Almanac` to disk using :any:`pickle`.

        Parameters
        ----------
        filename : :obj:`str`
            Name to give to the output file. Defaults to None, so that
            :any:`generate_file_name` will be used instead. Keeping this value
            as :obj:`None` is *strongly* recommended.
        filepath : :obj:`str`
            Directory to save the :any:`Almanac` to. Defaults to
            :any:`alm_file_path`.

        Raises
        ------
        ValueError
            Raised if the provided ``filepath`` does not end in '/'.
        """
        if filepath[-1] != '/':
            raise ValueError('filepath must end with /')
        if filename is None:
            filename = self.generate_file_name()
        with open('%s%s' % (filepath, filename, ), 'w') as fileobj:
            pickle.dump(self, fileobj, pickle.HIGHEST_PROTOCOL)
        return

    def load(self, filepath=alm_file_path):
        """
        Load almanac data from file.

        This function will attempt to load an almanac based on the file name of
        almanacs available in the directory specified by filepath. This will be
        based on the RA, Dec, starttime, endtime and resolution information
        encoded within the filename. If successful, the function returns True.
        If a filename matching the calling
        almanac is not found, then the function will exit and return False.

        Parameters
        ----------
        filepath : :obj:`str`
            Path to the directory in which saved :class:`Almanac` objects may
            be found.

        Returns
        -------
        bool
            True denotes that data was successfully loaded from file; False
            denotes otherwise.

        Raises
        ------
        ValueError
            Raised if the provided ``filepath`` does not end in '/'.
        """
        if filepath is None:
            filepath = './'
        if filepath[-1] != '/':
            raise ValueError('filepath must end with /')
        files_present = os.listdir(filepath)
        if self.generate_file_name() not in files_present:
            return False

        try:
            with open('%s%s' % (filepath,
                                self.generate_file_name(), )) as fileobj:
                file_almanac = pickle.load(fileobj)
        except EOFError:
            return False

        self.ra = file_almanac.ra
        self.dec = file_almanac.dec
        self.start_date = file_almanac.start_date
        self.end_date = file_almanac.end_date
        self.observer = file_almanac.observer
        self.resolution = file_almanac.resolution
        self.data = file_almanac.data
        self.minimum_airmass = file_almanac.minimum_airmass
        self.alm_file_path = file_almanac.alm_file_path

        return True

    # Computation functions
    def generate_almanac_bruteforce(self, full_output=True):
        """
        Compute Almanac data.

        Parameters
        ----------
        full_output : :obj:`bool`
            Denotes whether the function should return all the computed data
            (True), or the total number of data points computed (False).
            Defaults to True.

        Returns
        -------
        time_total : :obj:`float`, hours
            The number of hours that this :class:`Almanac` has had computed
            for it. Only returned if ``full_output=False``.
        dates_j2000, sol_alt, lun_alt, target_alt, dark_time : :obj:`numpy.array`
            Full computed Almanac data. Only returned if ``full_output=True``.

        """
        # Define almanac-dependent horizons
        observable = np.arcsin(1. / self.minimum_airmass)

        # initialise ephem fixed body object for pointing
        logging.debug('Creating ephem FixedBody at %3.1f, %2.1f' % (
            self.ra, self.dec,
        ))
        target = ephem.FixedBody()
        target._ra = np.radians(self.ra)  # radians everywhere
        target._dec = np.radians(self.dec)  # radians everywhere
        target._epoch = ephem.J2000
        logging.debug('ephem FixedBody generated (%1.2f, %1.2f, %s)' %
                      (target._ra, target._dec, target._epoch))

        # Calculate the time grid
        if self.data is None or len(self.data) == 0:
            # Set the observer start to the midday before the observations
            # should start
            start_dt = self.observer.date = get_utc_datetime(
                datetime.datetime.combine(self.start_date,
                                          datetime.time(12,0,0)))
            self.observer.date = start_dt
            # Similarly, the end date time is the midday after observing
            # should finish
            end_dt = get_utc_datetime(
                datetime.datetime.combine(self.end_date + datetime.timedelta(1),
                                          datetime.time(12, 0, 0)))
            observing_period = (
                end_dt - start_dt
                               ).total_seconds() / SECONDS_PER_DAY
            dates_j2000 = (self.observer.date +
                           np.arange(0, observing_period,
                                     self.resolution / 1440.))
        else:
            dates_j2000 = sorted(self.data['date'])

        # Perform the computation
        time_total = 0.
        sol_alt, lun_alt, target_alt, dark_time = [], [], [], []
        for d in dates_j2000:
            self.observer.date = d
            SUN.compute(self.observer)
            MOON.compute(self.observer)
            target.compute(self.observer)

            can_observe = (SUN.alt <
                           SOLAR_HORIZON) and (MOON.alt < LUNAR_HORIZON)
            if full_output:
                sol_alt.append(SUN.alt)
                lun_alt.append(MOON.alt)
                target_alt.append(target.alt)
                # logging.debug('Computed target alt at %5.2f: %1.3f (%2.2f)' %
                #               (d, target.alt, np.degrees(target.alt)))
                dark_time.append(can_observe)
            else:
                if can_observe and (target.alt > observable):
                    time_total += self.resolution

        if full_output:
            # convert results to numpy arrays
            sol_alt = np.array(sol_alt)
            lun_alt = np.array(lun_alt)
            target_alt = np.array(target_alt)
            dark_time = np.array(dark_time)
            return dates_j2000, sol_alt, lun_alt, target_alt, dark_time
        else:
            return time_total / 60.

    def calculate_airmass(self):
        logging.debug('Computing airmass values for almanac at %3.1f, %2.1f '
                      'from %s to %s' %
                      (self.ra, self.dec, self.start_date.strftime('%y-%m-%d'),
                       self.end_date.strftime('%y-%m-%d'),
                       ))
        dates, sun, moon, target, dark_time = self.generate_almanac_bruteforce(
            full_output=True)
        logging.debug('Min and max target_alt from generate_almanac_'
                      'bruteforce: '
                      '%1.3f, %1.3f' % (min(target), max(target)))

        airmass_values = np.clip(np.where(target > np.radians(10.),
                                          1./np.sin(target), 99.), 0., 9.)

        self.data = np.array([tuple(x) for x in
                              np.vstack((dates, airmass_values)).T.tolist()],
                             dtype=[
                                 ('date', float),
                                 ('airmass', float)
                             ])
        self.data.sort(axis=-1, order='date')

        return

    def next_observable_period(self, datetime_from, datetime_to=None,
                               tz=UKST_TIMEZONE):
        """
        Determine the next period when this field is observable, based on
        airmass only (i.e. doesn't consider daylight/night, dark/grey time,
        etc.).

        Parameters
        ----------
        datetime_from: :obj:`datetime.datetime`
            Datetime which to consider from. Must be within the bounds of the
            Almanac.
        datetime_to: :obj:`datetime.datetime`
            Datetime which to consider to. Defaults to None, in which case the
            entire Almanac is checked from datetime_from.
        tz: :obj:`pytz.timezone`
            The pytz timezone the observer is in. Defaults to UKST_TIMEZONE.

        Returns
        -------
        obs_start, obs_end: float
            The start and end times when this field is observable. Note that
            datetimes are returned in pyephem format.
        """
        logging.debug('Using next_observable_period')
        # Input checking
        if datetime_to is None:
            datetime_to = pytz.utc.localize(ephem_to_dt(
                self.data['date'][-1])).astimezone(tz).replace(tzinfo=None)
            logging.debug('Calculated datetime_to of %s' % str(datetime_to))
        if datetime_to < datetime_from:
            raise ValueError('datetime_from must occur before datetime_to')
        if datetime_from.date() < self.start_date:
            raise ValueError('datetime_from is before start_date for this '
                             'Almanac!')
        if datetime_to.date() > self.end_date + datetime.timedelta(1):
            raise ValueError('datetime_to is after end_date for this '
                             'Almanac!')

        ephem_dt = ephem.Date(tz.localize(datetime_from).astimezone(pytz.utc))
        ephem_limiting_dt = ephem.Date(tz.localize(
            datetime_to).astimezone(pytz.utc))
        logging.debug('Looking at airmasses from ephem dts %5.3f to %5.3f' %
                      (ephem_dt, ephem_limiting_dt))

        # Determine obs_start and obs_end
        try:
            obs_start = self.data[np.logical_and(
                self.data['airmass'] <= self.minimum_airmass,
                np.logical_and(
                    ephem_dt <= self.data['date'],
                    self.data['date'] < ephem_limiting_dt))]['date'][0]
        except IndexError:
            # No nights left in this DarkAlmanac, so return None for both
            obs_start, obs_end = None, None
            return obs_start, obs_end
        try:
            obs_end = self.data[np.logical_and(
                self.data['airmass'] > self.minimum_airmass,
                np.logical_and(
                    obs_start < self.data['date'],
                    self.data['date'] < ephem_limiting_dt
                ))]['date'][0]
            # obs_end = (t for t, b in sorted(self.airmass.iteritems()) if
            #            obs_start < t < ephem_limiting_dt and
            #            b > self.minimum_airmass).next()
        except IndexError:
            # No end time found, so use the last time in the almanac,
            # plus a resolution element
            obs_end = self.data['date'][-1] + (self.resolution /
                                               (SECONDS_PER_DAY /
                                                60.))

        if obs_start is None and obs_end is not None:
            raise RuntimeError("CODE ERROR: next_observable_period should not "
                               "return (None, valid value)")
        if obs_start is not None and obs_end is None:
            raise RuntimeError("CODE ERROR: next_observable_period should not "
                               "return (valid value, None)")
        return obs_start, obs_end

    def hours_observable(self, datetime_from, datetime_to=None,
                         exclude_grey_time=True,
                         exclude_dark_time=False,
                         dark_almanac=None,
                         tz=UKST_TIMEZONE,
                         hours_better=False,
                         minimum_airmass=2.0,
                         airmass_delta=0.05):
        """
        Calculate how many hours this field is observable for between two
        datetimes.

        Parameters
        ----------
        datetime_from, datetime_to: :obj:`datetime.datetime`
            The datetimes between which we should calculate the number of
            observable hours remaining. These datetimes must be between the
            start and end dates of the almanac, or an error will be returned.
            datetime_to will default to None, such that the remainder of the
            almanac will be used.
        exclude_grey_time, exclude_dark_time: :obj:`bool`
            Boolean value denoting whether to exclude grey time or dark time
            from the calculation. Defaults to exclude_grey_time=True,
            exclude_dark_time=False (so only dark time will be counted as
            'observable'.) The legal combinations are:
            False, False (all night time is counted)
            False, True (grey time only)
            True, False (dark time only)
            Attempting to set both value to True will raise a ValueError, as
            this would result in no available observing time.
        dark_almanac: :class:`DarkAlmanac`
            An instance of DarkAlmanac used to compute whether time is grey or
            dark. Defaults to None, at which point a DarkAlmanac will be
            constructed (if required).
        tz: :obj:`pytz.timezone`
            The timezone that the (naive) datetime objects are being passed as.
            Defaults to UKST_TIMEZONE.
        hours_better: :obj:`bool`
            Optional Boolean, denoting whether to return only
            hours_observable which have airmasses superior to the airmass
            at datetime_now (True) or not (False). Defaults to False.
        minimum_airmass: :obj:`float`
            Something of a misnomer; this is actually the *maximum* airmass at which
            a field should be considered visible (a.k.a. the minimum altitude).
            Defaults to 2.0 (i.e. an altitude of 30 degrees). If hours_better
            is used, the comparison airmass will be the minimum of the airmass at
            datetime_from and minimum_airmass.
        airmass_delta : :obj:`float`
            Denotes the delta airmass that should be used to compute
            hours_observable if hours_better=True. The hours_observable will be
            computed against a threshold airmass value of
            (airmass_now + airmass_delta). This
            has the effect of 'softening' the hours_observable calculation for
            zenith fields, i.e. fields rapidly heading towards the minimum_airmass
            limit will be prioritized over those just passing through zenith.

        Returns
        -------
        hours_obs: :obj:`float`
            The number of observable hours for this field between datetime_from
            and datetime_to.

        """
        logging.debug('Running hours_observable')
        # Input checking
        if exclude_grey_time and exclude_dark_time:
            raise ValueError('Cannot set both exclude_grey_time and '
                             'exclude_dark_time to True - this results in no '
                             'observing time!')
        if datetime_to is None:
            datetime_to = pytz.utc.localize(ephem_to_dt(self.data['date'][-1]))\
                .astimezone(tz).replace(tzinfo=None)
            logging.debug('Computed datetime_to: %s' %
                          datetime_to.strftime('%Y-%m-%d %H:%M'))
        if datetime_to < datetime_from:
            raise ValueError('datetime_from must occur before datetime_to')
        if datetime_from.date() < self.start_date:
            raise ValueError('datetime_from is before start_date for this '
                             'Almanac!')
        if datetime_to.date() > self.end_date + datetime.timedelta(1):
            raise ValueError('datetime_from is before start_date for this '
                             'Almanac!')
        if dark_almanac is not None and not isinstance(dark_almanac,
                                                       DarkAlmanac):
            raise ValueError('dark_almanac must be None, or an instance of '
                             'DarkAlmanac')

        if dark_almanac is None:
            dark_almanac = DarkAlmanac(datetime_from.date(),
                                       end_date=datetime_to.date() -
                                       datetime.timedelta(1),
                                       observer=self.observer,
                                       resolution=self.resolution,
                                       populate=True)

        # There are two things that need to be checked to calculate
        # hours observable:
        # - Is the target airmass below the specified threshold?
        # - Is it night/dark/grey time?
        # How we will do this is:
        # - Look at the airmass almanac, and find the intervals airmass is above
        # the threshold
        # - Loop over these intervals using one of the next_*_period functions,
        # as appropriate, until we exhaust the interval, keeping count of the
        # amount of time remaining

        if exclude_dark_time:
            period_function = dark_almanac.next_grey_period
        elif exclude_grey_time:
            period_function = dark_almanac.next_dark_period
        else:
            period_function = dark_almanac.next_night_period

        hours_obs = 0.
        dt_up_to = copy.copy(datetime_from)
        airmass_now = self.data[
            self.data['date'] >= ephem.Date(tz.localize(datetime_from).
                                            astimezone(pytz.utc))
        ]['airmass'][0]
        # airmass_now = (v for k, v in sorted(self.airmass.iteritems()) if
        #                k >= ephem.Date(tz.localize(
        #                    datetime_from).astimezone(pytz.utc))
        #                ).next()
        while dt_up_to < datetime_to:
            logging.debug('Up to %s; going til %s' %
                          (dt_up_to.strftime(EPHEM_DT_STRFMT),
                           datetime_to.strftime(EPHEM_DT_STRFMT), ))
            next_chance_start, next_chance_end = self.next_observable_period(
                dt_up_to, datetime_to=datetime_to, tz=tz
            )
            if next_chance_start is None:
                # No further opportunities - break out
                break
            logging.debug('Next ephem chance window: %5.3f to %5.3f' % (
                next_chance_start, next_chance_end,
            ))
            next_chance_start = pytz.utc.localize(
                ephem_to_dt(next_chance_start)
            ).astimezone(tz).replace(tzinfo=None)
            next_chance_end = pytz.utc.localize(
                ephem_to_dt(next_chance_end)
            ).astimezone(tz).replace(tzinfo=None)
            logging.debug('Next local chance window: %s to %s' %
                          (next_chance_start.strftime(EPHEM_DT_STRFMT),
                           next_chance_end.strftime(EPHEM_DT_STRFMT), ))

            # Go through this 'chance' window, looking for when the relevant
            # period function is satisfied
            while next_chance_start < min(next_chance_end, datetime_to):
                next_per_start, next_per_end = period_function(
                    next_chance_start, limiting_dt=next_chance_end, tz=tz)
                if next_per_start is None:
                    # No further opportunities - break out
                    break
                logging.debug('next_per_start: %5.3f, next_per_end: %5.3f' %
                              (next_per_start, next_per_end, ))

                if hours_better:
                    # Look at the Almanac over the next period to work out what
                    # proportion of the period is better than datetime_now
                    logging.debug('Searching for better hours - '
                                  'Period considered: %5.3f to %5.3f' %
                                  (next_per_start, next_per_end, ))
                    half_res_in_days = self.resolution * 60. / SECONDS_PER_DAY
                    better_per = self.data[np.logical_and(
                        np.logical_and(
                            (next_per_start - half_res_in_days) <
                            self.data['date'],
                            self.data['date'] <
                            (next_per_end - half_res_in_days)),
                        self.data['airmass'] <= min(airmass_now + airmass_delta,
                                                    minimum_airmass),
                    )]
                    # better_per = [k for
                    #               k, v in sorted(self.airmass.iteritems()) if
                    #               (next_per_start -
                    #                half_res_in_days) <
                    #               k < (next_per_end -
                    #                    half_res_in_days) and
                    #               v <= airmass_now]
                    logging.debug('Resl. elements in better_per: %d' %
                                  better_per.shape[-1])
                    whole_per = self.data[np.logical_and(
                        (next_per_start - half_res_in_days) < self.data['date'],
                        self.data['date'] < (next_per_end - half_res_in_days)
                    )]
                    # whole_per = [k for
                    #              k, v in sorted(self.airmass.iteritems()) if
                    #              (next_per_start -
                    #               half_res_in_days) <
                    #              k < (next_per_end -
                    #                   half_res_in_days)]
                    logging.debug('Resl. elements in whole_per: %d' %
                                  whole_per.shape[-1])
                    if whole_per.shape[-1] > 0:
                        logging.debug('Period bounds found: %5.3f to %5.3f '
                                      '(%d units of resolution %2.1f)' %
                                      (whole_per['date'][0],
                                       whole_per['date'][-1],
                                       whole_per.shape[-1],
                                       self.resolution))
                    hours_obs += better_per.shape[-1] * (self.resolution / 60.)
                    # factor = float(better_per.shape[-1]) / float(whole_per.shape[-1])
                else:
                    # factor = 1.
                    hours_obs += (next_per_end - next_per_start) * 24.  # hours
                # Update next chance start
                next_chance_start = pytz.utc.localize(ephem_to_dt(
                    next_per_end)).astimezone(tz).replace(tzinfo=None)

            # Update dt_up_to
            dt_up_to = pytz.utc.localize(ephem_to_dt(
                next_chance_end)).astimezone(tz).replace(tzinfo=None)
            logging.debug('Now up to %s' % dt_up_to.strftime(EPHEM_DT_STRFMT))

        return hours_obs


class DarkAlmanac(Almanac):
    """
    Subclass of Almanac, which holds information on what times are 'dark'
    """

    _sun_alt = None

    # Setters and getters
    @property
    def ra(self):
        """
        Locked to 0.
        """
        return 0.

    @ra.setter
    def ra(self, r):
        r = float(r)
        if abs(r - 0.) > 1e-5:
            raise ValueError('Dark almanac must have RA = 0')
        self._ra = 0.

    @property
    def dec(self):
        """
        Locked to 0.
        """
        return self._dec

    @dec.setter
    def dec(self, r):
        r = float(r)
        if abs(r - 0.) > 1e-5:
            raise ValueError('Dark almanac must have DEC = 0')
        self._dec = 0.

    @property
    def minimum_airmass(self):
        """
        Locked to 2.0.
        """
        return self._minimum_airmass

    @minimum_airmass.setter
    def minimum_airmass(self, a):
        self._minimum_airmass = 2.

    # Will alter the formation of the 'data' attribute to hold DarkAlmanac info

    @property
    def data(self):
        """
        As for the data attribute of :any:`Almanac`, but with different columns

        The columns required for a :any:`DarkAlmanac` ``data`` array are:
        'date'
        'dark_time'
        'sun_alt'
        """
        return self._data

    @data.setter
    def data(self, a):
        if not isinstance(a, np.ndarray):
            raise ValueError("data must be a numpy array")
        try:
            _ = a['date']
            _ = a['dark_time']
            _ = a['sun_alt']
        except:
            raise ValueError("data must be a numpy structured array "
                             "with columns 'date', 'dark_time' and 'sun_alt'")
        self._data = a

    # @property
    # def dark_time(self):
    #     """
    #     A catalogue of whether time should be considered 'dark' or not
    #     """
    #     return self._airmass
    #
    # @dark_time.setter
    # def dark_time(self, d):
    #     if not isinstance(d, dict):
    #         raise ValueError('dark_time must be a dictionary of datetimes '
    #                          'and corresponding Boolean values')
    #     self._airmass = d
    #
    # @property
    # def sun_alt(self):
    #     """
    #     Altitude of Sun at given time (radians)
    #     """
    #     return self._sun_alt
    #
    # @sun_alt.setter
    # def sun_alt(self, d):
    #     if not isinstance(d, dict):
    #         raise ValueError('sun_alt must be a dictionary of datetimes '
    #                          'and corresponding Boolean values')
    #     self._sun_alt = d

    # Class functions
    def generate_file_name(self):
        filename = 'darkalmanac_start%s_end%s_res%3f.amn' % (
            self.start_date.strftime('%y%m%d'),
            self.end_date.strftime('%y%m%d'), self.resolution,
        )
        return filename

    # def load(self, filepath='./'):
    #     # Supers the original class load, adds in sun_alt calculation
    #     logging.debug('super-ing base Almanac load method')
    #     load_available = super(DarkAlmanac, self).load(filepath=filepath)
    #
    #     # Re-load and do DarkAlmanac-specific load
    #     if load_available:
    #         try:
    #             with open('%s%s' % (filepath,
    #                                 self.generate_file_name(), )) as fileobj:
    #                 file_almanac = pickle.load(fileobj)
    #         except EOFError:
    #             return False
    #
    #         self.sun_alt = file_almanac.sun_alt
    #     else:
    #         return False
    #
    #     return True

    # Initialization
    def __init__(self, start_date, end_date=None,
                 observing_period=None, observer=UKST_TELESCOPE,
                 resolution=15., populate=True, alm_file_path='./'):
        # 'super' the Almanac __init__ method, but do NOT attempt to
        # populate the DarkAlmanac from this method (uses a different
        # method)
        logging.debug('super-ing standard almanac creation')
        super(DarkAlmanac, self).__init__(0., 0., start_date, end_date=end_date,
                                          observing_period=observing_period,
                                          observer=observer,
                                          minimum_airmass=None,
                                          resolution=resolution, populate=False,
                                          alm_file_path=alm_file_path)
        self._minimum_airmass = 2.
        self.data = np.array([], dtype=[
            ('date', float),
            ('dark_time', bool),
            ('sun_alt', float)
        ])
        self.alm_file_path = alm_file_path

        logging.debug('Looking for existing dark almanac file')
        # See if an almanac with these properties already exists in the PWD
        if populate:
            load_success = self.load(filepath=alm_file_path)
            if not load_success:
                # Couldn't find an almanac to load, so we need to
                # populate this one
                self.create_dark_almanac()

    def create_dark_almanac(self):
        """
        Perform the necessary calculations to populate this almanac
        """
        logging.debug('Creating dark almanac')
        logging.debug('Calculating from standard almanac functions...')
        dates, sun, moon, target, is_dark_time = \
            self.generate_almanac_bruteforce(full_output=True)

        times_per_day = 1440. / self.resolution
        if abs(times_per_day - int(times_per_day)) > 1e-5:
            raise ValueError('Dark almanac resolution must divide into a day '
                             'with no remainder (i.e. resolution must be a '
                             'divisor of 1440).')

        logging.debug('Populating DarkAlmanac data list')

        self.data = np.array([tuple(x) for x in
                              np.vstack((dates, is_dark_time, sun)).T.tolist()],
                             dtype=[
                                 ('date', float),
                                 ('dark_time', bool),
                                 ('sun_alt', float)
                             ])
        self.data.sort(axis=-1, order='date')

        # Blank or initialise the relevant dicts
        # self.dark_time = {}
        # self.sun_alt = {}
        #
        # logging.debug('Populating dark time dicts')
        # for i in range(len(dates)):
        #     self.dark_time[dates[i]] = is_dark_time[i]
        #     self.sun_alt[dates[i]] = sun[i]

        logging.debug('Dark almanac created!')

        return

    def compute_period_ephem_dts(self, dt, limiting_dt=None, tz=UKST_TIMEZONE):
        """
        Internal helper function.
        Calculate the start (and if needed, end) pyehpem dt for computing a
        night/grey/dark period.

        Parameters
        ----------
        dt: : :obj:`datetime.datetime`
            Datetime from which to begin searching.
        limiting_dt: :obj:`datetime.datetime`
            Datetime beyond which should not be investigated. Useful for, e.g.
            getting the dark time for a single night. Defaults to None, at which
            point the entire DarkAlmanac will be searched.
        tz: :obj:`pytz.timezone`
            The timezone of the naive datetime object passed as dt. Defaults
            to UKST_TELESCOPE.

        Returns
        -------
        ephem_dt, ephem_limiting_dt: float
            The start and end dts of the interval in question, expressed in the
            pyephem datetime convention.
        """
        logging.debug('Using compute_period_ephem_dts')
        # Input checking
        if not isinstance(dt, datetime.datetime):
            raise ValueError('dt must be an instance of datetime.datetime')
        if dt.date() < self.start_date or dt.date() > (self.end_date +
                                                       datetime.timedelta(1)):
            raise ValueError('dt (%s) must be between start_date (%s)'
                             ' and end_date (%s) for '
                             'this DarkAlmanac' %
                             (dt.strftime('%Y-%m-%d %H:%M'),
                              self.start_date.strftime('%Y-%m-%d'),
                              self.end_date.strftime('%Y-%m-%d')))

        # The datetime needs to be pushed into UT
        dt = tz.localize(dt).astimezone(pytz.utc).replace(tzinfo=None)
        ephem_dt = ephem.Date(dt)
        if limiting_dt is None:
            ephem_limiting_dt = self.data['date'][-1] + (
                self.resolution / (SECONDS_PER_DAY / 60.0)
            # ephem_limiting_dt = sorted(self.dark_time.keys())[-1] + (
            #     self.resolution / (SECONDS_PER_DAY / 60.0)
            )
        else:
            limiting_dt = tz.localize(
                limiting_dt).astimezone(pytz.utc).replace(tzinfo=None)
            ephem_limiting_dt = ephem.Date(limiting_dt)

        return ephem_dt, ephem_limiting_dt

    def next_night_period(self, dt, limiting_dt=None, tz=UKST_TIMEZONE):
        """
        Determine when the next period of night time is

        Parameters
        ----------
        dt: :obj:`datetime.datetime`
            Datetime from which to begin searching.
        limiting_dt: :obj:`datetime.datetime`
            Datetime beyond which should not be investigated. Useful for, e.g.
            getting the dark time for a single night. Defaults to None, at which
            point the entire DarkAlmanac will be searched.
        tz: :obj:`pytz.timezone`
            The timezone of the naive datetime object passed as dt. Defaults
            to UKST_TELESCOPE.

        Returns
        -------
        night_start, night_end: float
            The datetimes at which the next block of night time starts and ends.
            If dt is in a night time period, dark_start should be approximately
            dt (modulo the resolution of the DarkAlmanac). Note that values
            are returned using the pyephem date syntax; use EPHEM_TO_MJD to
            convert to MJD.
        """
        logging.debug('Using next_night_period')
        # All necessary input checking is done by compute_period_ephem_dts
        ephem_dt, ephem_limiting_dt = self.compute_period_ephem_dts(dt,
                                                                    limiting_dt=
                                                                    limiting_dt,
                                                                    tz=tz)
        logging.debug('Computed ephem dt range: %5.3f to %5.3f' %
                      (ephem_dt, ephem_limiting_dt, ))

        # Search for the next time slot when the Sun is below SOLAR_HORIZON
        try:
            night_start = self.data[np.logical_and(
                np.logical_and(
                    ephem_dt <= self.data['date'],
                    self.data['date'] < ephem_limiting_dt),
                self.data['sun_alt'] < SOLAR_HORIZON
            )]['date'][0]
            # night_start = (t for t, b in sorted(self.sun_alt.iteritems()) if
            #                ephem_dt <= t < ephem_limiting_dt and
            #                b < SOLAR_HORIZON).next()
        except IndexError:
            # No nights left in this DarkAlmanac, so return None for both
            return None, None
        try:
            night_end = self.data[np.logical_and(
                np.logical_and(
                    night_start < self.data['date'],
                    self.data['date'] < ephem_limiting_dt),
                self.data['sun_alt'] >= SOLAR_HORIZON
            )]['date'][0]
            # night_end = (t for t, b in sorted(self.sun_alt.iteritems()) if
            #              night_start < t < ephem_limiting_dt and
            #              b >= SOLAR_HORIZON).next()
        except IndexError:
            # No end time found, so use the limiting time (which is the end of
            # the almanac if not passed)
            night_end = ephem_limiting_dt

        return night_start, night_end

    def next_dark_period(self, dt, limiting_dt=None, tz=UKST_TIMEZONE):
        """
        Determine when the next period of dark time is

        Parameters
        ----------
        dt: :obj:`datetime.datetime`
            Datetime from which to begin searching.
        limiting_dt: :obj:`datetime.datetime`
            Datetime beyond which should not be investigated. Useful for, e.g.
            getting the dark time for a single night. Defaults to None, at which
            point the entire DarkAlmanac will be searched.
        tz: :obj:`pytz.timezone`
            The timezone of the naive datetime object passed as dt. Defaults
            to UKST_TELESCOPE.

        Returns
        -------
        dark_start, dark_end: float
            The datetimes at which the next block of dark time starts and ends.
            If dt is in a dark time period, dark_start should be approximately
            dt (modulo the resolution of the DarkAlmanac). Note that values
            are returned using the pyephem date syntax; use EPHEM_TO_MJD to
            convert to MJD.
        """
        logging.debug('Using next_dark_period')
        # All necessary input checking is done by compute_period_ephem_dts
        ephem_dt, ephem_limiting_dt = self.compute_period_ephem_dts(dt,
                                                                    limiting_dt=
                                                                    limiting_dt,
                                                                    tz=tz)

        try:
            dark_start = self.data[np.logical_and(
                np.logical_and(
                    ephem_dt <= self.data['date'],
                    self.data['date'] < ephem_limiting_dt
                ),
                self.data['dark_time']
            )]['date'][0]
            # dark_start = (t for t, b in sorted(self.dark_time.iteritems()) if
            #               ephem_dt <= t < ephem_limiting_dt and b).next()
        except IndexError:
            # No dark time left in this almanac; return None to both parameters
            return None, None
        try:
            dark_end = self.data[np.logical_and(
                np.logical_and(
                    dark_start < self.data['date'],
                    self.data['date'] < ephem_limiting_dt
                ),
                ~self.data['dark_time']
            )]['date'][0]
            # dark_end = (t for t, b in sorted(self.dark_time.iteritems()) if
            #             dark_start < t < ephem_limiting_dt and
            #             not b).next()
        except IndexError:
            # No end time found, so use the limiting time (which is the end of
            # the almanac if not passed)
            dark_end = ephem_limiting_dt

        return dark_start, dark_end

    def next_grey_period(self, dt, limiting_dt=None, tz=UKST_TIMEZONE):
        """
        Determine when the next period of grey time is

        Parameters
        ----------
        dt: :obj:`datetime.datetime`
            Datetime from which to begin searching.
        limiting_dt: :obj:`datetime.datetime`
            Datetime beyond which should not be investigated. Useful for, e.g.
            getting the dark time for a single night. Defaults to None, at which
            point the entire DarkAlmanac will be searched.
        tz: :obj:`pytz.timezone`
            The timezone of the naive datetime object passed as dt. Defaults
            to UKST_TELESCOPE.

        Returns
        -------
        grey_start, grey_end: float
            The datetimes at which the next block of grey time starts and ends.
            If dt is in a grey time period, dark_start should be approximately
            dt (modulo the resolution of the DarkAlmanac). Note that values
            are returned using the pyephem date syntax; use EPHEM_TO_MJD to
            convert to MJD, or ephem_to_td to convert to a UTC datetime.
        """
        logging.debug('Using next_grey_period')
        # All necessary input checking is done by compute_period_ephem_dts
        ephem_dt, ephem_limiting_dt = self.compute_period_ephem_dts(dt,
                                                                    limiting_dt=
                                                                    limiting_dt,
                                                                    tz=tz)

        # This is slightly more complex that just looking at the dark
        # time calculation - we need to check that:
        # 1) The Sun is below the twilight horizon
        # 2) It is *not* dark time
        # We do this by:
        # - Getting the next sun-down window
        # - Getting any dark period during that window
        # - Looking for either a grey period before or after the dark period

        # Get the next Sun-down period
        night_start, night_end = self.next_night_period(dt,
                                                        limiting_dt=limiting_dt,
                                                        tz=tz)
        if night_start is None:
            # No grey time available
            return None, None

        # Determine the next dark period
        dark_start, dark_end = self.next_dark_period(dt,
                                                     limiting_dt=limiting_dt,
                                                     tz=tz)

        if dark_start is None or dark_start > night_end:
            # There is no dark time left (or no dark time in the next night),
            # so the next night period must be all grey time
            grey_start, grey_end = night_start, night_end
        elif dark_start == night_start and dark_end < night_end:
            # The grey period must start when the dark ends
            # Make sure there isn't another dark period before night_end
            dark_start_next, dark_end_next = self.next_dark_period(
                pytz.utc.localize(ephem_to_dt(
                    dark_end)).astimezone(tz).replace(tzinfo=None),
                limiting_dt=pytz.utc.localize(ephem_to_dt(
                    night_end)).astimezone(tz).replace(tzinfo=None),
                tz=pytz.utc
            )
            if dark_start_next is None:
                grey_start, grey_end = dark_end, night_end
            else:
                grey_start, grey_end = dark_end, dark_start_next
        elif dark_end == night_end and dark_start > night_start:
            # The (next) grey period must be at the start of the night
            grey_start, grey_end = night_start, dark_start
        else:
            # If this point has been reached, none of the night is grey time
            # Therefore, we need to start searching the next night
            # Basically, we get the function to call itself with dt one day
            # later until it hits a solution, or returns None
            if limiting_dt - dt > datetime.timedelta(1):
                return self.next_grey_period(dt + datetime.timedelta(1),
                                             limiting_dt=limiting_dt,
                                             tz=tz)
            else:
                return None, None

        return grey_start, grey_end

# ______________________________________________________________________________


def airmass(ra, dec, date_j2000=None, observer=UKST_TELESCOPE):
    # initialise ephem observer object with specified datetime
    if date_j2000 is not None:
        observer.date = date_j2000
    # initialise ephem target object with specified coordinates
    target = ephem.FixedBody()             # create ephem target object
    target._ra = np.radians(ra)          # radians everywhere
    target._dec = np.radians(dec)        # radians everywhere
    target.compute(observer)             # intialise ephem target object

    # return airmass if target is accessible with UKST
    if np.degrees(target.alt) > 5:
        return 1./np.sin(target.alt)
    else:
        return 99.

# ______________________________________________________________________________


def next_dark_period(observer=UKST_TELESCOPE):
    _date = observer.date
    # determine next sun and moon rise and set times from now.
    sunrise, sunset = next_sunrise(observer), next_sunset(observer)
    moonrise, moonset = next_moonrise(observer), next_moonset(observer)
    # are the sun and moon up now? (will they set before they rise?)
    is_sun_up = sunset < sunrise
    is_moon_up = moonset < moonrise
    while (is_sun_up or is_moon_up):
        until = observer.date + 1./1440 # advance at least 1 min into the future
        # advance time to when sun/moon have set (whichever is furthest ahead)
        if is_sun_up:
            until = max(until,  sunset)
        if is_moon_up:
            until = max(until, moonset)
        observer.date = until
        # update rise/set times as necessary
        if sunrise <= until:
            sunrise = next_sunrise(observer)
        if sunset <= until:
            sunset = next_sunset(observer)
        if moonrise <= until:
            moonrise = next_moonrise(observer)
        if moonset <= until:
            moonset = next_moonset(observer)
        # how about now? are the sun and moon up now?
        is_sun_up = sunset < sunrise
        is_moon_up = moonset < moonrise
    darkstart = observer.date
    # dark time ends when either the sun or the moon rises
    darkend = min(sunrise, moonrise)
    logging.debug('next dark period: %s --> %s' % (darkstart, darkend, ))
    # return observer.date to its original value
    observer.date = _date
    return darkstart, darkend        

# ______________________________________________________________________________


def hours_observable(ra, dec, observer=UKST_TELESCOPE,
                     start_date=None, end_date=None,
                     observing_period=365.25 / 2.,
                     minimum_airmass=2.0, verbose=False):

    # Input checking
    if end_date is None and observing_period is None:
        raise ValueError('Either end_date or observing_period must be defined')

    if start_date is None:
        start_date = utc_to_j2000() # Set start time to now

    # determine end of observing period if necessary
    if end_date is None:
        observer.date = start_date  # pyephem accepts datetime object
        end_date = observer.date + observing_period
        observer.date = end_date

    if end_date <= start_date:
        raise ValueError('The end date is before the start date! Refine either '
                         'your passed end_date or observing_period')
        
    # initialise ephem fixed body object for target
    target = ephem.FixedBody()
    target._ra = np.radians(ra)    # radians everywhere
    target._dec = np.radians(dec)  # radians everywhere

    # define angle corresponding to minimum_airmass and twilight_horizon
    observable = np.arcsin(1. / minimum_airmass)  # radians everywhere

    # hack around weird pyephem behavior near target's zenith
    # Precisely what is this doing?
    zenith = np.abs(np.abs(observer.lat) + np.radians(dec - 90.))
    observable = np.clip(observable, 0., zenith - np.radians(.1))
    
    def next_rise(obs):
        obs.horizon = observable
        target.compute(obs)
        return obs.next_rising(target)

    def next_set(obs):
        obs.horizon = observable
        target.compute(obs)
        return obs.next_setting(target)
        
    # from start of observing, compute next rise/set dates 
    observer.date = start_date    # pyephem accepts datetime object
    sunrise = next_sunrise(observer)
    sunset = next_sunset(observer)
    moonrise = next_moonrise(observer)
    moonset = next_moonset(observer)

    # given its Dec, check whether target actually ever sets, or is circumpolar
    target.compute(observer)
    always_up_declination = -np.pi/2. + np.abs(observer.lat) - observable
    target_always_up = dec < np.degrees(always_up_declination)

    # and then compute target rise/set times if necessary
    if target_always_up:
        targetrise, targetset = observer.date, end_date
    else:
        targetrise = next_rise(observer)
        targetset = next_set(observer)

    # and now step forward through the observing period
    hours_total = 0.
    while observer.date < end_date:
        
        # if sun rises before it sets then it is nighttime (good)
        is_night_time = sunrise < sunset 
        # if moon rises before it sets then it is darktime (good)
        is_dark_time = moonrise < moonset
        # if target sets before it rises then it is visible (good)
        is_target_up = (targetset < targetrise)  # or target_always_up

        # if target is observable record how long it is observable for
        if is_night_time and is_dark_time and is_target_up:

            # determine when target becomes unobservable
            until = min(sunrise, moonrise, targetset, end_date)

            # record amount of observable time in hours
            hours_tonight = (until - observer.date) * 24.
            # add this to the total observable time in the full period
            hours_total += hours_tonight

            if verbose:
                print (observer.date, '-->', until,
                       '(%.3f hr)' % hours_tonight)
        else:
            # if target is not observable, determine when it might next be
            # advance by a minimum of one minute to sidestep pyephem oddnesses
            until = observer.date + 1./1440
            if not is_night_time:
                until = max(until, sunset)
            if not is_dark_time:
                until = max(until, moonset)
            if not is_target_up:
                until = max(until, targetrise)

        if observer.date == until:
            pass
            # This code block triggers if until is the same as the osberver date
            # So, that's in the event that either:
            # - The target is observable, but the until unobservable date is
            #   somehow the same;
            # - The target is unobservable, but somehow is magically observable
            #   at the current date
            # print 'oh no!', observer.date
            # print
            # until = observer.date
            # print until, float(until)
            # print is_night_time; until = max(until, sunset)
            # print until, float(until)
            # print is_dark_time;  until = max(until, moonset)
            # print until, float( until )
            # print is_target_up; until = max(until, targetrise)
            # print until, float(until)
            # print
            # print 'currently', observer.date, float(observer.date)
            # print 'sunrise', sunrise, sunrise - observer.date
            # print 'sunset', sunset, sunset - observer.date
            # print 'is_night_time', is_night_time, sunset - observer.date
            # print 'moonrise', moonrise, moonrise - observer.date
            # print 'moonset', moonset, moonset - observer.date
            # print 'is_dark_time', is_dark_time, moonset-observer.date
            # print 'targetrise', targetrise, targetrise - observer.date
            # print 'targetset', targetset, targetset - observer.date
            # print 'is_target_up', is_target_up, targetrise - observer.date
            # print 'checks', targetset <= until, targetset - until, \
            #     targetset - next_set()
            # print 'until', until, until - observer.date
            # print 'target_always_up', target_always_up
#            time.sleep( 10. )

        # advance time to next change in circumstances
        observer.date = min(until, end_date)
        
        # update any/all event times that need updating
        if sunrise <= until:
            sunrise = next_sunrise(observer=observer)
        if sunset <= until:
            sunset = next_sunset(observer=observer)
        if moonrise <= until:
            moonrise = next_moonrise(observer=observer)
        if moonset <= until:
            moonset = next_moonset(observer=observer)
        if targetrise <= until and not target_always_up:
            targetrise = next_rise()
        if targetset <= until and not target_always_up:
            targetset = next_set()

    # all done!
    return hours_total
        
        
# ______________________________________________________________________________

def hours_observable_bruteforce(ra, dec, observer=UKST_TELESCOPE,
                                start_date=None, end_date=None,
                                observing_period=365.25 / 2.,
                                minimum_airmass=2.0, verbose=0,
                                resolution=15., full_output=False,
                                dates_j2000=None):

    # TODO: Add input checking

    # define horizons for observability computation
    observable = np.arcsin(1./ minimum_airmass) # radians everywhere
    solar_horizon = SOLAR_TWILIGHT_HORIZON - np.radians(0.24)
    lunar_horizon = LUNAR_TWILIGHT_HORIZON - np.radians(0.27)
    # remember that the sun and moon both have a size! (0.24 and 0.27 deg)
    
    # initialise ephem fixed body object for pointing
    target = ephem.FixedBody()
    target._ra = np.radians(ra)    # radians everywhere
    target._dec = np.radians(dec)  # radians everywhere

    # instantiate ephem sun/moon objects
    sun, moon = ephem.Sun(), ephem.Moon()

    # define time grid if not given 
    if dates_j2000 is None:
        # initialise ephem observer object for UKST @ SSO
        observer.date = start_date

        if end_date is None:
            end_date = observer.date + observing_period
        else :
            observing_period = end_date - observer.date
        dates_j2000 = (observer.date +
                       np.arange(0, observing_period, resolution / 1440.))

    # and finally the actual computation
    time_total = 0.
    sol_alt, lun_alt, target_alt, dark_time = [], [], [], []
    for d in dates_j2000.ravel():
        observer.date = d
        sun.compute(observer)
        moon.compute(observer)
        target.compute(observer)

        can_observe = (sun.alt < solar_horizon) and (moon.alt < lunar_horizon)
        if full_output:
            sol_alt.append(sun.alt)
            lun_alt.append(moon.alt)
            target_alt.append(target.alt)
            dark_time.append(can_observe)
        else :
            if can_observe and (target.alt > observable):
                time_total += resolution
            
    if full_output :
        # convert results to numpy arrays
        sol_alt = np.array(sol_alt).reshape(dates_j2000.shape)
        lun_alt = np.array(lun_alt).reshape(dates_j2000.shape)
        target_alt=np.array(target_alt).reshape(dates_j2000.shape)
        dark_time=np.array(dark_time).reshape(dates_j2000.shape)
        return dates_j2000, sol_alt, lun_alt, target_alt, dark_time
    else:
        return time_total/60.


# ______________________________________________________________________________

def hours_observable_from_almanac(almanac_filename,
                                  dates_j2000, is_dark_time,
                                  observer=UKST_TELESCOPE,
                                  ra=None, dec=None,
                                  start_date_j2000=None, end_date_j2000=None,
                                  observing_period=365.25 / 2.,
                                  minimum_airmass=2.0):
    # load almanac if it exists
    if os.path.exists(almanac_filename):
        with np.load(almanac_filename) as save_dict:
            ddates, airmass = save_dict['dates_J2000'], save_dict['airmass']
        # check that almanac follows same date grid as input is_dark_time
        if not np.allclose(dates_j2000, ddates):
            logging.info('''\n\nBADNESS!
            dates found in almanac with filename %s do not match those given
            to hours_observable_from_almanac. this almanac must be remade.
            existing almanac is being removed and reconstructed.
            ''' % almanac_filename)
            os.remove(almanac_filename)
            # discard this almanac if its not right

    # create almanac if necessary
    if not os.path.exists(almanac_filename):
        if ra is None or dec is None:
            raise RuntimeError('''BADNESS!
            almanac with filename %s does not exist.
            call hours_observable_from_almanac using ra, dec keywords to
            create a new almanac.  dying gracelessly now.
            ''' % almanac_filename)
        airmass = create_almanac(ra, dec, dates_j2000, almanac_filename)

    if end_date_j2000 is None:
        end_date_j2000 = observer.date + observing_period

    # identify when target is observable during observing period
    fair_game = is_dark_time and \
                (start_date_j2000 <= dates_j2000) and \
                (dates_j2000 <= end_date_j2000)

#    better_entries = np.sum( fair_game & ( airmass <= minimum_airmass ) )
    better_entries = np.interp(minimum_airmass,
                               np.sort(airmass[fair_game]),
                               np.arange(fair_game.sum()))
    # this gives the number of dark time entries within the specified time
    # period where the airmass in the almanac is less than the minimum.

    # use dates grid to determine resolution in hours
    first_date = dates_j2000[np.unravel_index(0, dates_j2000.shape)]
    last_date = dates_j2000[np.unravel_index(dates_j2000.size - 1,
                                              dates_j2000.shape)]
    resolution_in_hours = (last_date - first_date) * 24. / dates_j2000.size
    better_hours = better_entries * resolution_in_hours

    return better_hours
        
    
# ______________________________________________________________________________

def create_dark_almanac(dark_almanac_filename,
                        start_date_j2000=None, end_date_j2000=None,
                        observing_period=int(4 * 365.25),
                        resolution=15.):

    logging.info('creating dark almanac %s ...' % (dark_almanac_filename), )
    # sys.stdout.flush()
    
    dates, sun, moon, target, is_dark_time = hours_observable_bruteforce(
        ra=0., dec=0., 
        start_date=start_date_j2000, end_date=end_date_j2000,
        observing_period=observing_period,
        resolution=resolution, full_output=True)

    solar_horizon = SOLAR_TWILIGHT_HORIZON - np.radians(0.24)
    lunar_horizon = LUNAR_TWILIGHT_HORIZON - np.radians(0.27)
    # remember that the sun and moon both have a size! (0.24 and 0.27 deg)

    times_per_day = 1440. / resolution
    if not times_per_day == int(times_per_day):
        logging.info('''BADNESS!
        resolution_in_min must divide a day into an integer no pieces.
        in other words, resolution_in_min must be a divisor of 1440.
        valid values include 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, etc.
        the recommended value is 15.''')
    shape = (dates.size/times_per_day, times_per_day)

    dates_J2000 = dates.reshape(shape)
    sun, moon = sun.reshape(shape), moon.reshape(shape)
    target = target.reshape(shape)
    is_dark_time = is_dark_time.reshape(shape)

    save_dict = { 'dates_J2000' : dates_J2000, 'is_dark_time' : is_dark_time }
    np.savez(dark_almanac_filename, **save_dict)
    logging.info('done.')
    
    return dates_J2000, is_dark_time
    
    
# ______________________________________________________________________________

def create_almanac(ra, dec, dates_j2000, almanac_filename):
    logging.info('creating almanac' % (almanac_filename, ))
    # sys.stdout.flush()
    
    dates, sun, moon, target, dark_time = hours_observable_bruteforce(
        ra, dec, dates_j2000=dates_j2000, full_output=True)

    airmass = np.clip(np.where(target > np.radians(10.),
                               1./np.sin(target), 99.), 0., 9.)

    save_dict = {'dates_J2000': dates_j2000, 'airmass': airmass}
    np.savez(almanac_filename, **save_dict)
    logging.info('done.')

    return airmass

        
# ______________________________________________________________________________
#
# CONVENIENCE FUNCTIONS FOR RISE AND SET TIMES

# ephem does everything internally using J2000 dates

def utc_to_j2000(datetime_utc=datetime.datetime.utcnow()):
    return ephem.julian_date(datetime_utc) - 2415020.


def next_sunrise(observer=UKST_TELESCOPE, horizon=SOLAR_TWILIGHT_HORIZON):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Sun()                   # set target as Sol
    target.compute(observer)             # intialise ephem target object
    return observer.next_rising(target)  # return next rise date


def next_sunset(observer=UKST_TELESCOPE, horizon=SOLAR_TWILIGHT_HORIZON):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Sun()                   # set target as Sol
    target.compute(observer)             # intialise ephem target object
    return observer.next_setting(target) # return next set date


def next_moonrise(observer=UKST_TELESCOPE, horizon=LUNAR_TWILIGHT_HORIZON):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Moon()                  # set target as the moon
    target.compute(observer)             # intialise ephem target object
    return observer.next_rising(target)  # return next rise date


def next_moonset(observer=UKST_TELESCOPE, horizon=LUNAR_TWILIGHT_HORIZON):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Moon()                  # set target as the moon
    target.compute(observer)             # intialise ephem target object
    return observer.next_setting(target)  # return next set date


def next_riseset(RA, Dec, observer=UKST_TELESCOPE, horizon_in_degrees=0.):
    observer.horizon = np.radians(horizon_in_degrees) # radians everywhere
    target = ephem.FixedBody()             # create ephem target object
    target._ra = np.radians(RA)          # radians everywhere
    target._dec = np.radians(Dec)        # radians everywhere
    target.compute(observer)             # intialise ephem target object
    risedate = observer.next_rising(target)
    setdate = observer.next_setting(target)
    return risedate, setdate               # return rise and set dates

# ______________________________________________________________________________
