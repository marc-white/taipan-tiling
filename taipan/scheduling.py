import numpy as np
import datetime

import os
import sys
import time
import logging

import ephem

# ______________________________________________________________________________
#
# CONSTANTS

# define twilight values for when observing can start/must end

SOLAR_TWILIGHT_HORIZON = np.radians(-10.)      # radians everywhere
LUNAR_TWILIGHT_HORIZON = np.radians(0.)        # radians everywhere

# define and establish emph.Observer object for UKST-Taipan
#  all from https://en.wikipedia.org/wiki/Siding_Spring_Observatory
UKST_LATITUDE = -31.272231
UKST_LONGITUDE = +149.071233 
UKST_ELEVATION = 1165  # metres

UKST_TELESCOPE = ephem.Observer()
UKST_TELESCOPE.lat = np.radians(UKST_LATITUDE)   # radians everywhere
UKST_TELESCOPE.lon = np.radians(UKST_LONGITUDE)  # radians everywhere
UKST_TELESCOPE.elevation = UKST_ELEVATION
UKST_TELESCOPE.temp = 10.

# ______________________________________________________________________________


def airmass(ra, dec, datej2000=None, observer=UKST_TELESCOPE):
    # initialise ephem observer object with specified datetime
    if date_J2000 is not None:
        observer.date = datej2000
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
        start_date = utc_to_J2000() # Set start time to now

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

def utc_to_J2000(datetime_utc=datetime.datetime.utcnow()):
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
