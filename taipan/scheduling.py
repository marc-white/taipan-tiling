import numpy as np
import datetime

import os, sys, time

import ephem

# ______________________________________________________________________________
#
# CONSTANTS

# define twilight values for when observing can start/must end

SOLAR_TWILIGHT_HORIZON = np.radians( -10. )      # radians everywhere
LUNAR_TWILIGHT_HORIZON = np.radians( 0. )        # radians everywhere

# define and establish emph.Observer object for UKST-Taipan
#  all from https://en.wikipedia.org/wiki/Siding_Spring_Observatory
UKST_LATITUDE = -31.272231
UKST_LONGITUDE = +149.071233 
UKST_ELEVATION = 1165 # metres

UKST_TELESCOPE = ephem.Observer()
UKST_TELESCOPE.lat = np.radians( UKST_LATITUDE )  # radians everywhere
UKST_TELESCOPE.lon = np.radians( UKST_LONGITUDE ) # radians everywhere
UKST_TELESCOPE.elevation = UKST_ELEVATION
UKST_TELESCOPE.temp = 10.

# ______________________________________________________________________________

def airmass( RA, Dec, date_J2000=None, observer=UKST_TELESCOPE ):
    # initialise ephem observer object with specified datetime
    if not ( date_J2000 is None ):
        observer.date = date_J2000
    # initialise ephem target object with specified coordinates
    target = ephem.FixedBody()             # create ephem target object
    target._ra = np.radians( RA )          # radians everywhere 
    target._dec = np.radians( Dec )        # radians everywhere 
    target.compute( observer )             # intialise ephem target object

    # return airmass if target is accessible with UKST
    if np.degrees( target.alt ) > 5 :
        return 1./np.sin( target.alt )
    else :
        return 99.

# ______________________________________________________________________________

def next_dark_period( UKST=UKST_TELESCOPE ):
    _date = UKST.date
    # determine next sun and moon rise and set times from now.
    sunrise, sunset = next_sunrise( UKST ), next_sunset( UKST )
    moonrise,moonset=next_moonrise( UKST ),next_moonset( UKST )
    # are the sun and moon up now? (will they set before they rise?)
    is_sun_up = sunset < sunrise
    is_moon_up = moonset < moonrise
    while ( is_sun_up or is_moon_up ) :
        until = UKST.date + 1./1440 # advance at least 1 min into the future
        # advance time to when sun/moon have set (whichever is furthest ahead)
        if is_sun_up :  until = max( until,  sunset ) 
        if is_moon_up : until = max( until, moonset )
        UKST.date = until
        # update rise/set times as necessary
        if sunrise <= until  : sunrise  = next_sunrise( UKST )
        if sunset <= until   : sunset   = next_sunset(  UKST )
        if moonrise <= until : moonrise = next_moonrise(UKST )
        if moonset <= until  : moonset  = next_moonset( UKST )
        # how about now? are the sun and moon up now? 
        is_sun_up = sunset < sunrise
        is_moon_up = moonset < moonrise
    darkstart = UKST.date
    # dark time ends when either the sun or the moon rises
    darkend = min( sunrise, moonrise )
    print 'next dark period: %s --> %s' % ( darkstart, darkend )
    # return UKST.date to its original value
    UKST.date = _date
    return darkstart, darkend        

# ______________________________________________________________________________

def hours_observable( ra, dec,
                      start_date_J2000=None, end_date_J2000=None,
                      observing_period_in_days=365.25/2.,
                      minimum_airmass=2.0, verbose=0 ):

    if start_date_J2000 is None :
        start_date_J2000 = utc_to_J2000()

    # determine end of observing period if necessary
    if end_date_J2000 is None :
        UKST_TELESCOPE.date = start_date_J2000  # pyephem accepts datetime object
        end_date_J2000 = UKST_TELESCOPE.date + observing_period_in_days
        UKST_TELESCOPE.date = end_date_J2000
        
    # initialise ephem fixed body object for target
    target = ephem.FixedBody()
    target._ra = np.radians( ra ) # radians everywhere
    target._dec = np.radians( dec ) # radians everywhere

    # define angle corresponding to minimum_airmass and twilight_horizon
    observable = np.arcsin( 1./ minimum_airmass ) # radians everywhere

    # hack around weird pyephem behavior near target's zenith
    zenith = np.abs( np.abs( UKST_TELESCOPE.lat ) + np.radians( dec - 90. ) )
#    print observable, np.degrees( observable ), 1./np.sin( observable )
#    print zenith, np.degrees( zenith ), 1./np.sin( zenith )
    observable = np.clip( observable, 0., zenith - np.radians(.1) )
#    print observable, np.degrees( observable ), 1./np.sin( observable )
    
    def next_rise( ):
        UKST_TELESCOPE.horizon = observable
        target.compute( UKST_TELESCOPE )
        return UKST_TELESCOPE.next_rising( target )
    def next_set( ):
        UKST_TELESCOPE.horizon = observable
        target.compute( UKST_TELESCOPE )
        return UKST_TELESCOPE.next_setting( target )

        
    # from start of observing, compute next rise/set dates 
    UKST_TELESCOPE.date = start_date_J2000    # pyephem accepts datetime object
    sunrise = next_sunrise( UKST_TELESCOPE )
    sunset = next_sunset( UKST_TELESCOPE )
    moonrise = next_moonrise( UKST_TELESCOPE )
    moonset = next_moonset( UKST_TELESCOPE )

    # given its Dec, check whether target actually ever sets, or is circumpolar
    target.compute( UKST_TELESCOPE )
    always_up_declination = -np.pi/2. + np.abs(UKST_TELESCOPE.lat) - observable
                                            # radians everywhere
    target_always_up = dec < np.degrees( always_up_declination )
    
#    print airmass( ra, dec, UKST_TELESCOPE.date )
#    print 'target_always_up', target_always_up

    # and then compute target rise/set times if necessary
    if target_always_up :
        targetrise, targetset = UKST_TELESCOPE.date, end_date_J2000
    else :
        targetrise = next_rise( )
        targetset = next_set( )

    # and now step forward through the observing period
    hours_total = 0.
    while UKST_TELESCOPE.date < end_date_J2000 :
        
        # if sun rises before it sets then it is nighttime (good)
        is_night_time = sunrise < sunset 
        # if moon rises before it sets then it is darktime (good)
        is_dark_time = moonrise < moonset
        # if target sets before it rises then it is visible (good)
        is_target_up = ( targetset < targetrise ) #or target_always_up

        # if target is observable record how long it is observable for
        if is_night_time and is_dark_time and is_target_up :

            # determine when target becomes unobservable
            until = min( sunrise, moonrise, targetset, end_date_J2000 )

            # record amount of observable time in hours
            hours_tonight = ( until - UKST_TELESCOPE.date ) * 24.
            # add this to the total observable time in the full period
            hours_total += hours_tonight

            if verbose > 1 :
                print ( UKST_TELESCOPE.date, '-->', until,
                        '(%.3f hr)' % hours_tonight )
            
        else :
            # if target is not observable, determine when it might next be
            # advance by a minimum of one minute to sidestep pyephem oddnesses
            until = UKST_TELESCOPE.date + 1./1440
            if not is_night_time : until = max( until, sunset )
            if not is_dark_time : until = max( until, moonset )
            if not is_target_up : until = max( until, targetrise )

        if UKST_TELESCOPE.date == until :
            print 'oh no!', UKST_TELESCOPE.date
            print
            until = UKST_TELESCOPE.date
            print until, float( until )
            print is_night_time ; until = max( until, sunset ) 
            print until, float( until )
            print is_dark_time ;  until = max( until, moonset )
            print until, float( until )
            print is_target_up ; until = max( until, targetrise )
            print until, float( until )
            print
            print 'currently', UKST_TELESCOPE.date, float( UKST_TELESCOPE.date )
            print 'sunrise', sunrise, sunrise - UKST_TELESCOPE.date
            print 'sunset', sunset, sunset- UKST_TELESCOPE.date
            print 'is_night_time', is_night_time, sunset-UKST_TELESCOPE.date
            print 'moonrise', moonrise, moonrise - UKST_TELESCOPE.date
            print 'moonset', moonset, moonset - UKST_TELESCOPE.date
            print 'is_dark_time', is_dark_time, moonset-UKST_TELESCOPE.date
            print 'targetrise', targetrise, targetrise - UKST_TELESCOPE.date
            print 'targetset', targetset, targetset - UKST_TELESCOPE.date
            print 'is_target_up', is_target_up, targetrise - UKST_TELESCOPE.date
            print 'checks', targetset <= until, targetset - until, targetset - next_set()
            print 'until', until, until - UKST_TELESCOPE.date
            print 'target_always_up', target_always_up
#            time.sleep( 10. )

            
        # advance time to next change in circumstances
        UKST_TELESCOPE.date = min( until, end_date_J2000 )
        
        # update any/all event times that need updating
        if sunrise <= until : sunrise = next_sunrise( observer=UKST_TELESCOPE )
        if sunset <= until : sunset = next_sunset( observer=UKST_TELESCOPE )
        if moonrise <= until : moonrise = next_moonrise(observer=UKST_TELESCOPE )
        if moonset <= until : moonset = next_moonset( observer=UKST_TELESCOPE )
        if targetrise <= until and not target_always_up :
            targetrise = next_rise()
        if targetset <= until and not target_always_up :
            targetset = next_set()

    # all done!
    return hours_total
        
        
# ______________________________________________________________________________

def hours_observable_bruteforce( ra, dec, 
                                 start_date_J2000=None, end_date_J2000=None,
                                 observing_period_in_days=365.25/2.,
                                 minimum_airmass=2.0, verbose=0,
                                 resolution_in_min=15., full_output=False,
                                 dates_J2000=None ):

    # define horizons for observability computation
    observable = np.arcsin( 1./ minimum_airmass ) # radians everywhere
    solar_horizon = SOLAR_TWILIGHT_HORIZON - np.radians( 0.24 )
    lunar_horizon = LUNAR_TWILIGHT_HORIZON - np.radians( 0.27 )
    # remember that the sun and moon both have a size! (0.24 and 0.27 deg)
    
    # initialise ephem fixed body object for pointing
    target = ephem.FixedBody()
    target._ra = np.radians( ra ) # radians everywhere
    target._dec = np.radians( dec ) # radians everywhere

    # instantiate ephem sun/moon objects
    sun, moon = ephem.Sun(), ephem.Moon()

    # define time grid if not given 
    if dates_J2000 is None :
        # initialise ephem observer object for UKST @ SSO
        UKST_TELESCOPE.date = start_date_J2000

        if end_date_J2000 is None :
            end_date_J2000 = UKST_TELESCOPE.date + observing_period_in_days
        else :
            observing_period_in_days = end_date_J2000 - UKST_TELESCOPE.date
        dates_J2000 = ( UKST_TELESCOPE.date +
            np.arange( 0, observing_period_in_days, resolution_in_min/1440. ) )
                                     
        
    # and finally the actual computation
    time_total = 0.
    sol_alt, lun_alt, target_alt, dark_time = [], [], [], []
    for d in dates_J2000.ravel() :
        UKST_TELESCOPE.date = d
        sun.compute( UKST_TELESCOPE ) 
        moon.compute( UKST_TELESCOPE )
        target.compute( UKST_TELESCOPE )

        can_observe = ( sun.alt < solar_horizon ) & ( moon.alt < lunar_horizon )
        if full_output :
            sol_alt.append( sun.alt )
            lun_alt.append( moon.alt )
            target_alt.append( target.alt )
            dark_time.append( can_observe )
        else :
            if ( can_observe & ( target.alt > observable ) ) :
                time_total += resolution_in_min
            
    if full_output :
        # convert results to numpy arrays
        sol_alt = np.array( sol_alt ).reshape( dates_J2000.shape ) 
        lun_alt = np.array( lun_alt ).reshape( dates_J2000.shape ) 
        target_alt=np.array(target_alt).reshape(dates_J2000.shape ) 
        dark_time=np.array( dark_time ).reshape( dates_J2000.shape )
        return dates_J2000, sol_alt, lun_alt, target_alt, dark_time
    else :
        return time_total/60.


# ______________________________________________________________________________

def hours_observable_from_almanac( almanac_filename,
                                   dates_J2000, is_dark_time,
                                   ra=None, dec=None,
                                   start_date_J2000=None, end_date_J2000=None,
                                   observing_period_in_days=365.25/2.,
                                   minimum_airmass=2.0 ):

    # load almanac if it exists
    if os.path.exists( almanac_filename ):
        with np.load( almanac_filename ) as save_dict :
            ddates, airmass = save_dict['dates_J2000'], save_dict['airmass']
        # check that almanac follows same date grid as input is_dark_time
        if not np.allclose( dates_J2000, ddates ) :
            print '''\n\nBADNESS!
            dates found in almanac with filename %s do not match those given
            to hours_observable_from_almanac. this almanac must be remade.
            existing almanac is being removed and reconstructed.
            ''' % almanac_filename
            os.remove( almanac_filename )
            # discard this almanac if its not right

    # create almanac if necessary
    if not os.path.exists( almanac_filename ) :
        if ( ra is None or dec is None ):
            print '''\n\nBADNESS!
            almanac with filename %s does not exist.
            call hours_observable_from_almanac using ra, dec keywords to
            create a new almanac.  dying gracelessly now.
            ''' % almanac_filename
            exit()
        airmass = create_almanac( ra, dec, dates_J2000, almanac_filename )

    if end_date_J2000 is None :
        end_date_J2000 = UKST_TELESCOPE.date + observing_period_in_days

    # identify when target is observable during observing period
    fair_game = ( is_dark_time 
                  & ( start_date_J2000 <= dates_J2000 )
                  & ( dates_J2000 <= end_date_J2000 ) )

#    better_entries = np.sum( fair_game & ( airmass <= minimum_airmass ) )
    better_entries = np.interp( minimum_airmass,
                                np.sort( airmass[ fair_game ] ),
                                np.arange( fair_game.sum() ) )    
    # this gives the number of dark time entries within the specified time
    # period where the airmass in the almanac is less than the minimum.

    # use dates grid to determine resolution in hours
    first_date = dates_J2000[ np.unravel_index( 0, dates_J2000.shape ) ]
    last_date = dates_J2000[ np.unravel_index( dates_J2000.size-1,
                                               dates_J2000.shape ) ]
    resolution_in_hours = ( last_date - first_date ) * 24. / dates_J2000.size
    better_hours = better_entries * resolution_in_hours

    return better_hours
        
    
# ______________________________________________________________________________

def create_dark_almanac( dark_almanac_filename,
                         start_date_J2000=None, end_date_J2000=None,
                         observing_period_in_days=int(4*365.25),
                         resolution_in_min=15. ):

    print
    print 'creating dark almanac', dark_almanac_filename, '...',
    sys.stdout.flush()
    
    dates, sun, moon, target, is_dark_time = hours_observable_bruteforce(
        ra=0., dec=0., 
        start_date_J2000=start_date_J2000, end_date_J2000=end_date_J2000,
        observing_period_in_days=observing_period_in_days,
        resolution_in_min=resolution_in_min, full_output=True )

    solar_horizon = SOLAR_TWILIGHT_HORIZON - np.radians( 0.24 )
    lunar_horizon = LUNAR_TWILIGHT_HORIZON - np.radians( 0.27 )
    # remember that the sun and moon both have a size! (0.24 and 0.27 deg)

    times_per_day = 1440./resolution_in_min
    if not times_per_day == int( times_per_day ):
        print '''\n\nBADNESS!
        resolution_in_min must divide a day into an integer no pieces.
        in other words, resolution_in_min must be a divisor of 1440.
        valid values include 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, etc.
        the recommended value is 15.'''
    shape = ( dates.size/times_per_day, times_per_day )

    dates_J2000 = dates.reshape( shape )
    sun, moon = sun.reshape( shape ), moon.reshape( shape )
    target = target.reshape( shape )
    is_dark_time = is_dark_time.reshape( shape )

    save_dict = { 'dates_J2000' : dates_J2000, 'is_dark_time' : is_dark_time }
    np.savez( dark_almanac_filename, **save_dict )
    print 'done.'
    
    return dates_J2000, is_dark_time
    
    
# ______________________________________________________________________________

def create_almanac( ra, dec, dates_J2000, almanac_filename ):
    print
    print 'creating almanac', almanac_filename,
    sys.stdout.flush()
    
    dates, sun, moon, target, dark_time = hours_observable_bruteforce(
        ra, dec, dates_J2000=dates_J2000, full_output=True )

    airmass = np.clip( np.where( target > np.radians( 10. ),
                                 1./np.sin( target ), 99. ), 0., 9. )

    save_dict = { 'dates_J2000' : dates_J2000, 'airmass' : airmass }
    np.savez( almanac_filename, **save_dict )
    print 'done.'

    return airmass

        
# ______________________________________________________________________________
#
# CONVENIENCE FUNCTIONS FOR RISE AND SET TIMES

# ephem does everything internally using J2000 dates

def utc_to_J2000( datetime_utc=datetime.datetime.utcnow() ):
    return ephem.julian_date( datetime_utc ) - 2415020.

def next_sunrise( observer=UKST_TELESCOPE, horizon=SOLAR_TWILIGHT_HORIZON ):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Sun()                   # set target as Sol
    target.compute( observer )             # intialise ephem target object
    return observer.next_rising( target )  # return next rise date

def next_sunset( observer=UKST_TELESCOPE, horizon=SOLAR_TWILIGHT_HORIZON ):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Sun()                   # set target as Sol
    target.compute( observer )             # intialise ephem target object
    return observer.next_setting( target ) # return next set date

def next_moonrise( observer=UKST_TELESCOPE, horizon=LUNAR_TWILIGHT_HORIZON ):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Moon()                  # set target as the moon
    target.compute( observer )             # intialise ephem target object
    return observer.next_rising( target )  # return next rise date

def next_moonset( observer=UKST_TELESCOPE, horizon=LUNAR_TWILIGHT_HORIZON ):
    observer.horizon = horizon             # specify horizon for rise/set
    target = ephem.Moon()                  # set target as the moon
    target.compute( observer )             # intialise ephem target object
    return observer.next_setting( target ) # return next set date

def next_riseset( RA, Dec, observer=UKST_TELESCOPE, horizon_in_degrees=0. ):
    observer.horizon = np.radians( horizon_in_degrees ) # radians everywhere 
    target = ephem.FixedBody()             # create ephem target object
    target._ra = np.radians( RA )          # radians everywhere 
    target._dec = np.radians( Dec )        # radians everywhere 
    target.compute( observer )             # intialise ephem target object
    risedate = observer.next_rising( target )
    setdate = observer.next_setting( target )
    return risedate, setdate               # return rise and set dates

# ______________________________________________________________________________

