import datetime
from taipan.scheduling import Almanac, DarkAlmanac
import numpy as np

if __name__ == "__main__":
    for year in [2013, 2014, 2015]:
        for month in range(1,11):
            for day in [1,5,10]:

                start_date = datetime.date(year, month, day)
                start_datetime = datetime.datetime(year, month, day+1, 12, 0)
                end_date = datetime.date(year, month, day + 10)
                ra = 120
                dec = -60

                al = Almanac(ra, dec, start_date, end_date=end_date)
                dal = DarkAlmanac(start_date, end_date=end_date)

                print(start_date, end_date, start_datetime)
                try:
                    a0 = np.round(al.hours_observable(start_datetime,
                                                      exclude_dark_time=False,
                                                      exclude_grey_time=False,
                                                      dark_almanac=dal), 8)
                    a1 = np.round(al.hours_observable(start_datetime,
                                                      exclude_dark_time=False,
                                                      exclude_grey_time=True,
                                                      dark_almanac=dal), 8)
                    a2 = np.round(al.hours_observable(start_datetime,
                                                      exclude_dark_time=True,
                                                      exclude_grey_time=False,
                                                      dark_almanac=dal), 8)
                except RuntimeError:
                    print(e)
                    continue
                print(a0, a1, a2, a1 + a2)
                assert a0 == a1 + a2
