#!python

# Global definitions for TAIPAN tiling code

# Created: Marc White, 2015-09-14
# Last modified: Marc White, 2016-01-19

"""Core functionality for taipan tiling/scheduling module

This module contains the basic functionality for the taipan tiling/scheduling
code. It defines the basic :class:`TaipanTarget` and :class:`TaipanTile`
objects (both of which are subclasses of a generic :class:`TaipanPoint` class),
and associated functionality/cosntants.
"""

import csv
import logging
import math
import operator
import os
import random
import string
import datetime

import numpy as np
from matplotlib.cbook import flatten
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree as skKDTree

#XXX
#import matplotlib.pyplot as plt

# -------
# CONSTANTS
# -------

# Code marking & JSON write-out
VERSION = 0.9
""":obj:`float`: Code version"""
NAME = 'TaipanSurveyTiler'
""":obj:`str`: Name for writing to tile definition files"""
SOFTWARE_TYPE = 'TaipanSurvey.Tiler'
""":obj:`str`: Software type for writing to tile definition files"""
SURVEY = 'Taipan|Taipan'
""":obj:`str`: Survey string for writing to tile definition files"""
TILE_CONFIG_FILE_VERSION = 2
""":obj:`float`:Version of tile definition file specification in use"""
INSTRUMENT_NAME = 'AAO.Taipan'
""":obj:`str`: Instrument name"""
FILE_PURPOSE = 'CoD.Testing'
""":obj:`str`: Purpose for generation of tiles (for tile definition file)"""
JSON_DTFORMAT_NAIVE = r'%Y-%m-%dT%H:%M:%S'
""":obj:`str`: Format for naive datetime strings in the tile definition file"""
JSON_DTFORMAT_TZ = r'%Y-%m-%dT%H:%M:%S%z'
""":obj:`str`: Format for timezone-d datetime strings in the tile definition file"""

# Computational break-even points
# Number of targets needed to make it worth doing difficulty calculations
# with a KDTREE
BREAKEVEN_KDTREE = 50
""":obj:`int`: Number of points after which KDTree distance calculation is done

Using a :any:`scipy.spatial.cKDTree` for computing the distances between points 
on a unit 
sphere (i.e. the distances/grouping of targets/tiles in the survey) is 
vastly more efficient for large numbers of targets. However, for only a few
targets, it is faster to just brute-force the distances between all points and
compute target distances/clustering directly. This value is an initial guess
for the break-even point where brute-force and :any:`scipy.spatial.cKDTree`
approaches takes
approximately the same wall time (note this hasn't actually been tested at 
all, it is a complete guess).
"""


# Instrument variables
TARGET_PER_TILE = 120
""":obj:`int`: The maximum number of science targets to be assigned per tile"""
STANDARDS_PER_TILE = 10
""":obj:`int`: The maximum number of standard targets to be assigned per tile"""
STANDARDS_PER_TILE_MIN = 5
""":obj:`int`: The minimum number of standard targets a tile must have to be 
valid"""
SKY_PER_TILE = 20
""":obj:`int`: The maximum number of sky targets to be assigned per tile"""
SKY_PER_TILE_MIN = 20
""":obj:`int`: The minimum number of sky targets a tile must have to be 
valid"""
GUIDES_PER_TILE = 9
""":obj:`int`: The maximum number of guide targets to be assigned per tile"""
GUIDES_PER_TILE_MIN = 3
""":obj:`int`: The minimum number of standard targets a tile must have to be 
valid"""
FIBRES_PER_TILE = (TARGET_PER_TILE + STANDARDS_PER_TILE 
    + SKY_PER_TILE + GUIDES_PER_TILE)
""":obj:`int`: The number of fibres per tile, based on the maximum numbers of 
each target type which may be assigned.

This is computed at module load as 
``FIBRES_PER_TILE = (TARGET_PER_TILE + STANDARDS_PER_TILE 
+ SKY_PER_TILE + GUIDES_PER_TILE)``.
"""
INSTALLED_FIBRES = 159
""":obj:`int`: The number of fibres installed in the instrument.

This value is simply a number. 

Raises
------
Exception
    Raised if :any:`FIBRES_NORMAL` + 
    :any:`FIBRES_GUIDE` does not equal :any:`INSTALLED_FIBRES`. Note this check
    is only performed at module load.
Exception
    An :any:`Exception` will be raised if the
    module detects that :any:`INSTALLED_FIBRES` does not equal 
    :any:`FIBRES_PER_TILE`.
"""
if FIBRES_PER_TILE < INSTALLED_FIBRES:
    raise Exception('WARNING: Not all fibres will be utilised. '
                    'Check the fibre constants in taipan/core.py.')
if FIBRES_PER_TILE > INSTALLED_FIBRES:
    raise Exception('You are attempting to assign more fibres than'
                    'are currently installed. Check the fibre '
                    'constants in taipan/core.py.')

# Fibre positioning
# Comment out lines to render that fibre inoperable

BUGPOS_MM = {1: (-129, 87.4),
             2: (-133.4, 70.9),
             3: (-137.8, 54.5),
             4: (-142.2, 38.1),
             5: (-146.6, 21.7),
             6: (-151, 5.3),
             7: (-108.2, 108.2),
             8: (-112.6, 91.8),
             9: (-117, 75.3),
             10: (-121.4, 58.9),
             11: (-125.8, 42.5),
             12: (-130.2, 26.1),
             13: (-134.6, 9.7),
             14: (-139, -6.8),
             15: (-143.4, -23.2),
             16: (-147.8, -39.6),
             17: (-87.4, 129),
             18: (-91.8, 112.6),
             19: (-96.2, 96.2),
             20: (-100.6, 79.7),
             21: (-105, 63.3),
             22: (-109.4, 46.9),
             23: (-113.8, 30.5),
             24: (-118.2, 14.1),
             25: (-122.6, -2.4),
             26: (-127, -18.8),
             27: (-131.4, -35.2),
             28: (-135.8, -51.6),
             29: (-140.2, -68),
             30: (-70.9, 133.4),
             31: (-75.3, 117),
             32: (-79.7, 100.6),
             33: (-84.1, 84.1),
             34: (-88.5, 67.7),
             35: (-92.9, 51.3),
             36: (-97.3, 34.9),
             37: (-101.7, 18.5),
             38: (-106.1, 2),
             39: (-110.5, -14.4),
             40: (-114.9, -30.8),
             41: (-119.3, -47.2),
             42: (-123.7, -63.6),
             43: (-128.1, -80.1),
             44: (-54.5, 137.8),
             45: (-58.9, 121.4),
             46: (-63.3, 105),
             47: (-67.7, 88.5),
             48: (-72.1, 72.1),
             49: (-76.5, 55.7),
             50: (-80.9, 39.3),
             51: (-85.3, 22.9),
             52: (-89.7, 6.4),
             53: (-94.1, -10),
             54: (-98.5, -26.4),
             55: (-102.9, -42.8),
             56: (-107.3, -59.2),
             57: (-111.7, -75.7),
             58: (-116.1, -92.1),
             59: (-38.1, 142.2),
             60: (-42.5, 125.8),
             61: (-46.9, 109.4),
             62: (-51.3, 92.9),
             63: (-55.7, 76.5),
             64: (-60.1, 60.1),
             65: (-64.5, 43.7),
             66: (-68.9, 27.3),
             67: (-73.3, 10.8),
             68: (-77.7, -5.6),
             69: (-82.1, -22),
             70: (-86.5, -38.4),
             71: (-90.9, -54.8),
             72: (-95.3, -71.3),
             73: (-99.7, -87.7),
             74: (-104.1, -104.1),
             75: (-21.7, 146.6),
             76: (-26.1, 130.2),
             77: (-30.5, 113.8),
             78: (-34.9, 97.3),
             79: (-39.3, 80.9),
             80: (-43.7, 64.5),
             81: (-48.1, 48.1),
             82: (-52.5, 31.7),
             83: (-56.9, 15.2),
             84: (-61.3, -1.2),
             85: (-65.7, -17.6),
             86: (-70.1, -34),
             87: (-74.5, -50.4),
             88: (-78.9, -66.9),
             89: (-83.3, -83.3),
             90: (-87.7, -99.7),
             91: (-92.1, -116.1),
             92: (-5.3, 151),
             93: (-9.7, 134.6),
             94: (-14.1, 118.2),
             95: (-18.5, 101.7),
             96: (-22.9, 85.3),
             97: (-27.3, 68.9),
             98: (-31.7, 52.5),
             99: (-36.1, 36.1),
             100: (-40.5, 19.6),
             101: (-44.9, 3.2),
             102: (-49.3, -13.2),
             103: (-53.7, -29.6),
             104: (-58.1, -46),
             105: (-62.5, -62.5),
             106: (-66.9, -78.9),
             107: (-71.3, -95.3),
             108: (-75.7, -111.7),
             109: (-80.1, -128.1),
             110: (6.8, 139),
             111: (2.4, 122.6),
             112: (-2, 106.1),
             113: (-6.4, 89.7),
             114: (-10.8, 73.3),
             115: (-15.2, 56.9),
             116: (-19.6, 40.5),
             117: (-24, 24),
             118: (-28.4, 7.6),
             119: (-32.8, -8.8),
             120: (-37.2, -25.2),
             121: (-41.6, -41.6),
             122: (-46, -58.1),
             123: (-50.4, -74.5),
             124: (-54.8, -90.9),
             125: (-59.2, -107.3),
             126: (-63.6, -123.7),
             127: (-68, -140.2),
             128: (23.2, 143.4),
             129: (18.8, 127),
             130: (14.4, 110.5),
             131: (10, 94.1),
             132: (5.6, 77.7),
             133: (1.2, 61.3),
             134: (-3.2, 44.9),
             135: (-7.6, 28.4),
             136: (-12, 12),
             137: (-16.4, -4.4),
             138: (-20.8, -20.8),
             139: (-25.2, -37.2),
             140: (-29.6, -53.7),
             141: (-34, -70.1),
             142: (-38.4, -86.5),
             143: (-42.8, -102.9),
             144: (-47.2, -119.3),
             145: (-51.6, -135.8),
             146: (39.6, 147.8),
             147: (35.2, 131.4),
             148: (30.8, 114.9),
             149: (26.4, 98.5),
             150: (22, 82.1),
             151: (17.6, 65.7),
             152: (13.2, 49.3),
             153: (8.8, 32.8),
             154: (4.4, 16.4),
             155: (0, 0),
             156: (-4.4, -16.4),
             157: (-8.8, -32.8),
             158: (-13.2, -49.3),
             159: (-17.6, -65.7),
             160: (-22, -82.1),
             161: (-26.4, -98.5),
             162: (-30.8, -114.9),
             163: (-35.2, -131.4),
             164: (-39.6, -147.8),
             165: (51.6, 135.8),
             166: (47.2, 119.3),
             167: (42.8, 102.9),
             168: (38.4, 86.5),
             169: (34, 70.1),
             170: (29.6, 53.7),
             171: (25.2, 37.2),
             172: (20.8, 20.8),
             173: (16.4, 4.4),
             174: (12, -12),
             175: (7.6, -28.4),
             176: (3.2, -44.9),
             177: (-1.2, -61.3),
             178: (-5.6, -77.7),
             179: (-10, -94.1),
             180: (-14.4, -110.5),
             181: (-18.8, -127),
             182: (-23.2, -143.4),
             183: (68, 140.2),
             184: (63.6, 123.7),
             185: (59.2, 107.3),
             186: (54.8, 90.9),
             187: (50.4, 74.5),
             188: (46, 58.1),
             189: (41.6, 41.6),
             190: (37.2, 25.2),
             191: (32.8, 8.8),
             192: (28.4, -7.6),
             193: (24, -24),
             194: (19.6, -40.5),
             195: (15.2, -56.9),
             196: (10.8, -73.3),
             197: (6.4, -89.7),
             198: (2, -106.1),
             199: (-2.4, -122.6),
             200: (-6.8, -139),
             201: (80.1, 128.1),
             202: (75.7, 111.7),
             203: (71.3, 95.3),
             204: (66.9, 78.9),
             205: (62.5, 62.5),
             206: (58.1, 46),
             207: (53.7, 29.6),
             208: (49.3, 13.2),
             209: (44.9, -3.2),
             210: (40.5, -19.6),
             211: (36.1, -36.1),
             212: (31.7, -52.5),
             213: (27.3, -68.9),
             214: (22.9, -85.3),
             215: (18.5, -101.7),
             216: (14.1, -118.2),
             217: (9.7, -134.6),
             218: (5.3, -151),
             219: (92.1, 116.1),
             220: (87.7, 99.7),
             221: (83.3, 83.3),
             222: (78.9, 66.9),
             223: (74.5, 50.4),
             224: (70.1, 34),
             225: (65.7, 17.6),
             226: (61.3, 1.2),
             227: (56.9, -15.2),
             228: (52.5, -31.7),
             229: (48.1, -48.1),
             230: (43.7, -64.5),
             231: (39.3, -80.9),
             232: (34.9, -97.3),
             233: (30.5, -113.8),
             234: (26.1, -130.2),
             235: (21.7, -146.6),
             236: (104.1, 104.1),
             237: (99.7, 87.7),
             238: (95.3, 71.3),
             239: (90.9, 54.8),
             240: (86.5, 38.4),
             241: (82.1, 22),
             242: (77.7, 5.6),
             243: (73.3, -10.8),
             244: (68.9, -27.3),
             245: (64.5, -43.7),
             246: (60.1, -60.1),
             247: (55.7, -76.5),
             248: (51.3, -92.9),
             249: (46.9, -109.4),
             250: (42.5, -125.8),
             251: (38.1, -142.2),
             252: (116.1, 92.1),
             253: (111.7, 75.7),
             254: (107.3, 59.2),
             255: (102.9, 42.8),
             256: (98.5, 26.4),
             257: (94.1, 10),
             258: (89.7, -6.4),
             259: (85.3, -22.9),
             260: (80.9, -39.3),
             261: (76.5, -55.7),
             262: (72.1, -72.1),
             263: (67.7, -88.5),
             264: (63.3, -105),
             265: (58.9, -121.4),
             266: (54.5, -137.8),
             267: (128.1, 80.1),
             268: (123.7, 63.6),
             269: (119.3, 47.2),
             270: (114.9, 30.8),
             271: (110.5, 14.4),
             272: (106.1, -2),
             273: (101.7, -18.5),
             274: (97.3, -34.9),
             275: (92.9, -51.3),
             276: (88.5, -67.7),
             277: (84.1, -84.1),
             278: (79.7, -100.6),
             279: (75.3, -117),
             280: (70.9, -133.4),
             281: (140.2, 68),
             282: (135.8, 51.6),
             283: (131.4, 35.2),
             284: (127, 18.8),
             285: (122.6, 2.4),
             286: (118.2, -14.1),
             287: (113.8, -30.5),
             288: (109.4, -46.9),
             289: (105, -63.3),
             290: (100.6, -79.7),
             291: (96.2, -96.2),
             292: (91.8, -112.6),
             293: (87.4, -129),
             294: (147.8, 39.6),
             295: (143.4, 23.2),
             296: (139, 6.8),
             297: (134.6, -9.7),
             298: (130.2, -26.1),
             299: (125.8, -42.5),
             300: (121.4, -58.9),
             301: (117, -75.3),
             302: (112.6, -91.8),
             303: (108.2, -108.2),
             304: (151, -5.3),
             305: (146.6, -21.7),
             306: (142.2, -38.1),
             307: (137.8, -54.5),
             308: (133.4, -70.9),
             309: (129, -87.4),
             }
""":obj:`dict` of :obj:`int`, (:obj:`float`, :obj:`float`) mappings: Home 
positions of the Taipan fibres in plate millimetres"""

BUGPOS_MM_ORIG = {
             1: (-129.0, 87.4),
             2: (-133.4, 70.9),
             3: (-137.8, 54.5),
             4: (-142.2, 38.1),
             5: (-146.6, 21.7),
             6: (-87.4, 129.0),
             7: (-91.8, 112.6),
             8: (-96.2, 96.2),
             9: (-100.6, 79.7),
             10: (-105.0, 63.3),
             11: (-109.4, 46.9),
             12: (-113.8, 30.5),
             13: (-118.2, 14.1),
             14: (-122.6, -2.4),
             15: (-127.0, -18.8),
             16: (-131.4, -35.2),
             17: (-135.8, -51.6),
             18: (-54.5, 137.8),
             19: (-58.9, 121.4),
             20: (-63.3, 105.0),
             21: (-67.7, 88.5),
             22: (-72.1, 72.1),
             23: (-76.5, 55.7),
             24: (-80.9, 39.3),
             25: (-85.3, 22.9),
             26: (-89.7, 6.4),
             27: (-94.1, -10.0),
             28: (-98.5, -26.4),
             29: (-102.9, -42.8),
             30: (-107.3, -59.2),
             31: (-111.7, -75.7),
             32: (-21.7, 146.6),
             33: (-26.1, 130.2),
             34: (-30.5, 113.8),
             35: (-34.9, 97.3),
             36: (-39.3, 80.9),
             37: (-43.7, 64.5),
             38: (-48.1, 48.1),
             39: (-52.5, 31.7),
             40: (-56.9, 15.2),
             41: (-61.3, -1.2),
             42: (-65.7, -17.6),
             43: (-70.1, -34.0),
             44: (-74.5, -50.4),
             45: (-78.9, -66.9),
             46: (-83.3, -83.3),
             47: (-87.7, -99.7),
             48: (-63.6, -123.7),
             49: (-59.2, -107.3),
             50: (-54.8, -90.9),
             51: (-50.4, -74.5),
             52: (-46.0, -58.1),
             53: (-41.6, -41.6),
             54: (-37.2, -25.2),
             55: (-32.8, -8.8),
             56: (-28.4, 7.6),
             57: (-24.0, 24.0),
             58: (-19.6, 40.5),
             59: (-15.2, 56.9),
             60: (-10.8, 73.3),
             61: (-6.4, 89.7),
             62: (-2.0, 106.1),
             63: (-39.6, -147.8),
             64: (-35.2, -131.4),
             65: (-30.8, -114.9),
             66: (-26.4, -98.5),
             67: (-22.0, -82.1),
             68: (-17.6, -65.7),
             69: (-13.2, -49.3),
             70: (-8.8, -32.8),
             71: (-4.4, -16.4),
             72: (0.0, 0.0),
             73: (4.4, 16.4),
             74: (8.8, 32.8),
             75: (13.2, 49.3),
             76: (17.6, 65.7),
             77: (22.0, 82.1),
             78: (26.4, 98.5),
             79: (-6.8, -139.0),
             80: (-2.4, -122.6),
             81: (2.0, -106.1),
             82: (6.4, -89.7),
             83: (10.8, -73.3),
             84: (15.2, -56.9),
             85: (19.6, -40.5),
             86: (24.0, -24.0),
             87: (28.4, -7.6),
             88: (32.8, 8.8),
             89: (37.2, 25.2),
             90: (41.6, 41.6),
             91: (46.0, 58.1),
             92: (50.4, 74.5),
             93: (54.8, 90.9),
             94: (59.2, 107.3),
             95: (26.1, -130.2),
             96: (30.5, -113.8),
             97: (34.9, -97.3),
             98: (39.3, -80.9),
             99: (43.7, -64.5),
             100: (48.1, -48.1),
             101: (52.5, -31.7),
             102: (56.9, -15.2),
             103: (61.3, 1.2),
             104: (65.7, 17.6),
             105: (70.1, 34.0),
             106: (74.5, 50.4),
             107: (78.9, 66.9),
             108: (83.3, 83.3),
             109: (2.4, 122.6),
             110: (30.8, 114.9),
             111: (35.2, 131.4),
             112: (63.6, 123.7),
             113: (68.0, 140.2),
             114: (92.1, 116.1),
             115: (87.7, 99.7),
             116: (58.9, -121.4),
             117: (63.3, -105.0),
             118: (67.7, -88.5),
             119: (72.1, -72.1),
             120: (76.5, -55.7),
             121: (80.9, -39.3),
             122: (85.3, -22.9),
             123: (89.7, -6.4),
             124: (94.1, 10.0),
             125: (98.5, 26.4),
             126: (102.9, 42.8),
             127: (107.3, 59.2),
             128: (111.7, 75.7),
             129: (116.1, 92.1),
             130: (140.2, 68.0),
             131: (135.8, 51.6),
             132: (131.4, 35.2),
             133: (127.0, 18.8),
             134: (122.6, 2.4),
             135: (118.2, -14.1),
             136: (113.8, -30.5),
             137: (109.4, -46.9),
             138: (105.0, -63.3),
             139: (100.6, -79.7),
             140: (96.2, -96.2),
             141: (91.8, -112.6),
             142: (133.4, -70.9),
             143: (137.8, -54.5),
             144: (142.2, -38.1),
             145: (-40.5, 19.6),
             146: (-44.9, 3.2),
             147: (-7.6, 28.4),
             148: (-12.0, 12.0),
             149: (-16.4, -4.4),
             150: (-20.8, -20.8),
             151: (25.2, 37.2),
             152: (20.8, 20.8),
             153: (16.4, 4.4),
             154: (12.0, -12.0),
             155: (7.6, -28.4),
             156: (3.2, -44.9),
             157: (44.9, -3.2),
             158: (49.3, 13.2),
             159: (146.6, -21.7)}
""":obj:`dict` of :obj:`int`, (:obj:`float`, :obj:`float`) mappings: Home 
positions of the Taipan fibres in millimetres relative to plate centre

Note
----

Deprecated. This was the original definition of :any:`BUGPOS_MM`, 
before the ability to switch to 300-fibre configuration was 
introduced.
"""

ARCSEC_PER_MM = 67.2
""":obj:`float`: Conversion factor from millimetres to arcseconds.

Correct usage is: arcseconds = :any:`ARCSEC_PER_MM` * millimetres."""
BUGPOS_ARCSEC = {key: (value[0]*ARCSEC_PER_MM, value[1]*ARCSEC_PER_MM)
                 for key, value in BUGPOS_MM.items()}
""":obj:`dict` of :obj:`int`, (:obj:`float`, :obj:`float`) mappings: Home 
positions of the Taipan fibres in arcseconds relative to plate centre

Values are computed at module load using :any:`BUGPOS_MM` and
:any:`ARCSEC_PER_MM`.
"""
# Convert these lateral shifts into a distance & PA from the
# tile centre
BUGPOS_OFFSET = {key : ( math.sqrt(value[0]**2 + value[1]**2), 
                         math.degrees(math.atan2(value[0], value[1])) 
                            % 360., )
                 for key, value in BUGPOS_ARCSEC.items()}
""":obj:`dict` of :obj:`int`, (:obj:`float`, :obj:`float`) mappings: Home 
positions of the Taipan fibres in arcsecond distance and position angle from
the plate centre

Values are computed at module load from :any:`BUGPOS_ARCSEC`. Position angle
values are stored in degrees.
"""

# Define which fibres are the guide bundles
FIBRES_GUIDE = [
    46, 51, 56, 149, 155, 161, 254, 259, 264,
]
""":obj:`list` of `int`: List of guide fibre IDs.

Raises
------
Exception
    Raised if :any:`FIBRES_NORMAL` + 
    :any:`FIBRES_GUIDE` does not equal :any:`INSTALLED_FIBRES`. Note this check
    is only performed at module load.
"""
FIBRES_GUIDE.sort()
# Uncomment the following code to force GUIDES_PER_TILE 
# to be as high as possible
# if len(GUIDE_FIBRES) != GUIDES_PER_TILE:
#   raise Exception('Length of GUIDE_FIBRES array does not match'
#       'GUIDES_PER_TILE. Check the constant in taipan.core')
FIBRES_NORMAL = [f for f in
                 [3,
                  4,
                  5,
                  6,
                  17,
                  18,
                  19,
                  20,
                  21,
                  22,
                  23,
                  24,
                  25,
                  26,
                  27,
                  28,
                  44,
                  45,
                  46,
                  47,
                  48,
                  49,
                  50,
                  51,
                  52,
                  53,
                  54,
                  55,
                  56,
                  57,
                  75,
                  76,
                  77,
                  78,
                  79,
                  80,
                  81,
                  82,
                  83,
                  84,
                  85,
                  86,
                  87,
                  88,
                  89,
                  90,
                  100,
                  101,
                  110,
                  111,
                  112,
                  113,
                  114,
                  115,
                  116,
                  117,
                  118,
                  119,
                  120,
                  121,
                  122,
                  123,
                  124,
                  125,
                  126,
                  135,
                  136,
                  137,
                  138,
                  147,
                  148,
                  149,
                  150,
                  151,
                  152,
                  153,
                  154,
                  155,
                  156,
                  157,
                  158,
                  159,
                  160,
                  161,
                  162,
                  163,
                  164,
                  171,
                  172,
                  173,
                  174,
                  175,
                  176,
                  183,
                  184,
                  185,
                  186,
                  187,
                  188,
                  189,
                  190,
                  191,
                  192,
                  193,
                  194,
                  195,
                  196,
                  197,
                  198,
                  199,
                  200,
                  208,
                  209,
                  219,
                  220,
                  221,
                  222,
                  223,
                  224,
                  225,
                  226,
                  227,
                  228,
                  229,
                  230,
                  231,
                  232,
                  233,
                  234,
                  252,
                  253,
                  254,
                  255,
                  256,
                  257,
                  258,
                  259,
                  260,
                  261,
                  262,
                  263,
                  264,
                  265,
                  281,
                  282,
                  283,
                  284,
                  285,
                  286,
                  287,
                  288,
                  289,
                  290,
                  291,
                  292,
                  305,
                  306,
                  307,
                  308]
                 if f not in FIBRES_GUIDE]
""":obj:`list` of :obj:`int`: List of normal (i.e. non-guide) fibres installed.

The module will always load with the 150-fibre configuration setup. The correct
way to switch to the 300-fibre configuration is to use :any:`_alter_fibres`.

Raises
------
Exception
    Raised if :any:`FIBRES_NORMAL` + 
    :any:`FIBRES_GUIDE` does not equal :any:`INSTALLED_FIBRES`. Note this check
    is only performed at module load.
"""
# FIBRES_NORMAL = [f for f in BUGPOS_OFFSET if f not in FIBRES_GUIDE]
FIBRES_NORMAL.sort()
FIBRES_NORMAL_150 = FIBRES_NORMAL[:]
""":obj:`list` of :obj:`int`: List of normal (i.e. non-guide) fibres installed
in the 150-fibre configuration.

This is copied from :any:`FIBRES_NORMAL` at module load."""

FIBRES_ACTIVE = FIBRES_NORMAL + FIBRES_GUIDE
FIBRES_ACTIVE.sort()

if len(FIBRES_NORMAL) + len(FIBRES_GUIDE) != INSTALLED_FIBRES:
    raise Exception('The number of fibre positions defined'
                    '(%d guides, %d normal) does'
                    ' not match the set value for INSTALLED_FIBRES (%d). '
                    'Please '
                    'check the fibre configuration variables in taipan.py.' %
                    (len(FIBRES_NORMAL), len(FIBRES_GUIDE), INSTALLED_FIBRES, ))

# FIBRE_EXCLUSION_DIAMETER = 10.0 * 60.0  # arcsec
# MCW 190814 - Updated value from Nu
MICRON_PER_MM = 1000
BUG_RADIUS = 4325  # microns
FIBRE_EXCLUSION_DIAMETER = (
                               (BUG_RADIUS + 1500) * 2.
                           ) / MICRON_PER_MM * ARCSEC_PER_MM
""":obj:`float`, arcseconds: The exclusion diameter of a fibre"""
FIBRE_EXCLUSION_RADIUS = FIBRE_EXCLUSION_DIAMETER / 2.0
""":obj:`float`, arcseconds: The exclusion radius of a fibre

Computed at module load as :any:`FIBRE_EXCLUSION_DIAMETER` :math:`/ 2.0`"""
# TILE_RADIUS = 3.0 * 60.0 * 60.0       # arcsec
# MCW 190814 - Updated value from Nu
TILE_RADIUS = 166500 / MICRON_PER_MM * ARCSEC_PER_MM  # arcsec
""":obj:`float`, arcseconds: Radius of a Taipan tile (i.e. the radius of the 
field plate

Note
----
The tiling code assumes that target centres can be placed on the field out to a 
distance of :any:`TILE_RADIUS` from the field plate centre. This value may need
further refinement.
"""
TILE_DIAMETER = 2.0 * TILE_RADIUS     # arcsec
""":obj:`float`, arcseconds: Diameter of a Taipan tile (i.e. the diameter of the 
field plate

Computed at module load from :any:`TILE_RADIUS`."""
# PATROL_RADIUS = 1.2 * 3600.           # arcsec
# MCW 190814 - New value from Nu
PATROL_RADIUS = 49500. / MICRON_PER_MM * ARCSEC_PER_MM  # arcsec
""":obj:`float`, arcseconds: Patrol radius of any given fibre

The patrol radius is the distance any fibre is permitted to move from its home
position as defined by, e.g., :any:`BUGPOS_ARCSEC` or :any:`BUGPOS_OFFSET`."""

# Quadrant definition for sky fibre allocation
QUAD_RADII = [TILE_RADIUS, 6830.52, 3415.26, 0.]
""":obj:`list` of :obj:`float`, arcseconds: Boundaries of the annuli used to
randomly assign sky fibres.

See :any:`TaipanTile.unpick_tile` for details of how this is used.

Raises
------
ValueError
    Raised if :any:`QUAD_RADII` does not have length one greater than
    :any:`QUAD_PER_RADII`.
"""
QUAD_PER_RADII = [12, 6, 2]
""":obj:`list` of :obj:`int`: Numbers of sky fibres to assign per annuli defined
by :any:`QUAD_RADII`.

See :any:`TaipanTile.unpick_tile` for details of how this is used.

Raises
------
ValueError
    Raised if :any:`QUAD_RADII` does not have length one greater than
    :any:`QUAD_PER_RADII`.
ValueError
    Raised if the sum of :any:`QUAD_PER_RADII` (that is, the total number of
    sky fibres to be assigned to a tile) does not match :any:`SKY_PER_TILE`.
"""
if len(QUAD_PER_RADII) != len(QUAD_RADII)-1:
    raise ValueError('QUAD_PER_RADII must have one less element than '
                     'QUAD_RADII')
if sum(QUAD_PER_RADII) != SKY_PER_TILE:
    raise UserWarning('The number of defined tile quandrants does not match '
                      'SKY_PER_TILE. These are meant to be the same!')

FIBRES_PER_QUAD = []
""":obj:`list` of :obj:`list` of :obj:`int`: Groups fibres by sky quadrant

Each sub-list of :any:`FIBRES_PER_QUAD` gives all the fibres IDs which reside 
in the same sky fibre quadrant, as computed at module load from 
:any:`QUAD_RADII` and :any:`QUAD_PER_RADII`."""
for i in range(len(QUAD_RADII))[:-1]:
    theta = 360. / QUAD_PER_RADII[i]
    for j in range(QUAD_PER_RADII[i]):
        FIBRES_PER_QUAD.append(
            [k for k in BUGPOS_OFFSET.keys() if
             k not in FIBRES_GUIDE and
             QUAD_RADII[i+1] <= BUGPOS_OFFSET[k][0] < QUAD_RADII[i] and
             j*theta <= BUGPOS_OFFSET[k][1] < (j+1)*theta]
        )


TARGET_PRIORITY_MIN = 0
""":obj:`int`: Minimum allowed priority value for a :any:`TaipanTarget`"""
TARGET_PRIORITY_MAX = 100
""":obj:`int`: Maximum allowed priority value for a :any:`TaipanTarget`"""
TARGET_PRIORITY_MS = 50
""":obj:`int`: Main-survey priority cutoff for :any:`TaipanTarget`

Targets with a priority less than :any:`TARGET_PRIORITY_MS` are considered 
to be main survey (i.e. not priority) targets, which affects the calculation of
things like target difficulty.
"""


# -----
# RECONFIGURATION FUNCTIONS
# These functions are 'hidden' to denote they should be used with
# *extreme* caution, not because they're actually not meant to be used
# -----

def _alter_fibres(no_fibres=150):
    """
    Alter module parameters so that we change the number of 'normal'
    (i.e. science-capable) fibres. The number (9) and position of guide
    fibres are not changed.

    .. warning: This function is designed to increase the number of fibres to
                300, or decrease it back down to 150. Sensible behaviour cannot
                be
                guaranteed if you raise the number of fibres above the default
                150,
                and then lower it to a number that isn't 150!

    Parameters
    ----------
    no_fibres : :obj:`int`, defaults to 150
        The number of science-capable fibres to place on the tile.
        The :any:`FIBRES_NORMAL`, :any:`INSTALLED_FIBRES`,
        :any:`TARGET_PER_TILE` and :any:`FIBRES_PER_TILE` values will be
        updated as needed.

    Returns
    -------
    Nil. Module parameters are updated.

    Raises
    ------
    :any:`ValueError`
        Raised if ``no_fibres`` isn't 150 or 300.
    """
    # Input checking
    no_fibres = int(no_fibres)
    if no_fibres not in [150, 300]:
        raise ValueError('Can only set the number of fibres '
                         'to be 150 or 300')

    global FIBRES_NORMAL
    global FIBRES_ACTIVE
    global BUGPOS_MM
    global BUGPOS_ARCSEC
    global BUGPOS_OFFSET
    global TARGET_PER_TILE
    global FIBRES_PER_TILE
    global INSTALLED_FIBRES

    if no_fibres == len(FIBRES_NORMAL):
        return  # No action required

    if no_fibres > len(FIBRES_NORMAL):
        FIBRES_NORMAL = [f for f in BUGPOS_OFFSET.keys() if
                         f not in FIBRES_GUIDE]
    else:
        FIBRES_NORMAL = FIBRES_NORMAL_150[:]
        
    FIBRES_ACTIVE = FIBRES_NORMAL + FIBRES_GUIDE
    FIBRES_ACTIVE.sort()

    # Re-compute relevant constants
    INSTALLED_FIBRES = len(FIBRES_NORMAL) + len(FIBRES_GUIDE)
    TARGET_PER_TILE = len(FIBRES_NORMAL) - STANDARDS_PER_TILE - SKY_PER_TILE
    FIBRES_PER_TILE = len(FIBRES_NORMAL) + len(FIBRES_GUIDE)


# ------
# GLOBAL UTILITY FUNCTIONS
# ------

def prod(iterable):
    """
    Compute product of all elements of an iterable.

    Parameters
    ----------
    iterable : any iterable object

    Returns
    -------
    product : type of the elements of iterable
        Product of all elements of the iterable
    """
    return reduce(operator.mul, iterable, 1)


def aitoff_plottable(radec, ra_offset=0.0):
    """
    Convert coordinates to those compatible with matplotlib
    aitoff projection 
    
    Parameters
    ----------
    radec : 2-tuple of float
        Right Ascension and Declination in degrees.
    ra_offset : float
        Number of degrees to offset the centre of the coordinate system.
        Defaults to 0.

    Returns
    -------
    ra, dec : 2-tuple of float, in radians
        (ra, dec) coordinates in radians
    """
    ra, dec = radec
    ra = (ra + ra_offset) % 360. - 180.
    return math.radians(ra), math.radians(dec)


def pa_points(ra, dec, ra1, dec1):
    """
    Compute the position angle of (ra1, dec1) from (ra, dec)

    Parameters
    ----------
    ra : :obj:`float`, degrees
        The RA of the initial position in degrees
    dec : :obj:`float`, degrees
        The dec of the initial position in degrees
    ra1 : :obj:`float`, degrees
        The RA of the second position in degrees
    dec1 : :obj:`float`, degrees
        The dec of the second position in degrees

    Returns
    -------
    :obj:`float`, degrees
        The position angle.
    """

    ra = np.radians(ra)
    dec = np.radians(dec)
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)

    radif = ra1 - ra

    pa = np.arctan2(np.sin(radif),
                    np.cos(dec)*np.tan(dec1) -
                    np.sin(dec)*np.cos(radif))

    pa = np.degrees(pa)
    if pa < 0.:
        pa += 360.

    return pa

    # rarad1 = np.radians(ra)
    # rarad2 = np.radians(ra1)
    # dcrad1 = np.radians(dec)
    # dcrad2 = np.radians(dec1)
    #
    # radif = rarad2 - rarad1
    #
    # angle = np.arctan2(np.sin(radif),
    #                    np.cos(dcrad1) * np.tan(dcrad2) -
    #                    np.sin(dcrad1) * np.cos(radif))
    #
    # result = np.degrees(angle)
    #
    # if True and (result < 0.0):
    #     result += 360.0

    return result


def dist_points(ra, dec, ra1, dec1):
    """
    Compute the distance between two points on the sky.

    Parameters
    ----------
    ra, dec : float, degrees
        First position
    ra1, dec1 : float, degrees
        Second position

    Returns
    -------
    dist: float, arcseconds
        The distance between the two input points in arcseconds

    """
    ra = math.radians(ra)
    dec = math.radians(dec)
    ra1 = math.radians(ra1)
    dec1 = math.radians(dec1)
    dist = 2*math.asin(math.sqrt((math.sin((dec1-dec)/2.))**2
        +math.cos(dec1)*math.cos(dec)*(math.sin((ra1-ra)/2.))**2))
    return math.degrees(dist) * 3600.


def dist_points_approx(ra, dec, ra1, dec1):
    """
    An approximate calculation for distance on the sky.

    Parameters
    ----------
    ra, dec : float, degrees
        First position
    ra1, dec1 : float, degrees
        Second position

    Returns
    -------
    dist: float, arcseconds
        The distance between the two input points in arcseconds

    """
    decfac = np.cos(dec * np.pi / 180.)
    dra = ra - ra1
    if np.abs(dra) > 180.:
        dra -= np.sign(dra) * 360.
    dist = np.sqrt((dra / decfac)**2 + (dec - dec1)**2)
    return dist * 3600.


def dist_points_mixed(ra, dec, ra1, dec1, dec_cut=30.0):
    """
    An approximate calculation for distance on the sky.

    Parameters
    ----------
    ra, dec : float, degrees
        First position
    ra1, dec1 : float, degrees
        Second position
    dec_cut: float, optional
        Declination value below which an approximate calculation can be used;
        otherwise, use the full calculation. Defaults to 30.0.

    Returns
    -------
    dist: float, arcseconds
        The distance between the two input points in arcseconds

    """
    dec_cut = abs(dec_cut)
    if abs(dec) <= abs(dec_cut) and abs(dec1) <= abs(dec_cut):
        dist = dist_points_approx(ra, dec, ra1, dec1)
    else:
        dist = dist_points(ra, dec, ra1, dec1)
    return dist


def dist_euclidean(dist_ang):
    """
    Compute a straight-line distance from an angular distance

    Parameters
    ----------
    dist_ang : float, degrees
        Angular distance to be considered

    Returns
    -------
    dist_straight: float
        Corresponding linear distance
    """
    return 2. * np.sin(np.radians(dist_ang/2.))

def dist_angular(dist_euclidean):
    """
    Compute an angular distance from an euclidean distance

    Parameters
    ----------
    dist_euclidean : float, degrees
        Euclidean distance to be considered

    Returns
    -------
    dist_ang: float
        Corresponding angular distance
    """
    return 2. * np.degrees(np.arcsin(dist_euclidean/2.))

def polar2cart(radec):
    """
    Convert RA, Dec coordinates to x, y, z

    Parameters
    ----------
    radec: 2-tuple of floats
        The RA and Dec of the required position in degrees

    Returns
    -------
    x, y, z: floats
        A representation of the passed position on the unit sphere
    """
    ra, dec = radec
    x = np.sin(np.radians(dec+90.)) * np.cos(np.radians(ra))
    y = np.sin(np.radians(dec+90.)) * np.sin(np.radians(ra))
    z = np.cos(np.radians(dec+90.))
    return x, y, z


def compute_offset_posn(ra, dec, dist, pa):
    """
    Compute a new position based on a given position, a distance
    from that position, and a position angle from that position.
    Based on http://williams.best.vwh.net/avform.htm.

    Parameters
    ----------
    ra, dec: float
        The initial position in decimal degrees.
    dist: float
        The distance to the new point in arcseconds.
    pa: float
        The position angle from the original position to the
        new position in decimal degrees.

    Returns
    -------
    ra_new, dec_new : float, degrees
        The new coordinates in decimal degrees.
    """
    # This calculation is done in radians, so need to convert
    # the necessary inputs to that
    ra = math.radians(ra)
    dec = math.radians(dec)
    dist = math.radians(dist / 3600.)
    pa = math.radians(pa % 360.)

    # Compute the new declination
    dec_new = math.asin((math.sin(dec) * math.cos(dist))
        + (math.cos(dec) * math.sin(dist) * math.cos(pa)))
    dlong = math.atan2(math.sin(pa) * math.sin(dist) * math.cos(dec),
        math.cos(dist) - (math.sin(dec) * math.sin(dec_new)))
    ra_new = ((ra + dlong + math.pi) % (2.*math.pi)) - math.pi

    ra_new = math.degrees(ra_new) % 360.
    dec_new = math.degrees(dec_new)
    return ra_new, dec_new


def generate_ranking_list(candidate_targets,
                          method='priority', combined_weight=1.0,
                          sequential_ordering=(2, 1)):
    """
    Generate a ranking list for target assignment.

    Parameters
    ----------
    candidate_targets : :class:`TaipanTarget` list
        The list of TaipanTargets to rank.
    method : string
        The ranking method. These methods are as for 
        :meth:`TaipanTile.assign_tile` and :meth:`TaipanTile.unpick_tile` --
        see the
        documentation for those functions for details. Defaults to 'priority'.
    combined_weight, sequential_ordering : float and string, optional
        Extra parameters for the
        'combined_weighted' and 'sequential' ranking options. See docs for
        assign_tile/unpick_tile for details. Defaults to 1.0 and (1,2)
        respectively.

    Returns
    -------
    ranking_list: list of int or float
        A list of ints/floats describing the ranking of the
        candidate_targets. The largest values corresponds to the
        most highly-ranked target. The positions in the list correspond to the
        positions of targets in candidate_targets.
    """

    # Compute the ranking list for the selection procedure
    # If the ranking method is sequential, compute the equivalent
    # combined_weight and change the method to combined_weighted
    if len(candidate_targets) == 0:
        ranking_list = []
    elif method == 'sequential':
        difficulty_list = [t.difficulty for t in candidate_targets]
        priority_list = [t.priority for t in candidate_targets]
        lists = [None, difficulty_list, priority_list]
        maxes = [None, max(difficulty_list), max(priority_list)]
        ranking_list = [maxes[sequential_ordering[1]] * lists[
            sequential_ordering[0]][i] + lists[
            sequential_ordering[1]][i] 
            for i in range(len(difficulty_list))]
    elif method == 'most_difficult':
        ranking_list = [t.difficulty for t in candidate_targets]
    elif method == 'priority':
        ranking_list = [t.priority for t in candidate_targets]
    elif method == 'combined_weighted':
        max_excluded_tgts = float(max(
            [t.difficulty for t 
                in candidate_targets])) / float(TARGET_PRIORITY_MAX)
        ranking_list = [combined_weight*t.priority 
            + float(t.difficulty)/max_excluded_tgts 
            for t in candidate_targets]

    return ranking_list


def grab_target_difficulty(target, target_list):
    """
    External means of computing the difficulty of a TaipanTarget.

    Parameters
    ----------
    target : :class:`TaipanTarget`
        The target of interest.
    target_list : :class:`TaipanTarget` list
        The list of targets with which to compare. Although not
        enforced, it should contain target.

    Returns
    -------
    target : :class:`TaipanTarget`
        The original target with an updated difficulty.
    """
    target.compute_difficulty(target_list)
    return target


def compute_target_difficulties(target_list, full_target_list=None,
                                verbose=False, leafsize=BREAKEVEN_KDTREE):
    """
    Compute the target difficulties for a list of targets.

    Compute the target difficulties for a given list of :class:`TaipanTarget`s.
    The 'difficulty' is simply the number of targets within
    :any:`FIBRE_EXCLUSION_RADIUS` of the target of interest. Note that the
    target itself will be included in this number; therefore,
    the minimum value of difficulty will be 1.

    Parameters
    ----------
    target_list : list of :class:`TaipanTarget`
        The list of TaipanTargets to compute the difficulty for.
    full_target_list : list of :class:`TaipanTarget`
        The full list of targets to use in the difficulty computation.

        This is useful for situations where targets need to be considered in a
        re-computation of difficulty, but the difficulty of those targets need
        not be updated. This occurs, e.g. when target difficulties must be
        updated after some targets have been assigned to a tile. Only targets
        within :any:`TILE_RADIUS` + :any:`FIBRE_EXCLUSION_RADIUS` of the tile
        centre need updating; however, targets within
        :any:`TILE_RADIUS` + 2 * :any:`FIBRE_EXCLUSION_RADIUS` need to be
        considered in the calculation.
        If given, ``target_list`` MUST be a sublist of ``full_target_list``.
        If not, a ValueError will be raised.
        Defaults to :any:`None`, in which case, target difficulties are computed
        for all targets in ``target_list`` against ``target_list`` itself.
    verbose : bool, optional
        Whether to print detailed debug information to the root logger. Defaults
        to False.
    leafsize : int, optional
        The leafsize (i.e. number of targets) where it becomes more efficient
        to construct a KDTree rather than brute-force the distances between
        targets. Defaults to the module default (i.e. :any:`BREAKEVEN_KDTREE`).

    Returns
    ------- 
    Nil. :class:`TaipanTarget` within ``target_list`` updated in-place.

    Raises
    ------
    ValueError
        Raised if ``target_list`` is not a sublist of ``full_target_list``.
    UserWarning
        Raised if any target in ``target_list`` ends up with a difficulty
        less than 1.
    """

    tree_function = cKDTree

    if len(target_list) == 0:
        return

    if full_target_list:
        if verbose:
            'Checking target_list against full_target_list...'
        # Notes about np.in1d - fails for object arrays that are unsortable. 
        # This is not a problem for the Taipan code, but for FunnelWeb, which 
        # implements object level equivalence (i.e. __eq__, __ne__, __cmp__), 
        # this check will always raise an exception, even when the arrays are 
        # subsets. Future versions of numpy will fix this, but for now a 
        # workaround is needed. See issue links for more details:
        # 1) https://github.com/numpy/numpy/issues/9874
        # 2) https://github.com/numpy/numpy/issues/9914
        if not isinstance(target_list[0], FWTarget) \
            and not np.all(np.in1d(target_list, full_target_list)):
            raise ValueError('target_list must be a sublist'
                             ' of full_target_list')
            
    if verbose:
        logging.debug('Forming Cartesian positions...')
    # Calculate UC positions if they haven't been done already
    burn = [t.compute_usposn() for t in target_list if t.usposn is None]
    cart_targets = np.asarray([t.usposn for t in target_list])
    if full_target_list:
        burn = [t.compute_usposn() for t in full_target_list 
                if t.usposn is None]
        full_cart_targets = np.asarray([t.usposn for t in full_target_list])
    else:
        full_cart_targets = np.copy(cart_targets)
    
    if verbose:
        logging.debug('Generating KDTree with leafsize %d' % leafsize)
    if tree_function == skKDTree:
        tree = tree_function(full_cart_targets, leaf_size=leafsize)
    else:
        tree = tree_function(full_cart_targets, leafsize=leafsize)

    dist_check = dist_euclidean(FIBRE_EXCLUSION_DIAMETER/3600.)

    if tree_function == skKDTree:
        difficulties = tree.query_radius(cart_targets,
                                         dist_euclidean(
                                             FIBRE_EXCLUSION_DIAMETER/3600.))
    else:
        if len(target_list) < (100*leafsize):
            if verbose:
                logging.debug('Computing difficulties...')
            difficulties = tree.query_ball_point(cart_targets,
                dist_euclidean(FIBRE_EXCLUSION_DIAMETER/3600.))
        else:
            if verbose:
                logging.debug('Generating subtree for difficulties...')
            subtree = tree_function(cart_targets, leafsize=leafsize)
            difficulties = subtree.query_ball_tree(
                tree,
                dist_euclidean(FIBRE_EXCLUSION_DIAMETER/3600.))
    difficulties = [len(d) for d in difficulties]

    if verbose:
        logging.debug('Assigning difficulties...')
    for i in range(len(difficulties)):
        target_list[i].difficulty = difficulties[i]
    if verbose:
        logging.debug('Difficulties done!')
        
    difficulties = [1]
    if min(difficulties) == 0:
        raise UserWarning

    return


def targets_in_range(ra, dec, target_list, dist, leafsize=BREAKEVEN_KDTREE, 
                     tree=None):
    """
    Return the subset of ``target_list`` within ``dist`` of (``ra``, ``dec``).

    Computes the subset of targets/tiles within range of the given (ra, dec)
    coordinates.

    Parameters
    ----------
    ra, dec : float
        The RA and Dec of the position to be investigated, in decimal
        degrees.
    target_list : list of :class:`TaipanPoint` (or children of)
        The list of TaipanPoint child objects to consider.
    dist : float
        The distance to test against, in *arcseconds*.
    leafsize : int, optional
        The size of the leaves in the KDTree structure. Defaults to
        :any:`BREAKEVEN_KDTREE`.
    tree : cKDTree
        Pre-computed tree to use to save on processing. Defaults to 
        :obj:`None` (at which point a new tree will be computed).
      
    Returns
    -------
    targets_in_range : list of :class:`TaipanPoint` (or children of)
        The list of input targets/tiles which are within dist of
        (ra, dec).
    """

    if len(target_list) == 0:
        return []

    # Decide whether to brute-force or construct a KDTree
    if len(target_list) <= BREAKEVEN_KDTREE:
        targets_in_range = [t for t in target_list
            if t.dist_point((ra, dec)) < dist]
    else:
        if not tree:
            # Do KDTree computation
            logging.debug('Generating KDTree with leafsize %d' % leafsize)
            cart_targets = np.asarray([t.usposn for t in target_list])
            # logging.debug(cart_targets)
            tree = cKDTree(cart_targets, leafsize=leafsize)
            logging.debug('Querying tree')
            inds = tree.query_ball_point(polar2cart((ra, dec)),
                                         dist_euclidean(dist / 3600.))
            targets_in_range = [target_list[i] for i in inds]
        else:
            # Tree has been supplied, save on computation time by using it
            inds = tree.query_ball_point(polar2cart((ra, dec)),
                                         dist_euclidean(dist / 3600.))
            targets_in_range = [target_list[i] for i in inds]
            
    return targets_in_range


def targets_in_range_multi(ra_dec_list, target_list, dist,
                           leafsize=BREAKEVEN_KDTREE):
    """
    Return the number of targets/tiles in target_list
    within each position specified in ra_dec_list.

    Computes the subset of targets within range of the given (ra, dec)
    coordinates.

    Parameters
    ----------
    ra_dec_list : iterable of 2-tuples of float
        An iterable of (ra, dec) tuples to compute the targets in range of.
    target_list : list of :class:`TaipanTarget`
        The list of TaipanTarget objects to consider.
    dist : float
        The distance to test against, in *arcseconds*.

    Returns
    -------
    targets_in_range : list of :class:`TaipanPoint`
        A list of lists of TaipanPoint child objects. Each sublist contains the 
        targets/tiles within dist of the corresponding (ra, dec) in ra_dec_list.
    """

    # Make sure ra_dec_list is an iterable
    try:
        _ = (e for e in ra_dec_list)
    except TypeError:
        ra_dec_list = [ra_dec_list]

    if len(target_list) == 0:
        return [[]] * len(ra_dec_list)

    cart_targets = np.asarray([t.usposn for t in target_list])
    tree = cKDTree(cart_targets, leafsize=leafsize)
    inds = [tree.query_ball_point(polar2cart(radec),
                                  dist_euclidean(dist / 3600.))
            for radec in ra_dec_list]
    targets = [[target_list[i] for i in ind] for ind in inds]

    return targets


def targets_in_range_tiles(tile_list, target_list, dist=TILE_RADIUS,
                           leafsize=BREAKEVEN_KDTREE):
    """
    Alias to :any:`targets_in_range_multi` for use when passing a list of
    TaipanPoint child objects (typically tiles).

    Parameters
    ----------
    tile_list : list of :class:`TaipanPoint` (or children of)
        List of TaipanPoint objects (typically TaipanTiles).
    target_list : list of :class:`TaipanPoint` (or children of)
        List of TaipanPoint objects (typically TaipanTargets) to consider.
    leafsize : int, optional
        Optional. Leafsize of the constructed KDTree. Defaults to
        :any:`BREAKEVEN_KDTREE`.

    Returns
    -------
    As for :any:`targets_in_range_multi`.
    """

    return targets_in_range_multi(
        [(t.ra, t.dec) for t in tile_list],
        target_list,
        dist,
        leafsize=leafsize
    )

# ------
# TILING OBJECTS
# ------


class TaipanPoint(object):
    """
    A root class for :class:`TaipanTarget` and :class:`TaipanTile`,
    including RA, Dec and associated convenience functions.
    """

    def __init__(self, ra, dec, usposn=None):
        """
        Parameters
        ----------
        ra, dec : float, degrees
            Target RA, Dec in degrees
        usposn : 3-tuple of float, optional
            Tile position in (x,y,z) on the unit sphere, defaults to None.
        """
        self._ra = None
        self._dec = None
        self._usposn = None
        
        # Insert the passed values
        # Doing it like this forces the setter functions to be
        # called, which provides error checking
        self.ra = ra
        self.dec = dec
        self.usposn = usposn
        
    @property
    def ra(self):
        """:obj:`float`: Right acsension (RA) in decimal degrees

        Raises
        ------
        Exception:
            If ``ra`` is set to :any:`None`; if ``ra`` is outside the range
            :math:`[0,360)`.
        """
        return self._ra

    @ra.setter
    def ra(self, r):
        if r is None: raise Exception('RA may not be blank')
        if r < 0.0 or r >= 360.0: 
            raise Exception('RA {} outside valid range'.format(r))
        self._ra = r

    @property
    def dec(self):
        """:obj:`float`: Declination (Dec) in decimal degrees

        Raises
        ------
        Exception:
            If ``dec`` is set to :any:`None`; if ``dec`` is outside the range
            :math:`[-90,90]`.
        """
        return self._dec

    @dec.setter
    def dec(self, d):
        if d is None: raise Exception('Dec may not be blank')
        if d < -90.0 or d > 90.0:
            raise Exception('Dec outside valid range')
        self._dec = d

    @property
    def usposn(self):
        """3-tuple of :obj:`float`: (x,y,z) position on the unit sphere

        This Cartesian representation of the point position is required
        to utilize the :any:`cKDTree` distance calculation method.

        Raises
        ------
        Exception
            Raised if ``usposn`` is not a 3-tuple (or list), if any of the three
            values are outside the valid range :math:`[-1,1]`, or if the
            three values do no satisfy :math:`x^2 + y^2 + z^2 = 1.0`.
        """
        return self._usposn

    @usposn.setter
    def usposn(self, value):
        if value is None:
            self._usposn = None
            return
        if len(value) != 3:
            raise Exception('usposn must be a 3-list or 3-tuple')
        if value[0] < -1. or value[0] > 1.:
            raise Exception('x value %f outside allowed bounds (-1 <= x <= 1)'
                % value[0])
        if value[1] < -1. or value[1] > 1.:
            raise Exception('y value %f outside allowed bounds (-1 <= x <= 1)'
                % value[1])
        if value[2] < -1. or value[2] > 1.:
            raise Exception('z value %f outside allowed bounds (-1 <= x <= 1)'
                % value[2])
        if abs(value[0]**2 + value[1]**2 + value[2]**2 - 1.0) > 0.001:
            raise Exception('usposn must lie on unit sphere '
                            '(x^2 + y^2 + z^2 = 1.0 - error of %f '
                            '(%f, %f, %f) )'
                            % (value[0]**2 + value[1]**2 + value[2]**2,
                               value[0], value[1], value[2]))
        self._usposn = list(value)

    def compute_usposn(self):
        """
        Compute the position of this target on the unit sphere from its
        RA and Dec values.

        No input parameters are required for this function - the internal values
        of ``ra`` and ``dec`` are accessed instead.

        Raises
        ------
        Exception:
            Raised if ``ra`` or ``dec`` is :any:`None`. Note that the input
            checking on these values should protect against this occurring.
        """
        if self.ra is None or self.dec is None:
            raise Exception('Cannot compute usposn because'
                            ' RA and/or Dec is None!')
        self.usposn = polar2cart((self.ra, self.dec))
        return

    def ranked_index(self, usposns, maxdist):
        """Given a list of positions on the unit sphere, find a ranked list of
        indices to these points within a maximum distance.

        .. note: This can be sped up even more in an obvious way!
        
        Parameters
        ----------
        usposns: numpy (npoints, 3) array
            The unit sphere positions we're to compute distances to
        
        maxdist: float
            A maximum distance for which we care about the distance.
        
        Returns
        -------
        ix : numpy int array
        """
        # distance vectors. A numpy way for speed.
        fibre_dists = usposns - np.tile(self.usposn, len(usposns)).reshape(
            usposns.shape)
        # Convert to arcsec, for the magnitude of the vector
        fibre_dists = dist_angular(np.sqrt(np.sum(fibre_dists**2,1)))*3600.
        # Consider the closest points/fibers only.
        permitted_ix = np.where(fibre_dists < PATROL_RADIUS)[0]
        return permitted_ix[np.argsort(fibre_dists[permitted_ix])]

    def dist_usposn(self, usposn):
        """Given a unit sphere position, find the distance to this
        :class:`TaipanPoint`
        
        Parameters
        ----------
        usposn: numpy (3) float array or 3-tuple of :obj:`float`
            The unit sphere position we want to find the distance to.
        
        Returns
        -------
        dist: float, arcsecond
            Distance in arcsec
        """
        fibre_dist = dist_angular(np.sqrt(np.sum( (usposn - np.asarray(self.usposn))**2)))*3600.
        return fibre_dist

    def dist_point(self, radec):
        """
        Compute the distance between this target and a given position

        Parameters
        ----------
        radec : 2-tuple of float
            The sky position to test. Should be in decimal degrees.

        Returns
        -------
        dist : float, arcsecond
            The angular distance between the two points in arcsec.
        """

        # Arithmetic implementation - fast
        # Convert all to radians
        ra, dec = radec
        ra = math.radians(ra)
        dec = math.radians(dec)
        ra1 = math.radians(self.ra)
        dec1 = math.radians(self.dec)
        dist = 2*math.asin(math.sqrt((math.sin((dec1-dec)/2.))**2
            + math.cos(dec1)*math.cos(dec)*(math.sin((ra1-ra)/2.))**2))
        return math.degrees(dist) * 3600.

    def dist_point_approx(self, radec):
        """
        Compute the distance between this target and a given position using
        an approximate calculation.

        This calculation is faster than the full :any:`dist_point` calculation,
        but sacrifices accuracy the closer the point(s) are to the poles.

        Parameters
        ----------
        radec : 2-tuple of float
            The sky position to test. Should be in decimal degrees.

        Returns
        -------
        dist : float, arcsecond
            The angular distance between the two points in arcsec.
        """
        ra, dec = radec
        decfac = np.cos(dec * np.pi / 180.)
        dra = ra - self.ra
        if np.abs(dra) > 180.:
            dra = dra - np.sign(dra) * 360.
        dist = np.sqrt((dra / decfac)**2 + (dec - self.dec)**2)
        return dist * 3600.

    def dist_point_mixed(self, radec, dec_cut=30.):
        """
        Compute the distance between this target and a given position using
        a split implementation

        This function uses an approximate calculation near the equator,
        and an exact implementation near the poles.

        Parameters
        ----------
        radec : 2-tuple of float
            The sky position to test. Should be in decimal degrees.
        dec_cut: float, optional
            The (absolute) RA value above which the exact implementation
            should be used. Defaults to 30.0.

        Returns
        -------
        dist : float, arcsecond
            The angular distance between the two points in arcsec.
        """
        ra, dec = radec
        dec_cut = abs(dec_cut)
        if abs(dec) <= abs(dec_cut) and abs(self.dec) <= abs(dec_cut):
            dist = self.dist_point_approx((ra, dec))
        else:
            dist = self.dist_point((ra, dec))
        return dist

    def dist_target(self, tgt):
        """
        Compute the distance between this target and another target.

        This function is a convenience wrapper to :any:`dist_point`.

        Parameters
        ----------
        tgt : :class:`TaipanTarget`
            The target to check against

        Returns
        -------
        dist : float, arcseconds
            The angular distance between the two points in arcsec.
        """

        return self.dist_point((tgt.ra, tgt.dec))

    def dist_target_approx(self, tgt):
        """
        Compute the approximates distance between this target and another
        target.

        This function is a convenience wrapper to :any:`dist_point_approx`.

        Parameters
        ----------
        tgt : :class:`TaipanTarget`
            The target to check against

        Returns
        -------    
        dist : float, arcseconds
            The angular distance between the two points in arcsec.
        """

        return self.dist_point_approx((tgt.ra, tgt.dec))

    def dist_target_mixed(self, tgt, dec_cut=30.):
        """
        Compute the mixed distance between this target and another target.

        This function is a convenience wrapper to :any:`dist_target_mixed`.

        Parameters
        ----------    
        tgt : 
            The target to check against
        dec_cut: float, optional
            The (absolute) RA value above which the exact implementation
            should be used. Defaults to 30.0.

        Returns
        -------    
        dist : float, arcseconds
            The angular distance between the two points in arcsec.
        """

        return self.dist_point_mixed((tgt.ra, tgt.dec), dec_cut=dec_cut)

 
class TaipanTarget(TaipanPoint):
    """
    Holds information and convenience functions for a Taipan
    observing target.
    """

    # Initialisation & input-checking
    def __init__(self, idn, ra, dec, usposn=None, pm_ra=0., pm_dec=0.,
                 priority=1, standard=False,
                 guide=False, difficulty=0, mag=None,
                 h0=False, vpec=False, lowz=False, science=True,
                 assign_science=True, sky=False):
        """
        Parameters
        ----------
        idn : int
            Unique target ID
        ra, dec: float, degrees
            RA, Dec position of target
        ucposn : 3-tuple of floats , optional
            Target (x, y, z) position on the unit sphere. Defaults to None,
            at which point the position will be calculated internally and
            stored
        priority : int, optional
            Target priority value. Defaults to 1
        standard : Boolean, optional
            Denotes this target as a standard. Defaults to False.
        guide : Boolean, optional
            Denotes this target as a guide. Defaults to False.
        difficulty : int, optional
            Number of targets that this target would exclude. Defaults to 0.
        mag : float, optional
            Target magnitude. Defaults to None.
        h0, vpec, lowz : Boolean, optional
            Booleans denoting if a target is an H0 target, a low-redshift (lowz)
            target, or a peculiar velocity (vpec) target. All default to False.
        science : Boolean, optional
            Denotes this target as a science target. defaults to True
        assign_science : Boolean, optional
            Do we automatically assign the science flag based on standard and
            guide flags? Defaults to True
        sky: boolean
            Denotes this target as a science target. Defaults to False.
        """
        # Initialise the base class
        TaipanPoint.__init__(self, ra, dec, usposn)

        self._pm_ra = None
        self._pm_dec = None
        
        self._idn = None
        self._priority = None
        self._priority_original = None
        self._standard = None
        self._guide = None
        self._science = None
        self._difficulty = None
        self._mag = None
        self._sky = None

        # Taipan-specific fields
        self._h0 = None
        self._vpec = None
        self._lowz = None

        # Insert given values
        # This causes the setter functions to be called, which does
        # error checking
        self.pm_ra = pm_ra
        self.pm_dec = pm_dec
        self.idn = idn
        self.priority = priority
        self.priority_original = priority
        self.standard = standard
        self.science = science
        self.guide = guide
        self.difficulty = difficulty
        self.mag = mag
        self.h0 = h0
        self.vpec = vpec
        self.lowz = lowz
        self.sky = sky
        
        # A default useful for Taipan (FunnelWeb will override,
        # as it takes its standards
        # largely from the science targets, and these are not mutually
        # exclusive)
        if assign_science:
            if self.standard or self.guide:
                self.science = False

    def __repr__(self):
        return 'TP TGT %s' % str(self._idn)

    def __str__(self):
        return 'TP TGT %s' % str(self._idn)

    # Uncomment to have target equality decided on ID
    # WARNING - make sure your IDs are unique!
    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return (self.idn == other.idn) and (self.standard
    #                                             == other.standard) and (
    #             self.guide == other.guide)
    #     return False
    # 
    # def __ne__(self, other):
    #     if isinstance(other, self.__class__):
    #         return not((self.idn == other.idn) and (self.standard
    #                                                 == other.standard) and (
    #             self.guide == other.guide))
    #     return True
    # 
    # def __cmp__(self, other):
    #     if isinstance(other, self.__class__):
    #         if (self.idn == other.idn) and (self.standard
    #                                         == other.standard) and (
    #                     self.guide == other.guide):
    #             return 0
    #     return 1


    @property
    def pm_ra(self):
        return self._pm_ra

    @pm_ra.setter
    def pm_ra(self, r):
        if r is None: raise ValueError('pm_ra may not be None')
        self._pm_ra = float(r)

    @property
    def pm_dec(self):
        return self._pm_dec

    @pm_dec.setter
    def pm_dec(self, r):
        if r is None: raise ValueError('pm_dec may not be None')
        self._pm_dec = float(r)

    @property
    def idn(self):
        """:obj:`int`: Target ID

        Raises
        ------
        Exception
            Raised if ``idn`` set to :any:`None`, or to anything other than
            :obj:`int`.
        """
        return self._idn

    @idn.setter
    def idn(self, d):
        if not d: raise Exception('ID may not be empty')
        if not isinstance(d, int): raise Exception('ID must be an int')
        self._idn = d

    @property
    def priority(self):
        """:obj:`int`: Target priority

        Raises
        ------
        ValueError
            Raised if priority is outside the range
            [:any:`TARGET_PRIORITY_MIN`, :any:`TARGET_PRIORITY_MAX`].
        """
        return self._priority

    @priority.setter
    def priority(self, p):
        # Make sure priority is an int
        p = int(p)
        if p < TARGET_PRIORITY_MIN or p > TARGET_PRIORITY_MAX:
            raise ValueError('Target priority must be %d <= p <= %d' 
                % (TARGET_PRIORITY_MIN, TARGET_PRIORITY_MAX, ))
        self._priority = p

    @property
    def priority_original(self):
        """:obj:`int`: Original :any:`priority` of target

        This is used to keep track of the previous `priority` of the
        :any:`TaipanTarget` if changed
        """
        return self._priority_original

    @priority_original.setter
    def priority_original(self, p):
        # Make sure priority is an int
        p = int(p)
        if p < TARGET_PRIORITY_MIN or p > TARGET_PRIORITY_MAX:
            raise ValueError('Target priority must be %d <= p <= %d' 
                % (TARGET_PRIORITY_MIN, TARGET_PRIORITY_MAX, ))
        self._priority_original = p

    @property
    def standard(self):
        """:obj:`bool`: Denotes if this target is a standard"""
        return self._standard

    @standard.setter
    def standard(self, b):
        b = bool(b)
        self._standard = b

    @property
    def science(self):
        """:obj:`bool`: Denotes if this target is a science target"""
        return self._science

    @science.setter
    def science(self, b):
        b = bool(b)
        self._science = b

    @property
    def guide(self):
        """:obj:`bool`: Denotes if this target is a guide"""
        return self._guide

    @guide.setter
    def guide(self, b):
        b = bool(b)
        self._guide = b

    @property
    def difficulty(self):
        """Difficulty, i.e. number of targets within FIBRE_EXCLUSION_DIAMETER"""
        return self._difficulty

    @difficulty.setter
    def difficulty(self, d):
        d = int(d)
        if d < 0:
            raise ValueError('Difficulty must be >= 0')
        self._difficulty = d

    @property
    def mag(self):
        """:obj:`float`: Target magnitude

        Raises
        ------
        AssertionError
            Raised if ``mag`` outside the range :math:`[-10,30]`.
        """
        return self._mag

    @mag.setter
    def mag(self, m):
        if m and not np.isnan(m):
            assert (m > -10 and m < 30), "mag {} outside valid range".format(m)
        self._mag = m

    @property
    def h0(self):
        """:obj:`bool`: Denotes if this target is an H0-survey target"""
        return self._h0

    @h0.setter
    def h0(self, b):
        b = bool(b)
        self._h0 = b

    @property
    def lowz(self):
        """:obj:`bool`: Denotes if this target is an low redshift-survey
        target"""
        return self._lowz

    @lowz.setter
    def lowz(self, b):
        b = bool(b)
        self._lowz = b

    @property
    def vpec(self):
        """:obj:`bool`: Denotes if this target is an peculiar velocity
        survey target"""
        return self._vpec

    @vpec.setter
    def vpec(self, b):
        b = bool(b)
        self._vpec = b
        
    @property
    def sky(self):
        """Is this target a sky fibre"""
        return self._sky

    @sky.setter
    def sky(self, b):
        b = bool(b)
        self._sky = b

    def return_target_code(self):
        """
        Return a single-character string based on the type of TaipanTarget
        passed as a function argument.

        Returns
        -------
        code : str
            A single-character string, denoting the type of TaipanTarget passed in.
            Codes are currently:
            X - science
            G - guide
            S - standard
            T - standard and science
            Z - an inconsistent target type!
        """

        if self.standard and self.science:
            code = 'T'
        elif self.standard:
            code = 'S'
        elif self.guide:
            code = 'G'
        elif self.science:
            code = 'X'
        else:
            code = 'Z'

        return code

    def excluded_targets(self, tgts):
        """
        Given a list of other TaipanTargets, return a list of those 
        targets that are too close to this target to be on the same
        tile. Note that, if the calling target is in the target list,
        it will appear in the returned list of forbidden targets.

        This is a convenient wrapper for :any:`targets_in_range`.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            The list of TaipanTargets to test against

        Returns
        -------    
        excluded_tgts : list of :class:`TaipanTarget`
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = targets_in_range(self.ra, self.dec, tgts,
                                         FIBRE_EXCLUSION_DIAMETER)
        return excluded_tgts

    def excluded_targets_approx(self, tgts):
        """
        As for excluded_targets, but using the approximate distance calculation.
        This will *under*-estimate difficulty (by overestimating distance),
        especially near the poles.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            The list of TaipanTargets to test against

        Returns
        -------    
        excluded_tgts : list of :class:`TaipanTarget`
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = [t for t in tgts 
            if self.dist_target_approx(t) < FIBRE_EXCLUSION_DIAMETER]

        return excluded_tgts

    def excluded_targets_mixed(self, tgts, dec_cut=30.):
        """
        As for excluded_targets, but using the mixed distance calculation.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            The list of TaipanTargets to test against
        dec_cut : float, optional
            (Absolute) declination value to use for mixed method. Defaults to
            30.0.

        Returns
        -------    
        excluded_tgts : list of :class:`TaipanTarget`
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = [t for t in tgts 
            if self.dist_target_mixed(t, dec_cut=dec_cut) 
            < FIBRE_EXCLUSION_DIAMETER]

        return excluded_tgts

    def compute_difficulty(self, tgts):
        """
        Calculate & set the difficulty of this target.

        Difficulty is defined as the number of targets within a 
        :any:`FIBRE_EXCLUSION_DIAMETER` of the calling target. This means that
        this function will need to be invoked every time the comparison list
        changes.

        This function simply measures the length of the return from
        :any:`excluded_targets`.

        Note that if the calling target is also within tgts, it will count
        towards the computed difficulty.

        Parameters
        ----------     
        tgts : list of :class:`TaipanTarget`
            List of targets to compare to.
        approx : bool, optional
            Boolean value, denoting whether to calculate distances using
            the approximate method. Defaults to False.
        mixed : bool, optional
            Boolean value, denoting whether to calculate distances using
            the mixed method (approx if dec < dec_cut, full otherwise). Defaults
            to False.

        Returns
        -------     
        Nil. Target's difficulty parameter is updated in place.
        """
        self.difficulty = len(self.excluded_targets(tgts))
        return

    def compute_difficulty_approx(self, tgts):
        """
        As for :any:`compute_difficulty`, but use the approximate distance
        calculation.

        Parameters
        ----------
        tgts : list of :class:`TaipanTarget`
            List of targets to compute the difficulty of this target against
        """
        self.difficulty = len(self.excluded_targets_approx(tgts))
        return

    def compute_difficulty_mixed(self, tgts, dec_cut=30.):
        """
        As for :any:`compute_difficulty`, but use the mixed distance
        calculation.

        Parameters
        ----------
        tgts : list of :class:`TaipanTarget`
            List of targets to compute the difficulty of this target against
        dec_cut : float, optional
            Break point for using an exact or approximate distance calculation.
            Defaults to 30.0
        """
        self.difficulty = len(self.excluded_targets_mixed(tgts,
                                                          dec_cut=30.))
        return

    def is_target_forbidden(self, tgts):
        """
        Test against a list of other targets to see if this target is forbidden.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            The list of targets to test against.

        Returns
        -------    
        forbidden : bool
            If this target is forbidden or not, based on the input target list.
        """
        if len(tgts) == 0:
            return False

        if len(targets_in_range(self.ra, self.dec, tgts, 
                                FIBRE_EXCLUSION_DIAMETER)) > 0:
            return True
        return False


class FWTarget(TaipanTarget):
    """Derived class for the FunnelWeb survey, primarily to allow object 
    equivalence.
    """
    def __init__(self, idn, ra, dec, usposn=None, priority=1, standard=False,
                 guide=False, difficulty=0, mag=None,
                 h0=False, vpec=False, lowz=False, science=True,
                 assign_science=True, sky=False):
        """
        Parameters
        ----------
        idn : int
            Unique target ID
        ra, dec: float, degrees
            RA, Dec position of target
        ucposn : 3-tuple of floats , optional
            Target (x, y, z) position on the unit sphere. Defaults to None,
            at which point the position will be calculated internally and
            stored
        priority : int, optional
            Target priority value. Defaults to 1
        standard : Boolean, optional
            Denotes this target as a standard. Defaults to False.
        guide : Boolean, optional
            Denotes this target as a guide. Defaults to False.
        difficulty : int, optional
            Number of targets that this target would exclude. Defaults to 0.
        mag : float, optional
            Target magnitude. Defaults to None.
        h0, vpec, lowz : Boolean, optional
            Booleans denoting if a target is an H0 target, a low-redshift 
            (lowz) target, or a peculiar velocity (vpec) target. All default 
            to False.
        science : Boolean, optional
            Denotes this target as a science target. Defaults to True
        assign_science : Boolean, optional
            Do we automatically assign the science flag based on standard/guide 
            flags? Defaults to True
        sky: boolean
            Denotes this target as a science target. Defaults to False.
        """
        # Initialise the base class
        TaipanTarget.__init__(self, idn, ra, dec, usposn, priority, standard,
                              guide, difficulty, mag, h0, vpec, lowz, science,
                              assign_science, sky)
        
    def __repr__(self):
        return 'FW TGT %s' % str(self._idn)

    def __str__(self):
        return 'FW TGT %s' % str(self._idn)

    # Two FWTargets should be considered equal if they have the same ID, and 
    # status as science/standard/guide/sky targets.
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return ((self.idn == other.idn) 
                    and (self.standard == other.standard)
                    and (self.guide == other.guide) 
                    and (self.sky == other.sky))
        return False
    
    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not((self.idn == other.idn) 
                       and (self.standard == other.standard) 
                       and (self.guide == other.guide)
                       and (self.sky == other.sky))
        return True
    
    def __cmp__(self, other):
        if isinstance(other, self.__class__):
            if ((self.idn == other.idn) and (self.standard == other.standard) 
                and (self.guide == other.guide) and (self.sky == other.sky)):
                return 0
        return 1
        
    def __hash__(self):
        return hash((self.idn, self.standard, self.guide, self.sky))
        

class TaipanTile(TaipanPoint):
    """
    Holds information and convenience functions for a Taipan tile configuration
    """

    def __init__(self, ra, dec, field_id=None, pk=None, usposn=None,
                 pa=0.0, mag_min=None, mag_max=None):
        """
        Parameters
        ----------
        ra, dec : float, degrees
            Target RA, Dec in degrees
        field_id : int, None
            field_id this tile belongs to, defaults to None.
        pk : int, optional
            Tile primary key (from database), defaults to None.
        usposn : 3-tuple of float, optional
            Tile position in (x,y,z) on the unit sphere, defaults to None.
        pa : float
            Position angle of tile. Defaults to 0.0.
        mag_min, mag_max : float, optional
            Minimum and maximum magnitudes of targets to be assigned to this
            tile, mostly for the benefit of FunnelWeb. Defaults to None.
        """
        #Initialise the base class
        TaipanPoint.__init__(self, ra, dec, usposn)
        #NB this should probably *always* happen as part of TaipanPoint
        #!!! Ask Marc why not?
        if self.usposn is None:
            self.compute_usposn()
        
        self._fibres = {}
        for i in FIBRES_NORMAL:
            self._fibres[i] = None
        for i in FIBRES_GUIDE:
            self._fibres[i] = None
        # self._fibres = self.fibres(fibre_init)
        self._field_id = None
        self._pk = None
        self._mag_min = None
        self._mag_max = None
        self._pa = 0.0

        # Insert the passed values
        # Doing it like this forces the setter functions to be
        # called, which provides error checking
        self.field_id = field_id
        self.pk = pk
        self.pa = pa

    def __str__(self):
        string = 'TP TILE RA %3.1f Dec %2.1f' % (self.ra, self.dec)
        if self._field_id is not None:
            string += ' (f %d)' % (self.field_id, )
        return string

    def __repr__(self):
        string = 'TP TILE RA %3.1f Dec %2.1f' % (self.ra, self.dec)
        if self._field_id is not None:
            string += ' (f %d)' % (self.field_id,)
        return string

    @property
    def fibres(self):
        """:obj:`dict`: Fibre assignments

        ``fibres`` takes the form of a dictionary, with keys corresponding
        to fibre IDs (as per :any:`FIBRES_NORMAL` and :any:`FIBRES_GUIDE`),
        and values being either a :any:`TaipanTarget` object, or special
        string(s) corresponding to special cases.

        Note that the fibres dictionary shouldn't be directly written to;
        :any:`assign_fibre` should be used instead. Direct assignment to the
        ``fibres`` dictionary may result in unexpected behaviour.

        Raises
        ------
        Exception
            Raised if the attempted value of ``fibres`` is not a dict, with
            keys matching :any:`FIBRES_NORMAL` + :any:`FIBRES_GUIDE`.
        """
        return self._fibres

    @fibres.setter
    def fibres(self, d):
        if (not isinstance(d, dict) or [i for
                                        i in d].sort() != [i for i in
                                                           FIBRES_NORMAL +
                                                           FIBRES_GUIDE
                                                           ].sort()):
            raise Exception('Tile fibres must be a dictionary'
                            ' with keys %s' % (
                                str(sorted([i for i in FIBRES_NORMAL +
                                            FIBRES_GUIDE])), )
                            )
        self._fibres = d

    @property
    def pa(self):
        """:obj:`int`: Tile position angle (PA)

        Raises
        ------
        ValueError
            Raised if ``pa`` outside the allowed range :math:`[0,360)`."""
        return self._pa

    @pa.setter
    def pa(self, p):
        p = float(p)
        if p < 0.0 or p >= 360.0:
            raise ValueError('PA must be 0 <= pa < 360')
        self._pa = p

    @property
    def field_id(self):
        """:obj:`int`: Field ID for this particular tile.

        In Taipan nomenclature, a *field* is simply a pre-defined position on
        the sky. Any number of tiles may be generated for a given field.
        """
        return self._field_id

    @field_id.setter
    def field_id(self, p):
        if p is None:
            return
            # This means field_id cannot be deleted, only altered
        p = int(p)
        self._field_id = p

    @property
    def pk(self):
        """:obj:`int`: Unique tile identifier

        Designed for use with :any:`taipandb`. Should be unique, although no
        attempt to enforce this is done within Python.
        """
        return self._pk

    @pk.setter
    def pk(self, p):
        if p is None:
            return
            # This means PK cannot be deleted, only altered
        p = int(p)
        self._pk = p

    @property
    def mag_max(self):
        """:obj:`float`: Maximum target magnitude allowed on this tile

        Raises
        ------
        AssertionError
            Raised if ``mag_max`` is outside the allowed range
            :math:`(-10,30)`."""
        return self._mag_max

    @mag_max.setter
    def mag_max(self, m):
        if m:
            assert (-10 < m < 30), "mag_max outside valid range"
        self._mag_max = m

    @property
    def mag_min(self):
        """:obj:`float`: Minimum target magnitude allowed on this tile

        Raises
        ------
        AssertionError
            Raised if ``mag_min`` is outside the allowed range
            :math:`(-10,30)`."""
        return self._mag_min

    @mag_min.setter
    def mag_min(self, m):
        if m:
            assert (m > -10 and m < 30), "mag_min outside valid range"
        self._mag_min = m

    def generate_json_dict(self, level=TILE_CONFIG_FILE_VERSION):
        """
        Generate a dictionary that represents the JSON configuration file
        used for passing this tile throughout the Taipan architecture
        
        Note that scheduling based fields (e.g. times, statuses) will *not*
        be included. These should be computed/inserted after this base-level
        dictionary is generated.

        Returns
        -------
        :obj:`dict`
            The dictionary representing the JSON configuration file for this
            tile
        :int:`level`
            The version number of the JSON format to use. Valid options are:

            1 - minimal (v1)
            2 - standard (v2)
        """
        level = int(level)
        if level < 1 or level > 2:
            raise ValueError('Invalid level requested ({})'.format(level))

        json_dict = dict()

        # Top-level information
        # json_dict['schemaID'] = TILE_CONFIG_FILE_VERSION
        json_dict['schemaID'] = level
        json_dict['instrumentName'] = INSTRUMENT_NAME
        json_dict['filePurpose'] = FILE_PURPOSE
        json_dict['origin'] = [{
            'name': str(SOFTWARE_TYPE),
            'software': 'executable.name.here',
            'version': str(VERSION),
            'execDate': datetime.datetime.now().strftime(JSON_DTFORMAT_TZ)
        }]
        json_dict['configFormatVersion'] = 'what.is.this'

        # Tile configuration
        json_dict['tilePK'] = self.pk
        json_dict['fieldID'] = self.field_id
        json_dict['fieldCentre'] = {
            'ra': self.ra,
            'dec': self.dec,
        }

        if level == 1:
            json_dict['targets'] = [
                {
                    'sbID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    # 'pmRA': tgt.pm_ra,
                    # 'pmDec': tgt.pm_dec,
                    # 'mag': tgt.mag,
                    'targetID': tgt.idn,
                    # 'type': 'science',
                } for b, tgt in self.get_assigned_targets_science(
                    return_dict=True).items()
            ]
            json_dict['targets'] += [
                {
                    'sbID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    # 'pmRA': tgt.pm_ra,
                    # 'pmDec': tgt.pm_dec,
                    # 'mag': tgt.mag,
                    'targetID': tgt.idn,
                    # 'type': 'std_star',
                } for b, tgt in self.get_assigned_targets_standard(
                    return_dict=True).items()
            ]
            json_dict['guideStars'] = [
                {
                    'sbID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    # 'pmRA': tgt.pm_ra,
                    # 'pmDec': tgt.pm_dec,
                    # 'mag': tgt.mag,
                    'targetID': tgt.idn,
                    # 'type': 'guide'
                } for b, tgt in self.get_assigned_targets_guide(
                    return_dict=True).items()
            ]
            json_dict['sky'] += [
                {
                    'sbID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    # 'pmRA': tgt.pm_ra,
                    # 'pmDec': tgt.pm_dec,
                    # 'mag': tgt.mag,
                    'targetID': tgt.idn,
                    # 'type': 'sky'
                } for b, tgt in self.get_assigned_targets_sky(
                    return_dict=True).items()
            ]
        elif level == 2:
            json_dict['targets'] = [
                {
                    'bugLemoID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    'pmRA': tgt.pm_ra,
                    'pmDec': tgt.pm_dec,
                    'mag': tgt.mag,
                    'targetID': tgt.idn,
                    'type': 'science',
                } for b, tgt in self.get_assigned_targets_science(
                    return_dict=True).items()
            ]
            json_dict['targets'] += [
                {
                    'bugLemoID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    'pmRA': tgt.pm_ra,
                    'pmDec': tgt.pm_dec,
                    'mag': tgt.mag,
                    'targetID': tgt.idn,
                    'type': 'std_star',
                } for b, tgt in self.get_assigned_targets_standard(
                    return_dict=True).items()
            ]
            json_dict['targets'] += [
                {
                    'bugLemoID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    'pmRA': tgt.pm_ra,
                    'pmDec': tgt.pm_dec,
                    'mag': tgt.mag,
                    'targetID': tgt.idn,
                    'type': 'guide'
                } for b, tgt in self.get_assigned_targets_guide(
                    return_dict=True).items()
            ]
            json_dict['targets'] += [
                {
                    'bugLemoID': b,
                    'ra': tgt.ra,
                    'dec': tgt.dec,
                    'pmRA': tgt.pm_ra,
                    'pmDec': tgt.pm_dec,
                    'mag': tgt.mag,
                    'targetID': tgt.idn,
                    'type': 'sky'
                } for b, tgt in self.get_assigned_targets_sky(
                    return_dict=True).items()
            ]

        # Router information
        json_dict['routable'] = 'Unknown'

        return json_dict



    def priority(self):
        """
        Calculate the priority ranking of this tile.

        Priority ranking is determined as the simple sum of the priority of
        all assigned science targets.

        Returns
        -------
        int
            The priority ranking of the tile.
        """
        priority = sum([t.priority for t in self.fibres 
                        if isinstance(t, TaipanTarget)
                        and not t.guide
                        and not t.standard])
        return priority

    def difficulty(self):
        """
        Calculate the difficulty ranking of this tile.

        Difficulty ranking is computed as the sum of
        the difficulties of the TaipanTargets within the tile.
        """
        difficulty = sum([t.difficulty for t in self.fibres 
                          if isinstance(t, TaipanTarget)
                          and not t.guide
                          and not t.standard])
        return difficulty

    def remove_duplicates(self, assigned_targets):
        """
        Un-assign any targets appearing in ``assigned_targets`` that are
        attached to a fibre in this tile.

        Parameters
        ----------    
        assigned_targets : list of :class:`TaipanTarget`
            A list or set of already-assigned
            TaipanTargets.

        Returns
        -------    
        removed_targets : list of :class:`TaipanTarget`
            A list of TaipanTargets that have been removed
            from this tile.
        """

        # Find all the targets in this tile that are in assigned_targets
        removed_targets_t = [t for t in self.fibres 
                             if self.fibres[t] in assigned_targets]
        removed_targets = [self.fibres[t] for t in removed_targets_t]
        for t in removed_targets_t:
            self.fibres[t] = None
        # Remove these objects from the tile
        return removed_targets

    def unassign_fibre(self, fibre):
        """
        Un-assign a fibre.

        Parameters
        ----------    
        fibre : int
            The fibre to un-assign.

        Returns
        -------    
        removed_target : :class:`TaipanTarget`
            The TaipanTarget/string removed from the fibre. If the
            fibre was already empty, None is returned.
        """
        removed_target = self._fibres[fibre]
        self._fibres[fibre] = None
        return removed_target

    def compute_fibre_posn(self, fibre):
        """
        Compute a fibre position.

        Parameters
        ----------    
        fibre : int
            The fibre to compute the position of.

        Returns
        -------    
        ra, dec : float
            The position of the fibre on the sky in degrees.
        """
        fibre = int(fibre)
        if fibre not in BUGPOS_MM:
            raise ValueError('Fibre does not exist in BUGPOS listing')

        fibre_offset = BUGPOS_OFFSET[fibre]
        pos = compute_offset_posn(self.ra, self.dec,
                                  fibre_offset[0],  # Fibre distance from
                                                    # tile centre
                                  (fibre_offset[1] + self.pa) % 360.  # Account
                                                                      # for tile
                                                                      # PA
                                  )
        return pos
        
    def compute_fibre_usposn(self, fibre):
        """
        Compute a fibre position on the unit sphere.

        Parameters
        ----------    
        fibre : int
            The fibre to compute the position of.

        Returns
        -------    
        [x,y,z] : float
            The position of the fibre on the sky on the unit sphere.
        """
        radec = self.compute_fibre_posn(fibre)
        return polar2cart(radec)

    def compute_fibre_travel(self, fibre):
        """
        Compute the distance a fibre is from its home position

        Parameters
        ----------
        fibre : int
            Fibre number

        Returns
        -------
        dist : float, arcsecs
            The number of arcsecs the fibre is from home. Returns
            :any:`numpy.nan` if this can't be computed (e.g. sky fibre).
        """
        fibre_pos = self.compute_fibre_posn(fibre)
        if isinstance(self.fibres[fibre], TaipanTarget):
            move_dist = dist_points(fibre_pos[0], fibre_pos[1],
                                    self.fibres[fibre].ra,
                                    self.fibres[fibre].dec)
            return move_dist
        return np.nan

    def get_assigned_targets(self, return_dict=False):
        """
        Return a list of all :any:`TaipanTarget` currently assigned to this
        tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets : list or dict of :class:`TaipanTarget`
            The list of TaipanTargets currently assigned to
            this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.items()
                            if isinstance(t, TaipanTarget)}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()

    def get_assigned_targets_science(self, return_dict=False,
                                     include_science_standards=True,
                                     only_science_standards=False):
        """
        Return a list of science TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.
            
        include_science_standards : bool
            Do we include here any targets that happen to be both science and
            standards?
    
        include_science_standards : bool
            Only include here any targets that happen to be both science and
            standards.

        Returns
        -------    
        assigned_targets : list or dict of :class:`TaipanTarget`
            The list of science TaipanTargets currently assigned
            to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.items()
                            if isinstance(t, TaipanTarget)}
        if include_science_standards:
            assigned_targets = {f: t for (f, t) in assigned_targets.items()
                                if t.science}
        elif only_science_standards:
            assigned_targets = {f: t for (f, t) in assigned_targets.items()
                                if t.science and t.standard}
        else:
            assigned_targets = {f: t for (f, t) in assigned_targets.items()
                                if t.science and not t.standard}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()

    def count_assigned_targets_science(self, include_science_standards=True):
        """
        Count the number of science targets assigned to this tile.

        Parameters
        ----------    

        include_science_standards : bool
            Do we include here any targets that happen to be both science and
            standards?

        Returns
        -------    
        no_assigned_targets : int
            The number of science targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_science(include_science_standards=include_science_standards))
        return no_assigned_targets

    def get_assigned_targets_standard(self, return_dict=False):
        """
        Return a list of standard TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets : list or dict of :class:`TaipanTarget`
            The list of standard TaipanTargets currently 
            assigned to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.items()
                            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.items()
                            if t.standard}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()

    def count_assigned_targets_standard(self):
        """
        Count the number of standard targets assigned to this tile.

        Returns
        -------    
        no_assigned_targets : int
            The number of standard targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_standard())
        return no_assigned_targets

    def get_assigned_targets_guide(self, return_dict=False):
        """
        Return a list of guide :any:`TaipanTarget` currently assigned to this
        tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets : list or dict of :class:`TaipanTarget`
            The list of guide TaipanTargets currently 
            assigned to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.items()
                            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.items()
                            if t.guide}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()

    def count_assigned_targets_guide(self):
        """
        Count the number of guide targets assigned to this tile.

        Returns
        -------    
        no_assigned_targets : int
            The number of guide targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_guide())
        return no_assigned_targets
        
    def get_assigned_targets_sky(self, return_dict=False):
        """
        Return a list of sky TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets : list or dict of :class:`TaipanTarget`
            The list of sky TaipanTargets currently 
            assigned to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.iteritems()
                            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.iteritems()
                            if t.sky}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()

    def count_assigned_targets_sky(self):
        """
        Count the number of sky targets assigned to this tile.

        Returns
        -------    
        no_assigned_targets : int
            The number of sky targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_sky())
        return no_assigned_targets

    def count_assigned_fibres(self):
        """
        Count the number of assigned fibres on this tile.

        Returns
        -------    
        assigned_fibres : int
            The integer number of empty fibres.
        """
        assigned_fibres = len([f for f in self._fibres.values()
            if f is not None])
        return assigned_fibres

    def count_empty_fibres(self):
        """
        Count the number of empty fibres on this tile.

        Returns
        -------    
        empty_fibres : int
            The integer number of empty fibres.
        """
        empty_fibres = len([f for f in self._fibres.values() if f is None])
        return empty_fibres

    def get_assigned_fibres(self):
        """
        Get the fibre numbers that have been assigned a target/sky.

        Returns
        -------    
        assigned_fibres_list : list of ints
            The list of fibre identifiers which
            have a target/sky assigned.
        """
        assigned_fibres_list = [f for f in self._fibres if f is not None]
        return assigned_fibres_list

    def excluded_targets(self, tgts):
        """
        Calculate which targets are excluded by targets already assigned.

        Compute which targets in the input list are excluded from this tile,
        because they would violate the FIBRE_EXCLUSION_DIAMETER of targets
        already assigned.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            List of targets to check against.

        Returns
        -------    
        excluded_targets : list of :class:`TaipanTarget`
            A subset of tgts, composed of targets which may
            not be assigned to this tile.
        """
        excluded_tgts = list(set([t.excluded_targets(tgts) 
            for t in self.get_assigned_targets()]))
        return excluded_tgts

    def available_targets(self, tgts, leafsize=BREAKEVEN_KDTREE):
        """
        Calculate which targets are within this tile.

        This function does not perform any exclusion checking.

        Parameters
        ----------    
        tgts : list of :class:`TaipanTarget`
            List of targets to check against.
        leafsize : int
            Leaf size for bulk distance calculations. Defaults to
            :any:`BREAKEVEN_KDTREE`.

        Returns
        -------    
        available_targets : list of :class:`TaipanTarget`
            A subset of tgts, composed of targets which are
            within the radius of this tile.
        """
        # available_targets = [t for t in tgts
        #   if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]
        available_targets = targets_in_range(self.ra, self.dec, tgts,
                                             TILE_RADIUS, leafsize=leafsize)
        return available_targets

    def calculate_tile_score(self, method='completeness',
                             combined_weight=1.0, disqualify_below_min=True,
                             exp_base=3.0):
        """
        Compute a ranking score for this tile.

        Tiling algorithms such as 'greedy' require the selection of the
        highest-ranked tile within a tile set for inclusion in the final
        tiling output. This function will calculate the ranking score
        of the calling tile according to the passed method. 

        Available ranking methods are as follows:
        
        'completeness' -- Simple number of assigned science targets within the
        tile.
        
        'difficulty-sum' -- The cumulative difficulty of the assigned targets.
        
        'difficulty-prod' -- The difficulty product of assigned targets.
        
        'priority-sum' -- The cumulative priority of the assigned targets.
        
        'priority-expsum' -- The cumulative priority of the assigned targets,
        raised to an exponential power (i.e. the sum of ``exp_base**priority``).

        'priority-prod' -- The priority product of the assigned targets.

        'combined-weighted-sum' -- The sum of a combined weight of difficulty
        and priority, as given by the combined_weight variable.

        'combined-weighted-prod' -- The product of a combined weight of
        difficulty and priority.

        Parameters
        ----------
        method : str
            String denoting the ranking method to be used. See above for
            details. Defaults to 'completeness'.

        combined_weight : float
            Denotes the combined weighting
            to be use for combined-weighted-sum/prod. Defaults to 1.0.
            
        disqualify_below_min : bool
            Denotes whether to
            rank tiles with a number of guides below :any:`GUIDES_PER_TILE_MIN`,
            a number of standards below :any:`STANDARDS_PER_TILE_MIN` or a
            number of skies below :any:`SKY_PER_TILE_MIN` score of 0.
            Defaults to True.
            
        exp_base : float
            The exponential base used for ``priority-expsum`` tile scores.
            Defaults to 3.0.

        Returns
        -------    
        ranking_score : float
            The ranking score of this tile. Will always return
            a float, even if the ranking could be expressed as an integer. A
            higher score denotes a better-ranked tile.
        """
        SCORE_METHODS = [
            'completeness',
            'difficulty-sum',
            'difficulty-prod',
            'priority-sum',
            'priority-expsum',
            'priority-prod',
            'combined-weighted-sum',
            'combined-weighted-prod',
        ]
        if method not in SCORE_METHODS:
            raise ValueError('Scoring method must be one of %s' 
                % str(SCORE_METHODS))
        
        # Count the number of sky fibres that have been allocated, but have 
        # not yet been assigned targets. This must be considered in cases
        # where we aren't allocating actual targets to the sky fibres, but
        # are still marking
        allocated_but_not_assigned_sky = len([fibre for fibre in self.fibres 
                                              if self._fibres[fibre] == "sky"])
                                                                         
        # Bail out now if tile doesn't meet guide/standard/sky requirements
        # i.e. if the tile does not have the minimum number of guides,  
        # standards, or either the minimum allocated/assigned sky fibres
        # (that it is acceptable to have > SKY_PER_TILE fibres assigned with
        # "sky" or allocated with actual sky targets)
        if disqualify_below_min and (self.count_assigned_targets_guide() 
            < GUIDES_PER_TILE_MIN or self.count_assigned_targets_standard()
            < STANDARDS_PER_TILE_MIN or (self.count_assigned_targets_sky() 
            < SKY_PER_TILE_MIN and allocated_but_not_assigned_sky 
            < SKY_PER_TILE_MIN)):
            return 0.

        # Get all the science targets
        targets_sci = self.get_assigned_targets_science()
        
        # If there are no science targets... we obviously have no score!
        if len(targets_sci) == 0:
            return 0.

        # Perform the calculation
        if method == 'completeness':
            ranking_score = len(targets_sci)
        elif method == 'difficulty-sum':
            ranking_score = sum([t.difficulty for t in targets_sci])
        elif method == 'difficulty-prod':
            ranking_score = prod([t.difficulty for t in targets_sci])
        elif method == 'priority-sum':
            ranking_score = sum([t.priority for t in targets_sci])
        elif method == 'priority-expsum':
            ranking_score = sum([exp_base**t.priority for t in targets_sci])
        elif method == 'priority-prod':
            ranking_score = prod([t.priority for t in targets_sci])
        elif 'combined-weighted' in method:
            max_difficulty = float(max([t.difficulty for t in targets_sci] +
                                       [1]))  # Stops NaN if all diffs are 0
            ranking_list = np.asarray([t.difficulty 
                for t in targets_sci])/max_difficulty + combined_weight * np.asarray(
                [t.priority for t in targets_sci]) / float(TARGET_PRIORITY_MAX)
            if '-sum' in method:
                ranking_score = sum(ranking_list)
            elif '-prod' in method:
                ranking_score = prod(ranking_list)

        # Shift the product ranking scores down by one, so a no-target
        # tile returns 0
        if '-prod' in method:
            ranking_score -= 1.

        ranking_score = float(ranking_score)
        return ranking_score

    def set_fibre(self, fibre, tgt):
        """
        Explicitly assign a TaipanTarget to a fibre on this tile.

        Parameters
        ----------    
        fibre : int
            The fibre to assign to.
        tgt : :class:`TaipanTarget`
            The TaipanTarget, or None, to assign to this fibre. Also, the
            special string 'sky' can be assigned to denote a fibre is
            assigned to sky observations.
        """
        fibre = int(fibre)
        if fibre not in BUGPOS_OFFSET:
            raise ValueError('Invalid fibre')
        if not(tgt is None or tgt == 'sky' or isinstance(tgt, TaipanTarget)):
            raise ValueError('tgt must be a TaipanTarget, "sky" or None')
        self._fibres[fibre] = tgt
        return

    def assign_sky(self):
        """
        Assign sky fibres by randomly selecting one fibre in each of the
        pre-defined quadrants of the tile.

        The aim of this sky assignment method is to avoid introducing
        systematics by forcing a pseudo-random selection of which fibre to
        assign to a sky target. The algorithm works as follows:

        - The tile is divided into a set of annuli, using the radial boundaries
          specified by :any:`QUAD_RADII`.
        - Each annuli is split into :any:`QUAD_PER_RADII` chunks. The values of
          :any:`QUAD_RADII` and :any:`QUAD_PER_RADII` have been tuned such that
          each resulting tile segment is of equal area.
        - Within each segment (i.e. for each sub-list within
          :any:`FIBRES_PER_QUAD`), one fibre is chosen at random and assigned
          the special value 'sky'.
        """
        logging.info('Assigning sky fibres')

        # Remove existing sky fibres
        for f in self._fibres.keys():
            if self._fibres[f] == 'sky':
                _ = self.unassign_fibre(f)

        def random_or_none(l):
            try:
                return random.choice(l)
            except IndexError:
                return None

        sky_fibres = []
        assigned_sky = 0
        while assigned_sky < SKY_PER_TILE:
            assigned_this_pass = 0
            sky_fibres_this_pass = [
                random_or_none([x for x in l if x in FIBRES_NORMAL and
                               self._fibres[x] is None and
                                x not in sky_fibres]) for
                l in FIBRES_PER_QUAD
            ]
            assigned_sky += np.count_nonzero(sky_fibres_this_pass)
            assigned_this_pass = np.count_nonzero(sky_fibres_this_pass)
            sky_fibres += sky_fibres_this_pass
            if assigned_this_pass == 0:
                break

        for f in [_ for _ in sky_fibres if _ is not None]:
            self._fibres[f] = 'sky'
            logging.debug('Added sky to fibre %d' % f)

    def assign_sky_seed(self, seed_fibre=None):
        """
        Assign sky fibres in a psuedo-random fashion to (attempt to) guarantee
        relatively even sky coverage.

        The algorithm for this procedure was developed by Nuria Lorente for
        the SAMI survey:
        1) Select a 'seed' normal fibre at random. This is the first sky fibre.
        2) Select the fibre farthest from the 'seed' fibre as a new sky fibre.
        3) Repeat step 2 by selecting the (on average) farthest fibre from the
        existing sky fibres as a new sky fibre, until the correct number of
        sky fibres are assigned (or we run out of fibres)

        Returns
        -------
        Nil. Sky fibres are updated on the parent tile.
        """
        sky_fibres = []
        logging.info('Assigning sky fibres')

        # Remove existing sky fibres
        for f in self._fibres.keys():
            if self._fibres[f] == 'sky':
                _ = self.unassign_fibre(f)

        if seed_fibre is None:
            # Pick a random seed fibre & append to fibre list
            sky_fibres.append(random.choice(FIBRES_NORMAL))
        else:
            if seed_fibre not in FIBRES_NORMAL:
                raise ValueError('Invalid seed fibre (%d) passed to '
                                 'assign_sky' % seed_fibre)
            sky_fibres.append(seed_fibre)
        logging.debug('Added sky to fibre %d' % sky_fibres[-1])

        while len(sky_fibres) < SKY_PER_TILE and np.any(
            [self._fibres[f] is None for f in FIBRES_NORMAL]
        ):
            # Compute the farthest fibre from the current sky fibres
            avg_dists = {f: np.average([
                                       np.abs(dist_points(
                                           *self.compute_fibre_posn(s)+
                                            self.compute_fibre_posn(f)) -
                                              2.75*PATROL_RADIUS) for s in
                                       sky_fibres
                                       ])
                         for f in FIBRES_NORMAL if
                         self._fibres[f] is None and
                         f not in sky_fibres}
            sky_fibres.append(
                min(avg_dists.items(), key=operator.itemgetter(1))[0]
            )
            logging.debug('Added sky to fibre %d' % sky_fibres[-1])

        for f in sky_fibres:
            self.set_fibre(f, 'sky')

        return

    def assign_fibre(self, fibre, candidate_targets, 
                     check_patrol_radius=True, check_tile_radius=True,
                     recompute_difficulty=True,
                     order_closest_secondary=True,
                     method='combined_weighted',
                     combined_weight=1.0,
                     sequential_ordering=(2,1,0)):
        """
        Assign a target from the target list to the given fibre.

        Assign a target from the target list to the given fibre in this 
        tile using any one of a multitude of methods. Note that this 
        function will first set the fibre to be empty, and then attempt 
        to fill it.

        This function will not assign a target to a fibre if no valid
        target is available. This is expected behaviour, and NO WARNING
        will be given. Therefore, if no target is available to the fibre,
        the fibre will have been reset to be empty.

        The methods are as follows:
        
        *closest* - Assign the target closest to the bug rest position.
        
        *most_diffucult* - Assign the target within the patrol radius
        that has the most other targets within its exclusion radius 
        (:any:`FIBRE_EXCLUSION_DIAMETER`).
        
        *priority* - Assign the highest-priority target within the patrol
        radius.
        
        *combined_weighted* - Prioritise targets within the patrol radius
        based on a weighted combination of most_diffucult and
        priority. Uses the combined_weight keyword (see below)
        as the weighting.
        
        *sequential* - Prioritise targets for this fibre based on closest,
        most_difficult and priority in the order given by
        ``sequential_ordering`` argument (see below).

        Parameters
        ----------    
        fibre : int
            The ID of the fibre to be assigned.
            
        candidate_targets : list of :class:`TaipanTarget`
            A list of potential TaipanTargets to assign.
            
        check_tile_radius : Boolean, optional
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile.
            Defaults to True.
            
        check_patrol_radius : bool
            Denotes whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the patrol radius of this fibre.
            Defaults to True.
            
        check_tile_radius : bool
            Denotes whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty : bool
            Denotes whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within :any:`FIBRE_EXCLUSION_RADIUS`
            + :any:`TILE_RADIUS` of the tile centre; targets for assignment are
            only within :any:`TILE_RADIUS`). Defaults to True.

        method : str
            The method to be used. Must be one of the methods
            specified above. Defaults to ``"combined_weighted"``.
            
        order_closest_secondary : bool
            A Boolean value describing if candidate
            targets should be ordered by distance from the fibre rest position
            as a secondary consideration. Has no effect if ``method='closest'``.
            Defaults to True.
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering : 3-tuple of int
            A three-tuple or length-3 list determining
            in which order to sequence potential targets. It must
            contain the integers 0, 1 and 2, which correspond to the
            position of closest (0), most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (2, 1, 0).

        Returns
        -------    
        remaining_targets : list of :class:`TaipanTarget`
            The list of candidate_targets, with the newly-
            assigned target removed. If the assignment is unsuccessful, 
            the entire candidate_targets list is returned.
            
        fibre_former_tgt : :class:`TaipanTarget`
            The target that was removed from this fibre
            during the allocation process. Returns None if no object was on
            this fibre originally.

        Raises
        ------
        ValueError
            Raised if the argument restrictions listed in ``Parameters`` are
            violated.
        """
        FIBRE_ALLOC_METHODS = [
            'closest',
            'most_difficult',
            'priority',
            'combined_weighted',
            'sequential',
        ]

        # Input checking & type-casting
        fibre = int(fibre)
        if fibre not in BUGPOS_MM:
            raise ValueError('Fibre does not exist in BUGPOS listing')
        if method not in FIBRE_ALLOC_METHODS:
            raise ValueError('Invalid allocation method passed! Must'
                             ' be one of %s' % (str(FIBRE_ALLOC_METHODS), ))
        combined_weight = float(combined_weight)
        if combined_weight <= 0.:
            raise ValueError('Combined weight must be > 0')
        if sorted(list(sequential_ordering)) != [0,1,2]:
            raise ValueError('sequential_ordering must be a list or 3-tuple '
                             'containing the integers 0, 1 and 2 once each')
        if recompute_difficulty and not check_tile_radius:
            raise ValueError('recompute_difficulty requires a full '
                             'target list (i.e. that would'
                             ' require check_tile_radius)')

        # Reset the fibre to be empty
        fibre_former_tgt = self._fibres[fibre]
        self._fibres[fibre] = None
        fibre_posn = self.compute_fibre_posn(fibre)
        # Analyze what targets are available
        existing_targets = self.get_assigned_targets()
        candidate_targets_return = candidate_targets[:]
        candidates_this_fibre = candidate_targets[:]
        if check_patrol_radius:
            candidates_this_fibre = [t for t in candidates_this_fibre 
                                     if t.dist_point(fibre_posn) <
                                     PATROL_RADIUS]
        if check_tile_radius:
            candidates_this_fibre = [t for t in candidates_this_fibre
                                     if t.dist_point((self.ra, self.dec)) <
                                     TILE_RADIUS]

        # Remove targets that are too close to already assigned targets
        candidates_this_fibre = [t for t in candidates_this_fibre
            if not t.is_target_forbidden(existing_targets)] 
        # Bail out now if no targets exist
        if len(candidates_this_fibre) == 0:
            return candidate_targets, fibre_former_tgt

        # Assign target to fibre
        # This code segment either finds the closest target, or, if
        # order_closest_secondary is given, re-orders the target list by
        # distance
        if method == 'closest':
            i = np.argmin([t.dist_point(fibre_posn) for t 
                in candidates_this_fibre])
            tgt = candidates_this_fibre[i]
            self._fibres[fibre] = candidate_targets_return.pop(
                candidate_targets_return.index(tgt))
            return candidate_targets_return
        elif order_closest_secondary:
            candidates_this_fibre.sort(key=lambda x: x.dist_point(fibre_posn))
        
        # This code handles the other possible selection criteria
        if method == 'sequential':
            distance_list = [t.dist_point(fibre_posn) 
                for t in candidates_this_fibre]
            distance_list = [max(distance_list) - d 
                for d in distance_list]
            difficulty_list = [t.difficulty 
                for t in candidates_this_fibre]
            priority_list = [t.priority for t in candidates_this_fibre]
            lists = [distance_list, difficulty_list, priority_list]
            maxes = [max(distance_list), max(difficulty_list), 
                max(priority_list)]
            ranking_list = [maxes[sequential_ordering[1]] * maxes[
                sequential_ordering[2]] * lists[
                sequential_ordering[0]][i] 
                + lists[sequential_ordering[2]][i] * lists[
                sequential_ordering[1]][i]
                + lists[sequential_ordering[2]][i]
                for i in range(len(difficulty_list))]
            i = np.argmax(ranking_list)
            tgt = candidates_this_fibre[i]
            self._fibres[fibre] = candidate_targets_return.pop(
                candidate_targets_return.index(tgt))
        elif method == 'most_difficult':
            i = np.argmax([t.difficulty for t in candidates_this_fibre])
            tgt = candidates_this_fibre[i]
            self._fibres[fibre] = candidate_targets_return.pop(
                candidate_targets_return.index(tgt))
        elif method == 'priority':
            i = np.argmax([t.priority for t in candidates_this_fibre])
            tgt = candidates_this_fibre[i]
            self._fibres[fibre] = candidate_targets_return.pop(
                candidate_targets_return.index(tgt))
        elif method == 'combined_weighted':
            max_excluded_tgts = float(max([t.difficulty
                for t in candidates_this_fibre])) / float(TARGET_PRIORITY_MAX)
            i = np.argmax([combined_weight*t.priority 
                + float(t.difficulty)/max_excluded_tgts 
                for t in candidates_this_fibre])
            tgt = candidates_this_fibre[i]
            self._fibres[fibre] = candidate_targets_return.pop(
                candidate_targets_return.index(tgt))

        # Do checking of the returns list
        # This should be removed in production
        # if len(candidate_targets) - len(candidate_targets_return) != 1:
        #     print '### WARNING - assign_fibre has mangled the target list'

        # Recompute target difficulties if requested
        # Only targets within FIBRE_EXCLUSION_DIAMETER of the newly-assigned
        # target need be computed
        if recompute_difficulty:
            compute_target_difficulties(
                targets_in_range(tgt.ra, tgt.dec,
                                 candidate_targets_return,
                                 FIBRE_EXCLUSION_DIAMETER))

        return candidate_targets_return, fibre_former_tgt

    def assign_tile(self, candidate_targets,
                    check_tile_radius=True, recompute_difficulty=True,
                    method='priority', combined_weight=1.0,
                    sequential_ordering=(2, 1),
                    overwrite_existing=False):
        """
        Assign a single target to a tile as a whole, choosing the best fibre
        to assign to.

        This function will attempt to assign targets to an entire tile,
        selecting the best available (i.e. nearest) fibre for the target it
        chooses. Targets are assigned to tile based on the method passed, whilst
        automatically excluding forbidden targets.

        There are four defined methods for choosing targets:
        
        *most_difficult* - Attempt to assign the targets in order of descending
        difficulty, where difficulty is defined as the number of targets
        within :any:`FIBRE_EXCLUSION_DIAMETER` of the target considered.
        NOTE: Difficulty is computed based on the entire candidate_targets
        list, not just those targets within the tile radius.
            
        *priority* - Attempt to assign targets in order of priority. This is the
        default.
            
        *combined_weighted* - Attempt to assign targets based on a weighted
        combination of difficulty and priority, as determined by the
        factor combined_weight (see below).
        
        *sequential* - Attempt to assign targets in order of priority and
        difficulty, based on the ordering given in sequential_ordering 
        (see below).

        Parameters
        ----------    
        candidate_targets : list of :class:`TaipanTarget`
            Objects to consider assigning to this tile.

        check_tile_radius : bool
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty : bool
            Boolean denoting whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within
            :any:`FIBRE_EXCLUSION_DIAMETER` + :any:`TILE_RADIUS` of the tile
            centre; targets for assignment are
            only within :any:`TILE_RADIUS`). Defaults to True.
            
        method : str
            The method to use to choose the target to assign (see above).
            Defaults to 'priority'.
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering : 2-tuple of int
            A two-tuple or length-2 list determining
            in which order to sequence potential targets. It must
            contain the integers 1 and 2, which correspond to the
            position of most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (2, 1).
            
        overwrite_existing : bool
            A Boolean value, denoting whether to overwrite an
            existing target allocation if the best fibre for the chosen target
            already has a target assigned. Defaults to False.

        Returns
        -------    
        candidate_targets : list of :class:`TaipanTarget`
            The list of candidate_targets originally passed, 
            less the target which has been assigned. If no target is assigned,
            the output matches the input.
            
        tile_former_tgt : :class:`TaipanTarget`
            The target that may have been removed from the tile
            during this procedure. Returns None if this did not occur.

        Raises
        ------
        ValueError
            Raised if any of the argument constraints listed in ``Parameters``
            are violated.
        """
        TILE_ALLOC_METHODS = [
            'most_difficult',
            'priority',
            'combined_weighted',
            'sequential',
        ]

        # Input checking and type casting
        if method not in TILE_ALLOC_METHODS:
            raise ValueError('Invalid allocation method passed! Must'
                             ' be one of %s' % (str(TILE_ALLOC_METHODS), ))
        combined_weight = float(combined_weight)
        if combined_weight <= 0.:
            raise ValueError('Combined weight must be > 0')
        if sorted(list(sequential_ordering)) != [1,2]:
            raise ValueError('sequential_ordering must be a list or 2-tuple '
                'containing the ints 1 and 2 once each')
        if recompute_difficulty and not check_tile_radius:
            raise ValueError('recompute_difficulty requires a full '
                             'target list (i.e. that would require '
                             'check_tile_radius)')

        fibre_former_tgt = None

        # Calculate rest positions for all fibres
        # Ensure that permitted_fibres *actually exist* (i.e. consider 150 vs 300)
        fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
                       for fibre in BUGPOS_MM if fibre not in FIBRES_GUIDE and
                       fibre in FIBRES_NORMAL}

        # Trim the candidate list to this tile
        candidate_targets_return = candidate_targets[:]
        candidates_this_tile = candidate_targets[:]
        if check_tile_radius:
            candidates_this_tile = [t for t in candidates_this_tile
                if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]
            # candidates_this_tile = targets_in_range(self.ra, self.dec,
            #   candidates_this_tile, TILE_RADIUS)
        # Abort now if no targets possible
        if len(candidates_this_tile) == 0:
            return candidate_targets, fibre_former_tgt

        # Compute the ranking list for the selection procedure
        # If the ranking method is sequential, compute the equivalent
        # combined_weight and change the method to combined_weighted
        ranking_list = generate_ranking_list(
            candidates_this_tile,
            method=method, combined_weight=combined_weight,
            sequential_ordering=sequential_ordering
        )

        # Search for the best assign-able target
        candidate_found = False
        while not(candidate_found) and len(candidates_this_tile) > 0:
            # print 'Identifying best target...'
            # Search for the best target according to the criterion
            i = np.argmax(ranking_list)
            tgt = candidates_this_tile[i]
            # Check if this target is forbidden - if so, restart the loop
            # This is more efficient that computing all forbidden targets
            # a priori
            if tgt.is_target_forbidden(self.get_assigned_targets()):
                # Remove tgt from consideration
                ranking_list.pop(i)
                candidates_this_tile.pop(i)
                continue
            # print 'Done!'

            # Identify the closest fibre to this target
            # print 'Finding available fibres...'
            # PARALLEL: The following lines take up 25% of this routine.
            # Shouldn't use
            # the GIL if possible for threading.
            fibre_dists = {fibre: tgt.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])
            # print 'Done!'

            # Attempt to make assignment
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
                if (overwrite_existing or
                            self._fibres[permitted_fibres[0]] is None):
                    # Assign the target and 'pop' it from the input list
                    fibre_former_tgt = self._fibres[permitted_fibres[0]]
                    self._fibres[permitted_fibres[0]] = candidate_targets_return.pop(
                        candidate_targets_return.index(tgt))
                    candidate_found = True
                    # print 'Done!'
                    # Update target difficulties if required
                    if recompute_difficulty:
                        fibre = permitted_fibres[0]
                        compute_target_difficulties(targets_in_range(
                            self._fibres[fibre].ra, self._fibres[fibre].dec,
                            candidate_targets_return, FIBRE_EXCLUSION_RADIUS),
                            full_target_list=candidate_targets_return)
            
                else:
                    permitted_fibres.pop(0)


            if not candidate_found:
                # If this point has been reached, the best target cannot be
                # assigned to this tile, so remove it from the
                # candidates_this_tile list
                ranking_list.pop(i)
                candidates_this_tile.pop(i)

        # Do checking of the returns list
        # This should be removed in production
        if candidate_found and len(candidate_targets) - len(
            candidate_targets_return) != 1:
            logging.error('### WARNING - assign_tile has '
                          'mangled the target list')

        return candidate_targets_return, fibre_former_tgt

    def assign_guides(self, guide_targets,
                      target_method='priority',
                      combined_weight=1.0, sequential_ordering=(2, 1),
                      check_tile_radius=True,
                      rank_guides=False):
        """
        Assign guides to this tile.

        Guides are assigned their own special fibres. This function will
        attempt to assign up to :any:`GUIDES_PER_TILE` to the guide fibres. If
        necessary, it will then remove targets from this tile to allow
        up to :any:`GUIDES_PER_TILE_MIN` to be assigned. This is more complex
        than for standards, because we are not removing targets to assign
        standards to their fibres - rather, we have to work out the best
        targets to drop so other fibres are no longer within
        :any:`FIBRE_EXCLUSION_DIAMETER` of the guide
        we wish to assign.

        Parameters
        ----------    
        guide_targets : list of :class:`TaipanTarget`
            The list of candidate guides for assignment.
            
        target_method : str
            The method that should be used to determine the
            lowest-priority target to remove to allow for an extra guide
            star assignment to be made. Values for this input are as for
            the 'method' option in assign_tile/unpick_tile. Defaults to
            'priority'.
            
        combined_weight, sequential_ordering : float, 2-tuple of ints
            Additional control options
            for the specified target_method. See docs for :any:`assign_tile`/
            :any:`unpick_tile` for description. Defaults to 1.0 and (2, 1)
            respectively.
            
        check_tile_radius : bool
            Boolean value, denoting whether to reduce the
            guide_targets list to only those targets within the tile radius.
            Defaults to True.
            
        rank_guides : bool
            Attempt to assign guides in priority order. This allows
            for 'better' guides to be specified. Defaults to False.

        Returns
        -------    
        removed_targets : list of :class:`TaipanTarget`
            A list of TaipanTargets that have been removed to
            make way for guide fibres. If no targets are removed, the empty
            list is returned. Targets are *not* separated by type (science or
            standard).
        """

        removed_targets = []

        # Calculate rest positions for all GUIDE fibres
        fibre_posns = {fibre: self.compute_fibre_posn(fibre) for
                       fibre in FIBRES_GUIDE}

        guides_this_tile = guide_targets[:]
        if check_tile_radius:
            guides_this_tile = [g for g in guides_this_tile
                if g.dist_point((self.ra, self.dec, )) < TILE_RADIUS]

        if rank_guides:
            logging.debug('Sorting input guide list by priority')
            guides_this_tile.sort(key=lambda x: -1 * x.priority)
        else:
            logging.debug('Sorting input guide list by dist to guide fibre')
            # Instead of having randomly ordered guides, let's rank them
            # by the distance to their nearest guide fibre
            guides_this_tile.sort(key=lambda x: np.min([
                x.dist_point(posn) for posn in fibre_posns.values()
            ]))

            # guide_targets_dists = None
            # for posn in fibre_posns.values():
            #     guide_dists = [g.dist_point(posn) for g in guide_targets]
            #     if guide_targets_dists is None:
            #         guide_targets_dists = np.asarray(guide_dists)
            #     else:
            #         guide_targets_dists = np.c_[guide_targets_dists,
            #                                     guide_dists]
            # # Pick the lowest distance value for each target
            # guide_targets_dists = np.min(guide_targets_dists,
            #                              axis=-1).tolist()
            # # Sort the guides by their closest distance to any guide fibre
            # guides_this_tile = [g for (d, g) in
            #                     sorted(zip(guide_targets_dists,
            #                                guides_this_tile),
            #                            key=lambda pair: pair[0])]

            # Assign up to GUIDES_PER_TILE guides
        # Attempt to assign guide stars to this tile
        assigned_guides = len([t for t in self._fibres.values() 
            if isinstance(t, TaipanTarget)
            and t.guide])

        logging.debug('Finding available fibres...')
        while assigned_guides < GUIDES_PER_TILE and len(guides_this_tile) > 0:

            guide = guides_this_tile[0]
            if guide.is_target_forbidden(self.get_assigned_targets()):
                guides_this_tile.pop(0)
                continue

            # Identify the closest fibre to this target
            fibre_dists = {fibre: guide.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])

            # Attempt to make assignment
            logging.debug('Looking to add to fiber...')
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                if self._fibres[permitted_fibres[0]] is None:
                    # Assign the target and 'pop' it from the input list
                    self._fibres[permitted_fibres[0]] = guides_this_tile.pop(0)
                    candidate_found = True
                    assigned_guides += 1
                    # print 'Done!'
                else:
                    permitted_fibres.pop(0)

            if not(candidate_found):
                # If this point has been reached, the best target cannot be
                # assigned to this tile, so remove it from the
                # candidates_this_tile list
                # print 'Candidate not possible!'
                guides_this_tile.pop(0)

        assigned_objs = self.get_assigned_targets()

        if assigned_guides < GUIDES_PER_TILE_MIN:
            logging.debug('Having to strip targets for guides...')
            guides_this_tile = [t for t in guide_targets
                if t not in assigned_objs]
            if check_tile_radius:
                guides_this_tile = [g for g in guides_this_tile
                    if g.dist_point((self.ra, self.dec, )) < TILE_RADIUS]
            
            # For the available guides, calculate the total weight of the
            # targets which may be blocking the assignment of that guide by ways
            # of the fibre exclusion radius
            # Weights are computed according to the passed target_method
            # Work out which targets are obscuring each available guide
            problem_targets = [g.excluded_targets(assigned_objs)
                for g in guides_this_tile]
            # Don't consider guides which are excluded by already-assigned
            # guides
            excluded_by_guides = [i for i in range(len(guides_this_tile)) if
                                  np.any(map(lambda x: x.guide,
                                             problem_targets[i]))]
            guides_this_tile = [guides_this_tile[i] for i in
                                range(len(guides_this_tile)) if
                                i not in excluded_by_guides]
            problem_targets = [problem_targets[i] for
                               i in range(len(problem_targets)) if
                               i not in excluded_by_guides]

            # Compute the total ranking weights for targets blocking the
            # remaining guide candidates
            # Note that, for consistency, we must calculate the target rankings
            # as a combined group, and then sum from that list on a piecewise-
            # basis. Otherwise, when we compute, e.g., a combined_weight
            # ranking, the scaling of the weights if we do the calculation for
            # each sub-list of problem_targets separately
            problem_targets_all = list(set(flatten(problem_targets)))
            ranking_list = generate_ranking_list(
                problem_targets_all,
                method=target_method, combined_weight=combined_weight,
                sequential_ordering=sequential_ordering
            )
            problem_targets_rankings = [np.sum([ranking_list[i] 
                for i in range(len(ranking_list)) 
                if problem_targets_all[i] in pt]) for pt in problem_targets]

            # Assign guides by removing the excluding target(s) with the lowest
            # weighting sum and assigning the guide
            while assigned_guides < GUIDES_PER_TILE_MIN and len(
                guides_this_tile) > 0:
                # Identify the lowest-ranked set of science targets excluding
                # a guide
                i = np.argmin(problem_targets_rankings)
                guide = guides_this_tile[i]
                # Check that the related guide can actually be assigned to an
                # available guide fibre
                fibre_dists = {fibre: guide.dist_point(fibre_posns[fibre])
                    for fibre in fibre_posns}
                permitted_fibres = sorted([fibre for fibre in fibre_dists
                    if fibre_dists[fibre] < PATROL_RADIUS],
                    key=lambda x: fibre_dists[x])
                if len(permitted_fibres) == 0:
                    burn = problem_targets_rankings.pop(i)
                    burn = problem_targets.pop(i)
                    burn = guides_this_tile.pop(i)
                    continue
                # Remove the offending targets from the tile, and assign the
                # guide
                fibres_for_removal = [f for (f, t) in self._fibres.items()
                    if t in problem_targets[i]]
                for f in fibres_for_removal:
                    removed_targets.append(self.unassign_fibre(f))
                self._fibres[permitted_fibres[0]] = guides_this_tile[i]
                assigned_guides += 1
                # Pop these candidates from the lists
                burn = problem_targets.pop(i)
                burn = problem_targets_rankings.pop(i)
                burn = guides_this_tile.pop(i)

        return removed_targets


    def assign_sky_fibres(self, sky_targets, target_method='priority',  
                          combined_weight=1.0,sequential_ordering=(2, 1), 
                          check_tile_radius=True, rank_sky=False):
        """
        Assign sky coordinates to this tile.

        Guides are assigned their own special fibres. This function will
        attempt to assign up to GUIDES_PER_TILE to the guide fibres. If
        necessary, it will then remove targets from this tile to allow
        up to GUIDES_PER_TILE_MIN to be assigned. This is more complex than for
        standards, because we are not removing targets to assign standards to
        their fibres - rather, we have to work out the best targets to drop so
        other fibres are no longer within FIBRE_EXCLUSION_DIAMETER of the guide
        we wish to assign.

        Parameters
        ----------    
        guide_targets : list of :class:`TaipanTarget`
            The list of candidate guides for assignment.
            
        target_method : str
            The method that should be used to determine the
            lowest-priority target to remove to allow for an extra guide
            star assignment to be made. Values for this input are as for
            the 'method' option in assign_tile/unpick_tile. Defaults to
            'priority'.
            
        combined_weight, sequential_ordering : float, 2-tuple of ints
            Additional control options
            for the specified target_method. See docs for assign_tile/
            unpick_tile for description. Defaults to 1.0 and (1,2)
            respectively.
            
        check_tile_radius : Boolean, optional
            Boolean value, denoting whether to reduce the
            guide_targets list to only those targets within the tile radius.
            Defaults to True.
            
        rank_sky : Boolean, optional
            Attempt to assign guides in priority order. This allows
            for 'better' guides to be specified. Defaults to False.

        Returns
        -------    
        removed_targets : list of :class:`TaipanTarget`
            A list of TaipanTargets that have been removed to
            make way for sky fibres. If no targets are removed, the empty
            list is returned. Targets are *not* separated by type (science or
            standard).
        """

        removed_targets = []
        
        # Each TaipanTile object will have a different set of fibres reserved  
        # as sky fibres. Grab these and assign sky "targets" to them
        FIBRES_SKY = [fibre for fibre in self.fibres if self._fibres[fibre] ==
                      "sky"]

        # Calculate rest positions for all sky fibres
        fibre_posns_sky = {fibre: self.compute_fibre_posn(fibre) for fibre in
                           FIBRES_SKY}
        
        # Calculate rest positions for all non-guide fibres
        # DONE: Consideration for avoiding standard fibres
        fibre_posns_all = {fibre: self.compute_fibre_posn(fibre) 
                           for fibre in FIBRES_NORMAL if fibre not in
                           FIBRES_GUIDE and
                           fibre not in
                           self.get_assigned_targets_standard(
                               return_dict=True).keys()
                           }

        # Reset the sky fibres
        for fibre in FIBRES_SKY:
            self._fibres[fibre] = None
        
        # Create a copy
        sky_this_tile = sky_targets[:]
        
        if check_tile_radius:
            sky_this_tile = [g for g in sky_this_tile
                             if g.dist_point((self.ra, self.dec, )) <
                             TILE_RADIUS]
        
        # Sort the sky targets
        if rank_sky:
            logging.debug('Sorting input sky list by priority')
            sky_this_tile.sort(key=lambda x: -1 * x.priority)
        else:
            logging.debug('Sorting input sky list by dist to sky fibre')
            # Instead of having randomly ordered sky, let's rank them
            # by the distance to their nearest sky fibre
            sky_this_tile.sort(key=lambda x: np.min([
                x.dist_point(posn) for posn in fibre_posns_sky.itervalues()
            ]))

        # Assign up to SKY_PER_TILE sky, check how many have already
        # been assigned
        assigned_sky = len([t for t in self._fibres.values() 
            if isinstance(t, TaipanTarget) and t.sky])

        logging.debug('Finding available fibres...')
        # Loop while we have not assigned the required number of sky fibres
        # *but* still have candidate sky targets remaining
        while assigned_sky < SKY_PER_TILE and len(sky_this_tile) > 0:
            # Check that it is possible to place current sky fibre
            sky = sky_this_tile[0]
            if sky.is_target_forbidden(self.get_assigned_targets()):
                sky_this_tile.pop(0)
                continue

            # Identify the closest fibre to this target
            fibre_dists = {fibre: sky.dist_point(fibre_posns_sky[fibre])
                for fibre in fibre_posns_sky}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])
            
            # Attempt to make assignment
            logging.debug('Looking to add to fiber...')
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                if self._fibres[permitted_fibres[0]] == None:
                    # Assign the target and 'pop' it from the input list
                    self._fibres[permitted_fibres[0]] = sky_this_tile.pop(0)
                    candidate_found = True
                    assigned_sky += 1
                    # print 'Done!'
                else:
                    permitted_fibres.pop(0)

            if not(candidate_found):
                # If this point has been reached, the best target cannot be
                # assigned to this tile, so remove it from the
                # candidates_this_tile list
                # print 'Candidate not possible!'
                sky_this_tile.pop(0)

        assigned_objs = self.get_assigned_targets()
        
        # If we have not assigned to the minimum accepted number of sky fibres,
        # but have exhausted all possibilities, we now have to remove science
        # targets to reach the minimum accepted.
        if assigned_sky < SKY_PER_TILE_MIN:
            logging.debug('Having to strip targets for sky...')
            sky_this_tile = [t for t in sky_targets if t not in assigned_objs]
            if check_tile_radius:
                sky_this_tile = [g for g in sky_this_tile
                    if g.dist_point((self.ra, self.dec, )) < TILE_RADIUS]
            
            # For the available sky, calculate the total weight of the targets
            # which may be blocking the assignment of that sky by ways of the
            # fibre exclusion radius
            # Weights are computed according to the passed target_method
            # Work out which targets are obscuring each available sky
            problem_targets = [g.excluded_targets(assigned_objs) for g in
                               sky_this_tile]
            
            # Don't consider sky which are excluded by already-assigned sky
            excluded_by_sky = [i for i in range(len(sky_this_tile)) if
                                  np.any(map(lambda x: x.sky,
                                             problem_targets[i]))]
            sky_this_tile = [sky_this_tile[i] for i in
                                range(len(sky_this_tile)) if
                                i not in excluded_by_sky]
            problem_targets = [problem_targets[i] for
                               i in range(len(problem_targets)) if
                               i not in excluded_by_sky]

            # Compute the total ranking weights for targets blocking the
            # remaining sky candidates. Note that, for consistency, we must
            # calculate the target rankings as a combined group, and then sum
            # from that list on a piecewise-basis. Otherwise, when
            # we compute, e.g., a combined_weight ranking, the
            # scaling of the weights if we do the calculation for each sub-list
            # of  problem_targets separately
            problem_targets_all = list(set(flatten(problem_targets)))
            
            ranking_list = generate_ranking_list(problem_targets_all,
                                                 method=target_method, 
                                                 combined_weight=
                                                 combined_weight,
                                                 sequential_ordering=
                                                 sequential_ordering)
                                                 
            problem_targets_rankings = [np.sum([ranking_list[i] 
                for i in range(len(ranking_list)) 
                if problem_targets_all[i] in pt]) for pt in problem_targets]

            # Assign guides by removing the excluding target(s) with the lowest
            # weighting
            # sum and assigning the sky
            while assigned_sky < SKY_PER_TILE_MIN and len(sky_this_tile) > 0:
                # Identify the lowest-ranked set of science targets excluding
                # a sky
                i = np.argmin(problem_targets_rankings)
                sky = sky_this_tile[i]
                
                # Check related sky can actually be assigned to an available
                # sky fibre
                fibre_dists = {fibre: sky.dist_point(fibre_posns_all[fibre])
                    for fibre in fibre_posns_all}
                    
                permitted_fibres = sorted([fibre for fibre in fibre_dists
                    if fibre_dists[fibre] < PATROL_RADIUS],
                    key=lambda x: fibre_dists[x])
                    
                if len(permitted_fibres) == 0:
                    burn = problem_targets_rankings.pop(i)
                    burn = problem_targets.pop(i)
                    burn = sky_this_tile.pop(i)
                    continue
                    
                # Remove offending targets from the tile, and assign the sky
                fibres_for_removal = [f for (f, t) in self._fibres.iteritems()
                    if t in problem_targets[i]]
                    
                for f in fibres_for_removal:
                    removed_targets.append(self.unassign_fibre(f))
 
                self._fibres[permitted_fibres[0]] = sky_this_tile[i]
                assigned_sky += 1
                
                # Pop these candidates from the lists
                burn = problem_targets.pop(i)
                burn = problem_targets_rankings.pop(i)
                burn = sky_this_tile.pop(i)
        
        return removed_targets
        

    #@profile
    def unpick_tile(self, candidate_targets,
                    standard_targets, guide_targets,
                    sky_targets=None,
                    overwrite_existing=False,
                    check_tile_radius=True, recompute_difficulty=True,
                    method='priority', combined_weight=1.0,
                    sequential_ordering=(2, 1),
                    rank_supplements=False,
                    repick_after_complete=True,
                    consider_removed_targets=True,
                    allow_standard_targets=False,
                    assign_sky_first=True,
                    assign_sky_fibres=False):
        """
        Unpick this tile, i.e. make a full allocation of targets, guides etc.

        To 'unpick' a tile is to make the optimal permitted allocation of 
        available targets, standards, guides and 'skies' to a particular tile.
        In the case of using a single target allocation method, it is more
        efficient to have a stand-alone function, rather than make repeated 
        calls to :any:`assign_fibre` or :any:`assign_tile`, because the ranking
        criterion for targets need only be computed once (although, note that
        this function does call :any:`assign_tile` and :any:`assign_fibre`
        in certain scenarios).

        For custom unpicking strategies, which may use different target 
        selection methods at different stages, construct your own routine that 
        uses calls to :any:`assign_fibre` and/or :any:`assign_tile`.

        This function will attempt to unpick around any existing fibre
        assignments on this tile. Use the ``overwrite_existing`` keyword
        argument to alter this behaviour.

        Target assignment methods are as for :any:`assign_tile`.

        Parameters
        ----------    
        candidate_targets :  :class:`TaipanTarget` list
            Objects to consider
            assigning to this tile. These are the science targets.
            
        standard_targets, guide_targets : :class:`TaipanTarget` list
            Objects to consider assigning to this tile as standards and guides
            respectively. Standards, guides and sky fibres are assigned after
            science targets.
            
        sky_targets : :class:`TaipanTarget` list
            Objects to consider assigning to this tile as sky targets.
            This argument defaults to None.
            
        overwrite_existing : :obj:`bool`
            Boolean, denoting whether to remove all existing
            fibre assignments on this tile before attempting to unpick. Defaults
            to False.
            
        check_tile_radius : :obj:`bool`
            Boolean denoting whether the
            input target lists need to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty : :obj:`bool`
            Boolean denoting whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within
            :any:`FIBRE_EXCLUSION_DIAMETER` +
            :any:`TILE_RADIUS` of the tile centre; targets for assignment are
            only within :any:`TILE_RADIUS`). Defaults to True.
            
        method : str
            The method to use to choose science targets to assign. See the
            documentation for assign_tile for an explanation of the available
            methods. Defaults to 'priority'.
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering : 2-tuple of int
            A two-tuple or length-2 list determining
            in which order to sequence potential targets. It must
            contain the integers 1 and 2, which correspond to the
            position of most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (2, 1).
            
        rank_supplements : bool
            Boolean value, denoting whether for rank the lists
            of standards and guides by their priority. This allows for the
            preferential selection of 'better' guides and standards, if this
            information is encapsulated in their stored priority values.
            Defaults to False.
            
        repick_after_complete : bool
            value denoting whether to invoke this
            tile's :any:`repick_tile` function once unpicking is complete.
            Defaults to True.
            
        consider_removed_targets : bool
            Boolean value denoting whether to
            place targets removed (due to having ``overwrite_existing=True``)
            back into the candidate_targets list. Defaults to True.

        assign_sky_first : bool
            Flag denoting whether to assign to sky fibres as the first step
            of unpicking (True), or simply make the 'leftover' fibres sky at
            the end of unpicking (False). Defaults to True. Assigning the
            sky fibres semi-randomly first via the :any:`assign_sky` method
            has the advantage of avoiding systematic issues with making
            certain fibres the desginated sky fibres, and attempts to ensure
            an even distribution of sky fibres.
            
        assign_sky_fibres : bool
            Flag denoting whether to actually assign sky fibres to sky
            targets, or simply leave with the special value 'sky'. Defaults
            to False. Note that if this argument is True and the sky_targets
            list is None, an error will be thrown.

        Returns
        -------    
        remaining_targets : list of :class:`TaipanTarget`
            The list of candidate_targets, less those targets
            which have been assigned to this tile.
            Updated copies of standard_targets and guide_targets are NOT
            returned, as repeating these objects in other tiles is not an issue.
            Any science targets that are removed from the tile and not
            re-assigned will also be appended to this list.
            
        removed_targets : empty :obj:`list`
            Deprecated - will now always be the empty list. A
            warning will be printed if this list somehow becomes non-empty.

        Raises
        ------
        ValueError
            Raised if any of the argument limitations are violated. See the
            ``Parameters`` section above, as well as the documentation for
            :any:`assign_fibre` and :any:`assign_tile`.
        """


        TILE_ALLOC_METHODS = [
            'most_difficult',
            'priority',
            'combined_weighted',
            'sequential',
        ]

        # Input checking and type casting
        if method not in TILE_ALLOC_METHODS:
            raise ValueError('Invalid allocation method passed! Must'
                             ' be one of %s' % (str(TILE_ALLOC_METHODS), ))
        combined_weight = float(combined_weight)
        if combined_weight <= 0.:
            raise ValueError('Combined weight must be > 0')
        if sorted(list(sequential_ordering)) != [1,2]:
            raise ValueError('sequential_ordering must be a list or 2-tuple '
                             'containing the 1 and 2 once each')
        if recompute_difficulty and not check_tile_radius:
            raise ValueError('recompute_difficulty requires a full '
                             'target list (i.e. that would require '
                             'check_tile_radius)')
        if sky_targets is None and assign_sky_fibres:
            raise ValueError('You must provide sky targets for assignment')

        removed_targets = []

        # If overwrite_existing, burn all fibre allocations on this tile
        if overwrite_existing:
            for f in self._fibres:
                if isinstance(self._fibres[f], TaipanTarget):
                    removed_targets.append(self._fibres[f])
                self._fibres[f] = None

        # Calculate rest positions for all non-guide fibres
        fibre_usposns = {fibre: self.compute_fibre_usposn(fibre) 
            for fibre in BUGPOS_MM if fibre in FIBRES_NORMAL}
        # More messy but FAST
        fibre_usposns_values = np.asarray(fibre_usposns.values())
        fibre_usposns_keys = np.asarray(fibre_usposns.keys())
        
        candidate_targets_return = candidate_targets[:]

        # Return the removed targets to the master candidates lists
        if consider_removed_targets:
            removed_candidates = [t for t in removed_targets
                if isinstance(t, TaipanTarget) 
                and t.science]
            candidate_targets_return = list(set(
                candidate_targets_return) | set(removed_candidates))
        # Re-blank the removed_targets list
        removed_targets = []

        logging.debug('Considering %d science, %d standard and '
                      '%d guide targets'
                      % (len(candidate_targets),
                         len(standard_targets),
                         len(guide_targets), )
                      )

        # If necessary, strip down the target lists so that they are 
        # restricted to targets on this tile only
        logging.debug('Trimming input lists...')
        candidates_this_tile = candidate_targets_return[:]
        standards_this_tile = standard_targets[:]
        guides_this_tile = guide_targets[:]
        if check_tile_radius:
            candidates_this_tile = targets_in_range(self.ra, self.dec,
                candidates_this_tile, TILE_RADIUS)
            logging.debug('%d science targets remain' %
                          len(candidates_this_tile))
            standards_this_tile = targets_in_range(self.ra, self.dec,
                standards_this_tile, TILE_RADIUS)
            logging.debug('%d standards targets remain' %
                          len(standards_this_tile))
            guides_this_tile = targets_in_range(self.ra, self.dec,
                guides_this_tile, TILE_RADIUS)
            logging.debug('%d guide targets remain' %
                          len(guides_this_tile))

        if len(candidates_this_tile) == 0:
            return candidate_targets, []#, removed_targets

        # Generate the ranking list for the candidate targets
        logging.debug('Computing ranking list...')
        ranking_list = generate_ranking_list(
            candidates_this_tile,
            method=method, combined_weight=combined_weight,
            sequential_ordering=sequential_ordering
        )

        # Re-order the ranking lists for guides and standards, if requested
        if rank_supplements:
            standards_this_tile.sort(key=lambda x: -1 * x.priority)
            guides_this_tile.sort(key=lambda x: -1 * x.priority)

        # First, assign the science targets to the tile
        # This step will only fill TARGET_PER_TILE fibres; that is, it assumes
        # that we will want an optimal number of standards, guides and skies

        if assign_sky_first:
            self.assign_sky()
        
        # For FunnelWeb, some of the targets are also standards. This is fine - 
        # as long as the same target isn't passed twice, the following algorithm
        # will work.
        logging.debug('Assigning targets...')
        assigned_tgts = len([t for t in self._fibres.values()
                             if isinstance(t, TaipanTarget)
                             and t.science])
        extra_standard_targets = 0
        while assigned_tgts < TARGET_PER_TILE + extra_standard_targets and len(
            candidates_this_tile) > 0:
            # Search for the best target according to the criterion
            i = np.argmax(ranking_list)
            tgt = candidates_this_tile[i]
            # Check if this target is forbidden - if so, restart the loop
            # This is more efficient that computing all forbidden targets
            # a priori
            if tgt.is_target_forbidden(self.get_assigned_targets()):
                # Remove tgt from consideration
                ranking_list.pop(i)
                candidates_this_tile.pop(i)
                continue
            # print 'Done!'

            # Identify the closest fibres to this target
            permitted_fibres = fibre_usposns_keys[
                tgt.ranked_index(fibre_usposns_values,
                                 PATROL_RADIUS)].tolist()
            
            # Attempt to make assignment
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
                if self._fibres[permitted_fibres[0]] is None:
                    # Assign the target and 'pop' it from the input list
                    # Target comes from either the input list, or from the
                    # removed targets list we generated earlier
                    tgt = candidates_this_tile.pop(i)
                    _ = ranking_list.pop(i)
                    self._fibres[
                        permitted_fibres[0]] = candidate_targets_return.pop(
                        candidate_targets_return.index(tgt))
                    candidate_found = True
                    assigned_tgts += 1
                    if allow_standard_targets and (
                                extra_standard_targets < STANDARDS_PER_TILE
                    ) and tgt.standard:
                        extra_standard_targets += 1
                    # print 'Done!'
                else:
                    permitted_fibres.pop(0)

            if not(candidate_found):
                # If this point has been reached, the best target cannot be
                # assigned to this tile, so remove it from the
                # candidates_this_tile list
                # print 'Candidate not possible!'
                ranking_list.pop(i)
                candidates_this_tile.pop(i)
                
        #XXX Something was going very wrong with target allocation - issues were in
        #the funnelWeb code in the end, but here are some good lines for debugging.
        if False: #assigned_tgts < 20:         
            cra = np.asarray([t.ra for t in candidate_targets])
            cdec = np.asarray([t.dec for t in candidate_targets])
            cmag = np.array([t.mag for t in candidate_targets])
            ara = np.asarray([t.ra for t in self.get_assigned_targets()])
            adec = np.asarray([t.dec for t in self.get_assigned_targets()])
            amag = np.asarray([t.mag for t in self.get_assigned_targets()])
            plt.clf()
            plt.plot(cra, cdec, 'b.')
            plt.plot(ara, adec, 'gx')
            plt.pause(0.001)
            #import pdb; pdb.set_trace()

        # Assign guides to this tile
        logging.debug('Assigning guides...')
        removed_for_guides = self.assign_guides(guides_this_tile,
                                                check_tile_radius=
                                                not(check_tile_radius), # don't recheck radius
                                                target_method=method,
                                                combined_weight=combined_weight,
                                                sequential_ordering=
                                                sequential_ordering,
                                                rank_guides=False)
        # Put any science targets back in candidate_targets, and any standards
        # back in standards_this_tile
        candidates_this_tile += [t for t in removed_for_guides 
            if isinstance(t, TaipanTarget) and t.science]
        candidate_targets_return += [t for t in removed_for_guides 
            if isinstance(t, TaipanTarget) and t.science]
        standards_this_tile += [t for t in removed_for_guides 
            if isinstance(t, TaipanTarget) and t.standard
            and not t in standards_this_tile]

        # Attempt to assign standards to this tile
        logging.debug('Assigning standards...')
        assigned_standards = len([t for t in self._fibres.values() 
            if isinstance(t, TaipanTarget)
            and t.standard])
        while assigned_standards < STANDARDS_PER_TILE and len(
            standards_this_tile) > 0:
            std = standards_this_tile[0]
            if std.is_target_forbidden(self.get_assigned_targets()):
                standards_this_tile.pop(0)
                continue

            # Identify the closest fibre to this target
            # print 'Finding available fibres...'
            permitted_fibres = fibre_usposns_keys[
                std.ranked_index(fibre_usposns_values, PATROL_RADIUS)].tolist()

            # Attempt to make assignment
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
                if self._fibres[permitted_fibres[0]] is None:
                    # Assign the target and 'pop' it from the input list
                    self._fibres[permitted_fibres[
                        0]] = standards_this_tile.pop(0)
                    candidate_found = True
                    assigned_standards += 1
                    # print 'Done!'
                else:
                    permitted_fibres.pop(0)

            if not(candidate_found):
                # If this point has been reached, the best target cannot be
                # assigned to this tile, so remove it from the
                # candidates_this_tile list
                # print 'Candidate not possible!'
                standards_this_tile.pop(0)

        # Notionally, we should now have full target assignments
        # If so, the rest of the fibres can be assigned to 'sky' and we can
        # exit
        # If not, we need to remove targets and add standards and/or guides
        # until we reach STANDARDS_PER_TILE_MIN and GUIDES_PER_TILE_MIN
        # Constitute lists of guides and standards that are not already assigned
        # to this plate
        assigned_objs = self.get_assigned_targets()
        logging.debug('Currenly have assigned %d/%d/%d (Sci/Std/Gd) targets' %
                      (self.count_assigned_targets_science(),
                       self.count_assigned_targets_standard(),
                       self.count_assigned_targets_guide()))

        if assigned_standards < STANDARDS_PER_TILE_MIN:
            standards_this_tile = [t for t in standard_targets
                if t not in assigned_objs]
            if check_tile_radius:
                standards_this_tile = [t for t in standards_this_tile
                    if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]
                # standards_this_tile = targets_in_range(self.ra, self.dec,
                #   standards_this_tile, TILE_RADIUS)
            if len(standards_this_tile) > 0:
                logging.debug('Having to strip targets for standards...')
                if rank_supplements:
                    standards_this_tile.sort(key=lambda x: -1 * x.priority)
                failure_detected = False
                while (assigned_standards < STANDARDS_PER_TILE_MIN) and (not
                        failure_detected):
                    standards_avail = len(standards_this_tile)
                    standards_this_tile, removed = self.assign_tile(
                        standards_this_tile,
                        check_tile_radius=check_tile_radius,
                        method='priority',
                        overwrite_existing=True)
                    if len(standards_this_tile) != standards_avail:
                        assigned_standards += 1
                    else:
                        failure_detected = True
                        # print 'Failure detected!'
                    if removed is not None and removed != 'sky':
                        if (isinstance(removed, TaipanTarget)
                            and (not removed.guide)
                            and (not removed.standard)
                            and (removed not in candidate_targets)):
                            candidates_this_tile.append(removed)

        # Assign remaining fibres to sky, up to SKY_PER_TILE fibres
        if not assign_sky_first:
            for f in [f for f in self._fibres
                      if self._fibres[f] is None
                      and f not in FIBRES_GUIDE][
                     :SKY_PER_TILE]:
                self._fibres[f] = 'sky'

        if assign_sky_fibres:
            removed_targets += self.assign_sky_fibres(sky_targets,
                                                      target_method='priority',
                                                      check_tile_radius=
                                                      check_tile_radius,
                                                      rank_sky=False)

        # All fibres should now be assigned, unless there
        # are some inaccessible fibres for guides/standards/skies. In this case,
        # we'll call assign_tile on the science targets list to try to 
        # re-populate those fibres before assigning skies
        # Will need to add any science targets in removed_targets back into
        # the candidate_targets list

        candidate_targets_return += [t for t in removed_targets
            if isinstance(t, TaipanTarget) and t.science]
        removed_targets = []

        if len([f for f in self._fibres if self._fibres[f]
             is None and f not in FIBRES_GUIDE]) > SKY_PER_TILE:
            logging.debug('Looking to assign targets to '
                          'remaining empty fibres...')
            # Reconstruct the targets_this_tile list
            candidates_this_tile = candidate_targets_return[:]
            if check_tile_radius:
                candidates_this_tile = targets_in_range(self.ra, self.dec,
                                                        candidates_this_tile,
                                                        TILE_RADIUS)

            failure_detected = False
            while len([f for f in self._fibres 
                       if self._fibres[f] is None]) > SKY_PER_TILE and (
                    not failure_detected):
                candidates_before = candidates_this_tile[:]
                candidates_this_tile, removed_target = self.assign_tile(
                    candidates_this_tile,
                    check_tile_radius=check_tile_radius,
                    method=method, combined_weight=combined_weight,
                    sequential_ordering=sequential_ordering)
                # Overwrite is False, so removed_target will always be None
                if len(candidates_this_tile) == len(candidates_before):
                    failure_detected = True
                    # print 'Failure detected!'

        # Perform a repick if requested
        if repick_after_complete:
            logging.debug('Repicking...')
            self.repick_tile()

        # Update difficulties if requested
        if recompute_difficulty:
            logging.debug('Recomputing difficulty...')
            # Just calculate difficulty for everything on the tile + fibre
            # exclusion radius
            # May calculate if change not strictly required, but no mucking
            # around working out which targets need an update
            assigned_targets_sci = self.get_assigned_targets_science()
            compute_target_difficulties([t for t in candidate_targets_return
                if np.any(np.asarray([t.dist_point((at.ra, at.dec)) 
                    for at in assigned_targets_sci]) 
                < FIBRE_EXCLUSION_DIAMETER)],
                full_target_list=candidate_targets_return)

        logging.info('Made tile with %d science, %d standard '
                     'and %d guide targets' %
                     (len(self.get_assigned_targets_science()),
                      len(self.get_assigned_targets_standard()),
                      len(self.get_assigned_targets_guide()), ))


        return candidate_targets_return, removed_targets

    def worst_fibre(self, fibres_to_check, usposns):
        """Look for the worst fiber (i.e. distance from home position) 
        based on unit sphere positions.
        
        Parameters
        ----------
        fibres_to_check: list
            List of fibres 
            
        usposns: dict
            Dictionary of {fiber index, unit sphere position}
        
        """
        #distance vectors. A numpy way for speed.
        try:
            current_usposns_array = np.asarray([self._fibres[ix].usposn for ix in fibres_to_check])
        except:
            raise ValueError("Indexing the fibre array with un-allocated fibre indices!")
        
        fibre_dists = np.asarray([usposns[f] for f in fibres_to_check]) - current_usposns_array
        
        #Find the maximum magnitude of the vector, and return the fiber index of this.
        return fibres_to_check[np.argmax(np.sum(fibre_dists**2,1))]

    #@profile
    def repick_tile(self):
        """
        Re-assign targets to avoid unnecessary cross-over between bugs.

        Depending on the method used, fibre assignment may result in a
        large number of fibres moving a long distance from their home
        positions, when closer targets are available on the same tile but are
        assigned to another fibre.
        The role of repick_tile is to find such targets, and 
        'swap' target assignments between fibres until an optimum arrangement 
        of fibres is found. This should reduce the number of non-configurable
        tiles generated.

        Guide and 'standard' fibres are repicked separately. Note that sky
        fibres are NOT available to be repicked.
        """

        # Do unpicking separately for guides and science/standards/skies
        for fibres_list in [FIBRES_NORMAL, FIBRES_GUIDE]:
            # Calculate rest positions for all fibres ZZZ remove ZZZ
            #fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
            #    for fibre in fibres_list}
            
            # Calculate rest positions for all fibres
            fibre_usposns = {fibre: self.compute_fibre_usposn(fibre) 
                for fibre in fibres_list if self._fibres != 'sky'}
            # More messy but FAST
            fibre_usposns_values = np.asarray(fibre_usposns.values())
            fibre_usposns_keys = np.asarray(fibre_usposns.keys())
            # print fibre_posns

            # Step through the fibres in reverse order of rest position-target
            # distance, and look for ideal swaps (that is, swaps which reduce
            # the rest position-target distance for both candidates).
            # Identify the fibres with targets assigned
            # Do NOT include the guides in this procedure
            fibres_assigned_targets = [fibre for fibre in fibre_usposns_keys
                if isinstance(self._fibres[fibre], TaipanTarget)]
            # print fibres_assigned_targets

            # Keep iterating until all fibres have been 'popped' from the 
            # fibres_assigned_targets list for having no better options
            while len(fibres_assigned_targets) > 0:
                # print 'Within reassign loop'
                #ZZZ remove ZZZ
                #fibre_dists = {fibre: self._fibres[fibre].dist_point(
                #    fibre_posns[fibre]) 
                #    for fibre in fibres_assigned_targets}
                
                wf = self.worst_fibre(fibres_assigned_targets, fibre_usposns)
                
                # ID the 'worst' remaining assigned fibre
                
                #ZZZ remove ZZZ
                #wfold = max(fibre_dists.items(),
                #    key=operator.itemgetter(1))[0]
                #if wfold != wf:
                #    import pdb; pdb.set_trace()
                
                i = fibres_assigned_targets.index(wf)
                tgt_wf = self._fibres[wf]                
                
                dist_wf = tgt_wf.dist_usposn(fibre_usposns[wf])
                
                #ZZZ remove ZZZ (checked)
                #dist_wfold = tgt_wf.dist_point(fibre_posns[wf])
                
                # ID any other fibres that could potentially take this target
                # Identify the closest fibre to this target
                # print 'Finding available fibres...'
                candidate_fibres = fibre_usposns_keys[tgt_wf.ranked_index(
                    fibre_usposns_values, PATROL_RADIUS)].tolist()
            
                #candidate_fibres = [fibre for fibre in fibre_posns 
                #    if tgt_wf.dist_point(fibre_posns[fibre]) < PATROL_RADIUS] #ZZZ
                    
                # print 'Candidates for shifting: %d' % len(candidate_fibres)
                # ID which of these fibres would
                # be a better match to the 'worst'
                # target than the 'worst' fibre
                # This is a combination of fibres which are:
                # - empty (None), or
                # - Have a currently assigned target which is further from the 
                # fibre home than the 'worst' target, and is closer to the 
                # 'worst fibre'
                candidate_fibres_better = [fibre for fibre in candidate_fibres
                    if (self._fibres[fibre] is None 
                        # or self._fibres[fibre] == 'sky'
                        )
                        and tgt_wf.dist_usposn(fibre_usposns[fibre]) < dist_wf]
                candidate_fibres_better += [fibre for fibre in candidate_fibres
                    if isinstance(self._fibres[fibre], TaipanTarget)
                    and tgt_wf.dist_usposn(fibre_usposns[fibre]) < dist_wf
                    and self._fibres[fibre].dist_usposn(
                        fibre_usposns[wf]) < dist_wf]
                # print candidate_fibres_better
                # print 'Refined candidates: %d' % len(candidate_fibres_better)
                if len(candidate_fibres_better) == 0:
                    # Remove this fibre from further consideration, 
                    # can't be improved
                    fibres_assigned_targets.pop(i)
                else:
                    #ZZZ As candidate_fibres was sorted, candidate_fibres_better is also sorted!
                    #candidate_fibres_better.sort(
                    #    key=lambda x: tgt_wf.dist_point(fibre_posns[x])
                    #)
                    # Do the swap
                    swap_to = candidate_fibres_better[0]
                    self._fibres[wf] = self._fibres[swap_to]
                    self._fibres[swap_to] = tgt_wf
                    # print 'Swapped %d and %d' % (wf, swap_to, )
                    fibres_assigned_targets = [fibre for fibre in fibre_usposns
                        if isinstance(self._fibres[fibre], TaipanTarget)]

    def save_to_file(self, save_path='', return_filename=False):
        """
        Save configuration information for this tile to a simple text config
        file.

        Parameters
        ----------
        save_path : str, path
            The path to save the file to. Can be relative to present
            working directory or absolute. Details to the empty string
            (i.e. file will be saved in present working directory.) Note that,
            if defined, the destination directory must already exist, or an
            IOError will be raised.
        return_filename : Boolean
            Boolean value denoting whether the function should return the
            full path to the written file. Defaults to False.

        Returns
        -------
        filename : str
            Full path to the file written. Only returned if
            return_filename=True.
        """

        # Generate a unique filename to save this file as
        unique_name = 'TaipanTile_R%3.1f_D%3.1f_P%4f_D%4d_%s.csv' % (
            self.ra, self.dec, self.priority, self.difficulty,
            random.sample(string.letters, 3) # random element
        )

        # Generate the grid of values to write out
        values = []
        values.append(['Target', 'RA', 'Dec', 'Type'])
        for fibre in self.fibres:
            if isinstance(self.fibres[fibre], TaipanTarget):
                values.append([fibre, self.fibres[fibre].ra,
                               self.fibres[fibre].dec,
                               self.fibres[fibre].return_target_code()])

        # Open the file and write out
        with open(save_path + os.sep + unique_name, 'wb') as fileobj:
            csvwriter = csv.writer(delimiter=' ')
            csvwriter.writerows(values)

        if return_filename:
            filename = os.path.abspath(save_path + os.sep + unique_name)
            return filename

        return
