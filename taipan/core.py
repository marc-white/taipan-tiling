#!python

# Global definitions for TAIPAN tiling code

# Created: Marc White, 2015-09-14
# Last modified: Marc White, 2016-01-19

import numpy as np
import os
import math
import random
import string
import operator
import logging
from matplotlib.cbook import flatten
from scipy.spatial import KDTree, cKDTree
from sklearn.neighbors import KDTree as skKDTree

# -------
# CONSTANTS
# -------

# Computational break-even points
# Number of targets needed to make it worth doing difficulty calculations
# with a KDTREE
BREAKEVEN_KDTREE = 50


# Instrument variables
TARGET_PER_TILE = 120
STANDARDS_PER_TILE = 10
STANDARDS_PER_TILE_MIN = 5
SKY_PER_TILE = 20
SKY_PER_TILE_MIN = 20
GUIDES_PER_TILE = 9
GUIDES_PER_TILE_MIN = 3
FIBRES_PER_TILE = (TARGET_PER_TILE + STANDARDS_PER_TILE 
    + SKY_PER_TILE + GUIDES_PER_TILE)
INSTALLED_FIBRES = 159
if FIBRES_PER_TILE < INSTALLED_FIBRES:
    raise Exception('WARNING: Not all fibres will be utilised. '
                    'Check the fibre constants in taipan/core.py.')
if FIBRES_PER_TILE > INSTALLED_FIBRES:
    raise Exception('You are attempting to assign more fibres than'
                    'are currently installed. Check the fibre '
                    'constants in taipan/core.py.')

# Fibre positioning
# Comment out lines to render that fibre inoperable
BUGPOS_MM = {
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
if len(BUGPOS_MM) != INSTALLED_FIBRES:
    raise Exception('The number of fibre positions defined does'
                    ' not match the set value for INSTALLED_FIBRES. Please '
                    'check the fibre configuration variables in taipan.py.')
ARCSEC_PER_MM = 67.2
# Convert BUGPOS_MM to arcsec
BUGPOS_ARCSEC = {key: (value[0]*ARCSEC_PER_MM, value[1]*ARCSEC_PER_MM)
                 for key, value in BUGPOS_MM.iteritems()}
# Convert these lateral shifts into a distance & PA from the
# tile centre
BUGPOS_OFFSET = {key : ( math.sqrt(value[0]**2 + value[1]**2), 
                         math.degrees(math.atan2(value[0], value[1])) 
                            % 360., )
                 for key, value in BUGPOS_ARCSEC.iteritems()}

# Define which fibres are the guide bundles
FIBRES_GUIDE = [
    40,
    57,
    74,
    90,
    55,
    72,
    88,
    70,
    86,
]
FIBRES_GUIDE.sort()
# Uncomment the following code to force GUIDES_PER_TILE 
# to be as high as possible
# if len(GUIDE_FIBRES) != GUIDES_PER_TILE:
#   raise Exception('Length of GUIDE_FIBRES array does not match'
#       'GUIDES_PER_TILE. Check the constant in taipan.core')
FIBRES_NORMAL = [f for f in BUGPOS_OFFSET if f not in FIBRES_GUIDE]

FIBRE_EXCLUSION_RADIUS = 10.0 * 60.0 # arcsec
TILE_RADIUS = 3.0 * 60.0 * 60.0      # arcsec
TILE_DIAMETER = 2.0 * TILE_RADIUS    # arcsec
PATROL_RADIUS = 1.2 * 3600.          # arcsec

TARGET_PRIORITY_MIN = 0
TARGET_PRIORITY_MAX = 10


# ------
# GLOBAL UTILITY FUNCTIONS
# ------

def prod(iterable):
    """
    Compute product of all elements of an iterable
    """
    return reduce(operator.mul, iterable, 1)


def aitoff_plottable((ra, dec), ra_offset=0.0):
    """
    Convert coordinates to those compatible with matplotlib
    aitoff projection 
    
    Parameters
    ----------
    (ra,dec): float
        Right Ascension and Declination in degrees.
    """
    ra = (ra + ra_offset) % 360. - 180.
    return math.radians(ra), math.radians(dec)


def dist_points(ra, dec, ra1, dec1):
    ra = math.radians(ra)
    dec = math.radians(dec)
    ra1 = math.radians(ra1)
    dec1 = math.radians(dec1)
    dist = 2*math.asin(math.sqrt((math.sin((dec1-dec)/2.))**2
        +math.cos(dec1)*math.cos(dec)*(math.sin((ra1-ra)/2.))**2))
    return math.degrees(dist) * 3600.


def dist_points_approx(ra, dec, ra1, dec1):
    decfac = np.cos(dec * np.pi / 180.)
    dra = ra - ra1
    if np.abs(dra) > 180.:
        dra -= np.sign(dra) * 360.
    dist = np.sqrt((dra / decfac)**2 + (dec - dec1)**2)
    return dist * 3600.


def dist_points_mixed(ra, dec, ra1, dec1, dec_cut=30.0):
    """
    Compute an approx position if abs(dec) < dec_cut, otherwise do full calc
    """
    dec_cut = abs(dec_cut)
    if abs(dec) <= dec_cut and abs(dec1) <= dec_cut:
        dist = dist_points_approx(ra, dec, ra1, dec1)
    else:
        dist = dist_points(ra, dec, ra1, dec1)
    return dist


def dist_euclidean(dist_ang):
    """Compute a straight-line distance from an angular one"""
    return 2. * np.sin(np.radians(dist_ang/2.))


def polar2cart((ra, dec)):
    """
    Convert RA, Dec to x, y, z
    """
    x = np.sin(np.radians(dec+90.)) * np.cos(np.radians(ra))
    y = np.sin(np.radians(dec+90.)) * np.sin(np.radians(ra))
    z = np.cos(np.radians(dec+90.))
    return x, y, z


def compute_offset_posn(ra, dec, dist, pa):
    """
    Compute a new position based on a given position, a distance
    from that position, and a position angle from that position.
    Based on http://williams.best.vwh.net/avform.htm .

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
    ra_new, dec_new : 
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
        method='priority', combined_weight=1.0, sequential_ordering=(1,2)):
    """
    Generate a ranking list for target assignment.

    Parameters
    ----------
    candidate_targets : :class:`TaipanTarget` list
        The list of TaipanTargets to rank.
    method : string
        The ranking method. These methods are as for 
        :meth:`TaipanTile.assign_tile` and :meth:`TaipanTile.unpick_tile` -- see the 
        documentation for those functions for details. Defaults to 'priority'.
    combined_weight, sequential_ordering : 
        Extra parameters for the
        'combined_weighted' and 'sequential' ranking options. See docs for
        assign_tile/unpick_tile for details. Defaults to 1.0 and (1,2)
        respectively.

    Returns
    -------
    ranking_list: 
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

    Compute the target difficulties for a given list of TaipanTargets. The
    'difficulty' is simply the number of targets within FIBRE_EXCLUSION_RADIUS
    of the target of interest. Note that the target itself will be included
    in this number; therefore, the minimum value of difficulty will be 1.

    Parameters
    ----------
    target_list :
        The list of TaipanTargets to compute the difficulty for.
    full_target_list :
        The full list of targets to use in the difficulty computation. This is
        useful for situations where targets need to be considered in a 
        re-computation of difficulty, but the difficulty of those targets need
        not be updated. This occurs, e.g. when target difficulties must be
        updated after some targets have been assigned to a tile. Only targets
        within TILE_RADIUS + FIBRE_EXCLUSION_RADIUS of the tile centre need
        updating; however, targets within 
        TILE_RADIUS + 2*FIBRE_EXCLUSION_RADIUS need to be considered in the 
        calculation.
        If given, target_list MUST be a sublist of full_target_list. If not,
        an ValueError will be thrown.
        Defaults to None, in which case, target difficulties are computed for
        all targets in target_lists against target_list itself.
    verbose:
        Whether to print detailed debug information to the root logger. Defaults
        to False.
    leafsize:
        The leafsize (i.e. number of targets) where it becomes more efficient
        to construct a KDTree rather than brute-force the distances between
        targets. Defaults to the module default (i.e. BREAKEVEN_KDTREE).

    Returns
    ------- 
        Nil. TaipanTargets updated in-place.
    """

    tree_function = cKDTree

    if len(target_list) == 0:
        return

    if full_target_list:
        if verbose:
            'Checking target_list against full_target_list...'
        if not np.all(np.in1d(target_list, full_target_list)):
            raise ValueError('target_list must be a sublist'
                             ' of full_target_list')

    if verbose:
        logging.debug('Forming Cartesian positions...')
    # Calculate UC positions if they haven't been done already
    burn = [t.compute_ucposn() for t in target_list if t.ucposn is None]
    cart_targets = np.asarray([t.ucposn for t in target_list])
    if full_target_list:
        burn = [t.compute_ucposn() for t in full_target_list 
            if t.ucposn is None]
        full_cart_targets = np.asarray([t.ucposn for t in full_target_list])
    else:
        full_cart_targets = np.copy(cart_targets)
    
    if verbose:
        logging.debug('Generating KDTree with leafsize %d' % leafsize)
    if tree_function == skKDTree:
        tree = tree_function(full_cart_targets, leaf_size=leafsize)
    else:
        tree = tree_function(full_cart_targets, leafsize=leafsize)

    dist_check = dist_euclidean(FIBRE_EXCLUSION_RADIUS/3600.)

    if tree_function == skKDTree:
        difficulties = tree.query_radius(cart_targets,
            dist_euclidean(FIBRE_EXCLUSION_RADIUS/3600.))
    else:
        if len(target_list) < (100*leafsize):
            if verbose:
                logging.debug('Computing difficulties...')
            difficulties = tree.query_ball_point(cart_targets,
                dist_euclidean(FIBRE_EXCLUSION_RADIUS/3600.))
        else:
            if verbose:
                logging.debug('Generating subtree for difficulties...')
            subtree = tree_function(cart_targets, leafsize=leafsize)
            difficulties = subtree.query_ball_tree(tree,
                dist_euclidean(FIBRE_EXCLUSION_RADIUS/3600.))
    difficulties = [len(d) for d in difficulties]

    if verbose:
        logging.debug('Assigning difficulties...')
    for i in range(len(difficulties)):
        target_list[i].difficulty = difficulties[i]
    if verbose:
        logging.debug('Difficulties done!')
        
    difficulties = [1]
    if min(difficulties) ==0:
        raise UserWarning

    return


def targets_in_range(ra, dec, target_list, dist,
                     leafsize=BREAKEVEN_KDTREE):
    """
    Return the subset of target_list within dist of (ra, dec).

    Computes the subset of targets within range of the given (ra, dec)
    coordinates.

    Parameters
    ----------
    ra, dec :
        The RA and Dec of the position to be investigated, in decimal
        degrees.
    target_list :
        The list of TaipanTarget objects to consider.
    dist :
        The distance to test against, in *arcseconds*. 

    Returns
    -------
    targets_in_range :
        The list of input targets which are within dist of
        (ra, dec).
    """

    if len(target_list) == 0:
        return []

    # Decide whether to brute-force or construct a KDTree
    if len(target_list) <= BREAKEVEN_KDTREE:
        targets_in_range = [t for t in target_list
            if t.dist_point((ra, dec)) < dist]
    else:
        # Do KDTree computation
        # logging.debug('Generating KDTree with leafsize %d' % leafsize)
        cart_targets = np.asarray([t.ucposn for t in target_list])
        tree = cKDTree(cart_targets, leafsize=leafsize)
        inds = tree.query_ball_point(polar2cart((ra, dec)),
                                     dist_euclidean(dist / 3600.))
        targets_in_range = [target_list[i] for i in inds]

    return targets_in_range



# ------
# TILING OBJECTS
# ------

class TaipanTarget(object):
    """
    Holds information and convenience functions for a TAIPAN
    observing target.
    """

    # Initialisation & input-checking
    def __init__(self, idn, ra, dec, ucposn=None, priority=1, standard=False,
                 guide=False, difficulty=0, mag=None,
                 h0=False, vpec=False, lowz=False):
        self._idn = None
        self._ra = None
        self._dec = None
        self._ucposn = None
        self._priority = None
        self._standard = None
        self._guide = None
        self._difficulty = None
        self._mag = None
        # Taipan-specific fields
        self._h0 = None
        self._vpec = None
        self._lowz = None

        # Insert given values
        # This causes the setter functions to be called, which does
        # error checking
        self.idn = idn
        self.ra = ra
        self.dec = dec
        self.ucposn = ucposn
        self.priority = priority
        self.standard = standard
        self.guide = guide
        self.difficulty = difficulty
        self.mag = mag
        self.h0 = h0
        self.vpec = vpec
        self.lowz = lowz

    def __repr__(self):
        return 'TP TGT %s' % str(self._idn)

    def __str__(self):
        return 'TP TGT %s' % str(self._idn)

    # Uncomment to have target equality decided on ID
    # WARNING - make sure your IDs are unique!
    # def __eq__(self, other):
    #   if isinstance(other, self.__class__):
    #       return (self.idn == other.idn) and (self.standard 
    #           == other.standard) and (self.guide == other.guide)
    #   return False

    # def __ne__(self, other):
    #   if isinstance(other, self.__class__):
    #       return not((self.idn == other.idn) and (self.standard 
    #           == other.standard) and (self.guide == other.guide))
    #   return True

    # def __cmp__(self, other):
    #   if isinstance(other, self.__class__):
    #       if (self.idn == other.idn) and (self.standard 
    #           == other.standard) and (self.guide == other.guide):
    #           return 0
    #   return 1

    @property
    def idn(self):
        """TAIPAN target ID"""
        return self._idn

    @idn.setter
    def idn(self, d):
        if not d: raise Exception('ID may not be empty')
        self._idn = d

    @property
    def ra(self):
        """Target RA"""
        return self._ra

    @ra.setter
    def ra(self, r):
        if r is None: raise Exception('RA may not be blank')
        if r < 0.0 or r >= 360.0: 
            raise Exception('RA outside valid range')
        self._ra = r

    @property
    def dec(self):
        """Target dec"""
        return self._dec

    @dec.setter
    def dec(self, d):
        if d is None: raise Exception('Dec may not be blank')
        if d < -90.0 or d > 90.0:
            raise Exception('Dec outside valid range')
        self._dec = d

    @property
    def ucposn(self):
        """Target position on the unit sphere, should be 3-list or 3-tuple"""
        return self._ucposn

    @ucposn.setter
    def ucposn(self, value):
        if value is None:
            self._ucposn = None
            return
        if len(value) != 3:
            raise Exception('ucposn must be a 3-list or 3-tuple')
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
            raise Exception('ucposn must lie on unit sphere '
                            '(x^2 + y^2 + z^2 = 1.0 - error of %f '
                            '(%f, %f, %f) )'
                            % (value[0]**2 + value[1]**2 + value[2]**2,
                               value[0], value[1], value[2]))
        self._ucposn = list(value)
    

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, p):
        # Make sure priority is an int
        p = int(p)
        if p < TARGET_PRIORITY_MIN or p > TARGET_PRIORITY_MAX:
            raise ValueError('Target priority must be %d < p < %d' 
                % (TARGET_PRIORITY_MIN, TARGET_PRIORITY_MAX, ))
        self._priority = p

    @property
    def standard(self):
        """Is this target a standard"""
        return self._standard

    @standard.setter
    def standard(self, b):
        b = bool(b)
        self._standard = b

    @property
    def guide(self):
        """Is this target a guide"""
        return self._guide

    @guide.setter
    def guide(self, b):
        b = bool(b)
        self._guide = b

    @property
    def difficulty(self):
        """Difficulty, i.e. number of targets within FIBRE_EXCLUSION_RADIUS"""
        return self._difficulty

    @difficulty.setter
    def difficulty(self, d):
        d = int(d)
        if d < 0:
            raise ValueError('Difficulty must be >= 0')
        self._difficulty = d

    @property
    def mag(self):
        """Target Magnitude"""
        return self._mag

    @mag.setter
    def mag(self, m):
        if m:
            assert (m > -10 and m < 30), "mag outside valid range"
        self._mag = m

    @property
    def h0(self):
        """Is this a h0 target?"""
        return self._h0

    @h0.setter
    def h0(self, b):
        b = bool(b)
        self._h0 = b

    @property
    def lowz(self):
        """Is this a lowz target?"""
        return self._lowz

    @lowz.setter
    def lowz(self, b):
        b = bool(b)
        self._lowz = b

    @property
    def vpec(self):
        """Is this a vpec (peculiar velocity) target?"""
        return self._vpec

    @vpec.setter
    def vpec(self, b):
        b = bool(b)
        self._vpec = b

    def return_target_code(self):
        """
        Return a single-character string based on the type of TaipanTarget passed
        as a function argument.

        Parameters
        ----------
        Nil.

        Returns
        -------
        code:
            A single-character string, denoting the type of TaipanTarget passed in.
            Codes are currently:
            X - science
            G - guide
            S - standard
        """

        if self.standard:
            code = 'S'
        elif self.guide:
            code = 'G'
        else:
            code = 'X'

        return code


    def compute_ucposn(self):
        """
        Compute the position of this target on the unit circle from its
        RA and Dec values.

        Parameters
        ----------
        Nil.

        Returns
        -------
        Nil. TaipanTarget updated in-situ.
        """
        if self.ra is None or self.dec is None:
            raise Exception('Cannot compute ucposn because'
                ' RA and/or Dec is None!')
        self.ucposn = polar2cart((self.ra, self.dec))
        return


    def dist_point(self, (ra, dec)):
        """
        Compute the distance between this target and a given position

        Parameters
        ----------
        ra, dec :
            The sky position to test. Should be in decimal degrees.

        Returns
        -------
        dist :
            The angular distance between the two points in arcsec.
        """

        # Arithmetic implementation - fast
        # Convert all to radians
        ra = math.radians(ra)
        dec = math.radians(dec)
        ra1 = math.radians(self.ra)
        dec1 = math.radians(self.dec)
        dist = 2*math.asin(math.sqrt((math.sin((dec1-dec)/2.))**2
            +math.cos(dec1)*math.cos(dec)*(math.sin((ra1-ra)/2.))**2))
        return math.degrees(dist) * 3600.


    def dist_point_approx(self, (ra, dec)):
        decfac = np.cos(dec * np.pi / 180.)
        dra = ra - self.ra
        if np.abs(dra) > 180.:
            dra = dra - np.sign(dra) * 360.
        dist = np.sqrt((dra / decfac)**2 + (dec - self.dec)**2)
        return dist * 3600.


    def dist_point_mixed(self, (ra, dec), dec_cut=30.):
        dec_cut = abs(dec_cut)
        if abs(dec) <= dec_cut and abs(self.dec) <= dec_cut:
            dist = self.dist_point_approx((ra, dec))
        else:
            dist = self.dist_point((ra, dec))
        return dist


    def dist_target(self, tgt):
        """
        Compute the distance between this target and another target.

        Parameters
        ----------
        tgt :
            The target to check against

        Returns
        -------
        dist :
            The angular distance between the two points in arcsec.
        """

        return self.dist_point((tgt.ra, tgt.dec))

    
    def dist_target_approx(self, tgt):
        """
        Compute the APPROX distance between this target and another target.

        Parameters
        ----------
        tgt : 
        The target to check against

        Returns
        -------    
        dist : 
        The angular distance between the two points in arcsec.
        """

        return self.dist_point_approx((tgt.ra, tgt.dec))

    def dist_target_mixed(self, tgt, dec_cut=30.):
        """
        Compute the mixed distance between this target and another target.

        Parameters
        ----------    
        tgt : 
            The target to check against

        Returns
        -------    
        dist : 
            The angular distance between the two points in arcsec.
        """

        return self.dist_point_mixed((tgt.ra, tgt.dec), dec_cut=dec_cut)

    def excluded_targets(self, tgts):
        """
        Given a list of other TaipanTargets, return a list of those 
        targets that are too close to this target to be on the same
        tile. Note that, if the calling target is in the target list,
        it will appear in the returned list of forbidden targets.

        Parameters
        ----------    
        tgts : 
            The list of TaipanTargets to test against

        Returns
        -------    
        excluded_tgts : 
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = targets_in_range(self.ra, self.dec, tgts,
                                         FIBRE_EXCLUSION_RADIUS)
        return excluded_tgts

    def excluded_targets_approx(self, tgts):
        """
        As for excluded_targets, but using the approximate distance calculation.
        This will *under*estimate difficulty (by overestimating distance),
        especially near the poles.

        Parameters
        ----------    
        tgts : 
            The list of TaipanTargets to test against

        Returns
        -------    
        excluded_tgts : 
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = [t for t in tgts 
            if self.dist_target_approx(t) < FIBRE_EXCLUSION_RADIUS]

        return excluded_tgts

    def excluded_targets_mixed(self, tgts, dec_cut=30.):
        """
        As for excluded_targets, but using the mixed distance calculation.

        Parameters
        ----------    
        tgts : 
            The list of TaipanTargets to test against
        dec_cut : 
            (Absolute) declination value to use for mixed method.

        Returns
        -------    
        excluded_tgts : 
            The subset of tgts that cannot be on the same
                       tiling as the calling target.
        """
        excluded_tgts = [t for t in tgts 
            if self.dist_target_mixed(t, dec_cut=dec_cut) 
            < FIBRE_EXCLUSION_RADIUS]

        return excluded_tgts

    def compute_difficulty(self, tgts):
        """Calculate & set the difficulty of this target.

        Difficulty is defined as the number of targets within a 
        FIBRE_EXCLUSION_RADIUS of the calling target. This means that this
        function will need to be invoked every time the comparison list changes.

        Parameters
        ----------     
        tgts :
            List of targets to compare to.
        approx : bool
            Boolean value, denoting whether to calculate distances using
            the approximate method. Defaults to False.
        mixed : bool
            Boolean value, denoting whether to calculate distances using
            the mixed method (approx if dec < dec_cut, full otherwise). Defaults
            to False.
        dec_cut :
            (Absolute) declination value to use for mixed method.

        Returns
        -------     
            Nil. Target's difficulty parameter is updated in place. Note
            that if the calling target is also within tgts, it will count
            towards the computed difficulty.
        """
        self.difficulty = len(self.excluded_targets(tgts))
        return

    def compute_difficulty_approx(self, tgts):
        """
        As for compute_difficulty, but use the approx distance calculation.
        """
        self.difficulty = len(self.excluded_targets_approx(tgts))
        return

    def compute_difficulty_mixed(self, tgts, dec_cut=30.):
        """
        As for compute_difficulty, but use the mixed distance calculation.
        """
        self.difficulty = len(self.excluded_targets_mixed(tgts,
            dec_cut=30.))
        return

    def is_target_forbidden(self, tgts):
        """
        Test against a list of other targets to see if this target is forbidden.

        Parameters
        ----------    
        tgts :
            The list of targets to test against.

        Returns
        -------    
        forbidden :
            Boolean value.
        """
        if len(tgts) == 0:
            return False

        if len(targets_in_range(self.ra, self.dec, tgts, 
            FIBRE_EXCLUSION_RADIUS)) > 0:
            return True
        return False


class TaipanTile(object):
    """
    Holds information and convenience functions for a TAIPAN tiling
    solution
    """

    def __init__(self, ra, dec, field_id=None, ucposn=None,
                 pa=0.0, mag_min=None, mag_max=None):
        self._fibres = {}
        for i in range(1, FIBRES_PER_TILE+1):
            self._fibres[i] = None
        # self._fibres = self.fibres(fibre_init)
        self._ra = None
        self._dec = None
        self._ucposn = None
        self._field_id = None
        self._mag_min = None
        self._mag_max = None
        self._pa = 0.0

        # Insert the passed values
        # Doing it like this forces the setter functions to be
        # called, which provides error checking
        self.ra = ra
        self.dec = dec
        self.ucposn = ucposn
        self.field_id = field_id
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
        """Assignment of fibres"""
        return self._fibres
    @fibres.setter
    def fibres(self, d):
        if (not isinstance(d, dict) or [i for
                                        i in d].sort() != [i for i in
                                                           BUGPOS_MM].sort()):
            raise Exception('Tile fibres must be a dictionary'
                            ' with keys %s' % (
                                str(sorted([i for i in BUGPOS_MM])), )
                            )
        self._fibres = d

    @property
    def ra(self):
        """Target RA"""
        return self._ra

    @ra.setter
    def ra(self, r):
        r = float(r)
        if r < 0.0 or r >= 360.0: 
            raise Exception('RA outside valid range')
        self._ra = r

    @property
    def dec(self):
        """Target dec"""
        return self._dec

    @dec.setter
    def dec(self, d):
        d = float(d)
        if d < -90.0 or d > 90.0:
            raise Exception('Dec outside valid range')
        self._dec = d

    @property
    def ucposn(self):
        """Target position on the unit sphere, should be 3-list or 3-tuple"""
        return self._ucposn

    @ucposn.setter
    def ucposn(self, value):
        if value is None:
            self._ucposn = None
            return
        if len(value) != 3:
            raise Exception('ucposn must be a 3-list or 3-tuple')
        if value[0] < -1. or value[0] > 1.:
            raise Exception('x value %f outside allowed bounds (-1 <= x <= 1)'
                            % value[0])
        if value[1] < -1. or value[1] > 1.:
            raise Exception('y value %f outside allowed bounds (-1 <= x <= 1)'
                            % value[1])
        if value[2] < -1. or value[2] > 1.:
            raise Exception('z value %f outside allowed bounds (-1 <= x <= 1)'
                            % value[2])
        if abs(value[0] ** 2 + value[1] ** 2 + value[2] ** 2 - 1.0) > 0.001:
            raise Exception('ucposn must lie on unit sphere '
                            '(x^2 + y^2 + z^2 = 1.0 - error of %f'
                            ' (%f, %f, %f) )'
                            % (value[0] ** 2 + value[1] ** 2 + value[2] ** 2,
                               value[0], value[1], value[2]))
        self._ucposn = list(value)

    @property
    def pa(self):
        """Tile position angle (PA)"""
        return self._pa
    @pa.setter
    def pa(self, p):
        p = float(p)
        if p < 0.0 or p >= 360.0:
            raise ValueError('PA must be 0 <= pa < 360')
        self._pa = p

    @property
    def field_id(self):
        """Tile field ID"""
        return self._field_id
    @field_id.setter
    def field_id(self, p):
        p = int(p)
        self._field_id = p

    @property
    def mag_max(self):
        """Maximum Target Magnitude"""
        return self._mag_max
    @mag_max.setter
    def mag_max(self, m):
        if m:
            assert (m > -10 and m < 30), "mag_max outside valid range"
        self._mag_max = m

    @property
    def mag_min(self):
        """Minimum Target Magnitude"""
        return self._mag_min
    @mag_min.setter
    def mag_min(self, m):
        if m:
            assert (m > -10 and m < 30), "mag_min outside valid range"
        self._mag_min = m

        
    def priority(self):
        """
        Calculate the priority ranking of this tile. Do this by summing
        up the priorities of the TaipanTargets within the tile.
        """
        priority = sum([t.priority for t in self.fibres 
                        if isinstance(t, TaipanTarget)])
        return priority


    def difficulty(self):
        """
        Calculate the difficulty ranking of this tile. Do this by summing
        up the difficulties of the TaipanTargets within the tile.
        """
        difficulty = sum([t.difficulty for t in self.fibres 
                          if isinstance(t, TaipanTarget)])
        return difficulty


    def remove_duplicates(self, assigned_targets):
        """
        Un-assign any targets appearing in assigned_objects that are
        attached to a fibre in this tile.

        Parameters
        ----------    
        assigned_targets :
            A list or set of already-assigned
            TaipanTargets.

        Returns
        -------    
        removed_targets :
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
        fibre :
            The fibre to un-assign.

        Returns
        -------    
        removed_target :
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
        fibre :
            The fibre to compute the position of.

        Returns
        -------    
        ra, dec : float
            The position of the fibre on the sky.
        """
        fibre = int(fibre)
        if fibre not in BUGPOS_MM:
            raise ValueError('Fibre does not exist in BUGPOS listing')

        fibre_offset = BUGPOS_OFFSET[fibre]
        pos = compute_offset_posn(self.ra, self.dec,
            fibre_offset[0], # Fibre distance from tile centre
            (fibre_offset[1] + self.pa) % 360.) # Account for tile PA
        return pos


    def get_assigned_targets(self, return_dict=False):
        """
        Return a list of all TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets :
            The list of TaipanTargets currently assigned to
            this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.iteritems()
            if isinstance(t, TaipanTarget)}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()


    def get_assigned_targets_science(self, return_dict=False):
        """
        Return a list of science TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict :
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets :
            The list of science TaipanTargets currently assigned
            to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.iteritems()
            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.iteritems()
            if not t.guide and not t.standard}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()


    def count_assigned_targets_science(self):
        """
        Count the number of science targets assigned to this tile.

        Parameters
        ----------    

        Returns
        -------    
        no_assigned_targets :
            The number of science targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_science())
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
        assigned_targets :
            The list of standard TaipanTargets currently 
            assigned to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.iteritems()
            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.iteritems()
            if t.standard}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()


    def count_assigned_targets_standard(self):
        """
        Count the number of standard targets assigned to this tile.

        Parameters
        ----------     

        Returns
        -------    
        no_assigned_targets :
            The number of standard targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_standard())
        return no_assigned_targets


    def get_assigned_targets_guide(self, return_dict=False):
        """
        Return a list of guide TaipanTargets currently assigned to this tile.

        Parameters
        ----------    
        return_dict : bool
            Boolean value denoting whether to return the result as
            a dictionary with keys corresponding to fibre number (True), or
            a simple list of targets (False). Defaults to False.

        Returns
        -------    
        assigned_targets :
            The list of guide TaipanTargets currently 
            assigned to this tile.
        """
        assigned_targets = {f: t for (f, t) in self._fibres.iteritems()
            if isinstance(t, TaipanTarget)}
        assigned_targets = {f: t for (f, t) in assigned_targets.iteritems()
            if t.guide}
        if return_dict:
            return assigned_targets
        return assigned_targets.values()


    def count_assigned_targets_guide(self):
        """
        Count the number of guide targets assigned to this tile.

        Parameters
        ----------     

        Returns
        -------    
        no_assigned_targets :
            The number of guide targets assigned to this
            tile.
        """
        no_assigned_targets = len(self.get_assigned_targets_guide())
        return no_assigned_targets


    def count_assigned_fibres(self):
        """
        Count the number of assigned fibres on this tile.

        Parameters
        ----------     

        Returns
        -------    
        assigned_fibres :
            The integer number of empty fibres.
        """
        assigned_fibres = len([f for f in self._fibres.values()
            if f is not None])
        return assigned_fibres


    def count_empty_fibres(self):
        """
        Count the number of empty fibres on this tile.

        Parameters
        ----------     

        Returns
        -------    
        empty_fibres :
            The integer number of empty fibres.
        """
        empty_fibres = len([f for f in self._fibres if f is None])
        return empty_fibres


    def get_assigned_fibres(self):
        """
        Get the fibre numbers that have been assigned a target/sky.

        Parameters
        ----------     

        Returns
        -------    
        assigned_fibres_list :
            The list of fibre identifiers (integers) which
            have a target/sky assigned.
        """
        assigned_fibres_list = [f for f in self._fibres if f is not None]
        return assigned_fibres_list


    def excluded_targets(self, tgts):
        """
        Calculate which targets are excluded by targets already assigned.

        Compute which targets in the input list are excluded from this tile,
        because they would violate the FIBRE_EXCLUSION_RADIUS of targets
        already assigned.

        Parameters
        ----------    
        tgts :
            List of targets to check against.

        Returns
        -------    
        excluded_targets :
            A subset of tgts, composed of targets which may
            not be assigned to this tile.
        """
        excluded_tgts = list(set([t.excluded_targets(tgts) 
            for t in self.get_assigned_targets()]))
        return excluded_tgts

    def available_targets(self, tgts):
        """
        Calculate which targets are within this tile.

        This function does not perform any exclusion checking.

        Parameters
        ----------    
        tgts :
            List of targets to check against.

        Returns
        -------    
        available_targets :
            A subset of tgts, composed of targets which are
            within the radius of this tile.
        """
        # available_targets = [t for t in tgts
        #   if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]
        available_targets = targets_in_range(self.ra, self.dec, tgts,
            TILE_RADIUS)
        return available_targets

    def calculate_tile_score(self, method='completeness',
                             combined_weight=1.0, disqualify_below_min=True):
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
        
        'priority-prod' -- The priority product of the assigned targets.
        
        'combined-weighted-sum' -- The sum of a combined weight of difficulty
        and priority, as given by the combined_weight variable.
        
        'combined-weighted-prod' -- The product of a combined weight of
        difficulty and priority.

        Parameters
        ----------    
        method :
            String denoting the ranking method to be used. See above for
            details. Defaults to 'completeness'.
            
        combined_weight :
            Optional float denoting the combined weighting
            to be use for combined-weighted-sum/prod. Defaults to 1.0.
            
        disqualify_below_min :
            Optional Boolean value denoting whether to
            rank tiles with a nubmer of guides below GUIDES_PER_TILE_MIN or
            a number of standards below STANDARDS_PER_TILE_MIN a score of
            0. Defaults to True.

        Returns
        -------    
        ranking_score :
            The ranking score of this tile. Will always return
            a float, even if the ranking could be expressed as an integer. A
            higher score denotes a better-ranked tile.
        """
        SCORE_METHODS = [
            'completeness',
            'difficulty-sum',
            'difficulty-prod',
            'priority-sum',
            'priority-prod',
            'combined-weighted-sum',
            'combined-weighted-prod',
        ]
        if method not in SCORE_METHODS:
            raise ValueError('Scoring method must be one of %s' 
                % str(SCORE_METHODS))

        # Bail out now if tile doesn't meet guide or standards requirements
        if disqualify_below_min and (self.count_assigned_targets_guide() 
            < GUIDES_PER_TILE_MIN or self.count_assigned_targets_standard()
            < STANDARDS_PER_TILE_MIN):
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
        elif method == 'priority-prod':
            ranking_score = prod([t.priority for t in targets_sci])
        elif 'combined-weighted' in method:
            max_difficulty = float(max([t.difficulty for t in targets_sci]))
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
        fibre :
            The fibre to assign to.
        tgt :
            The TaipanTarget, or None, to assign to this fibre.

        Returns
        -------     
        """
        fibre = int(fibre)
        if fibre not in BUGPOS_OFFSET:
            raise ValueError('Invalid fibre')
        if not(tgt is None or isinstance(tgt, TaipanTarget)):
            raise ValueError('tgt must be a TaipanTarget or None')
        self._fibres[fibre] = tgt
        return

    def assign_fibre(self, fibre, candidate_targets, 
                     check_patrol_radius=True, check_tile_radius=True,
                     recompute_difficulty=True,
                     order_closest_secondary=True,
                     method='combined_weighted',
                     combined_weight=1.0,
                     sequential_ordering=(0,1,2)):
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
        (FIBRE_EXCLUSION_RADIUS).
        
        *priority* - Assign the highest-priority target within the patrol
        radius.
        
        *combined_weighted* - Prioritise targets within the patrol radius
        based on a weighted combination of most_diffucult and
        priority. Uses the combined_weight keyword (see below)
        as the weighting.
        
        *sequential* - Prioritise targets for this fibre based on closest,
        most_difficult and priority in the order given by
        sequential_ordering variable (see below).

        Parameters
        ----------    
        fibre :
            The ID of the fibre to be assigned.
            
        candidate_targets :
            A list of potential TaipanTargets to assign.
            
        check_tile_radius :
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile.
            Defaults to True.
            
        check_patrol_radius :
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the patrol radius of this fibre.
            Defaults to True.
            
        check_tile_radius :
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty :
        
            Boolean denoting whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within FIBRE_EXCLUSION_RADIUS +
            TILE_RADIUS of the tile centre; targets for assignment are
            only within TILE_RADIUS). Defaults to True.
        method :
            The method to be used. Must be one of the methods
            specified above.
            
        order_closest_secondary :
            A Boolean value describing if candidate
            targets should be ordered by distance from the fibre rest position
            as a secondary consideration. Has no effect if method='closest'.
            Defaults to True.
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering :
            A three-tuple or length-3 list determining
            in which order to sequence potential targets. It must
            contain the integers 0, 1 and 2, which correspond to the
            position of closest (0), most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (0, 1, 2).

        Returns
        -------    
        remaining_targets :
            The list of candidate_targets, with the newly-
            assigned target removed. If the assignment is unsuccessful, 
            the entire candidate_targets list is returned.
            
        fibre_former_tgt :
            The target that was removed from this fibre
            during the allocation process. Returns None if no object was on
            this fibre originally.
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
                if t.dist_point(fibre_posn) < PATROL_RADIUS]
        if check_tile_radius:
            candidates_this_fibre = [t for t in candidates_this_fibre
                if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]

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
                sequential_ordering[2]] *lists[
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
        # Only targets within FIBRE_EXCLUSION_RADIUS of the newly-assigned
        # target need be computed
        if recompute_difficulty:
            compute_target_difficulties(targets_in_range(tgt.ra, tgt.dec,
                candidate_targets_return, FIBRE_EXCLUSION_RADIUS))

        return candidate_targets_return, fibre_former_tgt

    def assign_tile(self, candidate_targets,
                    check_tile_radius=True, recompute_difficulty=True,
                    method='priority', combined_weight=1.0,
                    sequential_ordering=(1,2),
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
        within FIBRE_EXCLUSION_RADIUS of the target considered.
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
        candidate_targets : :class:`TaipanTarget` list
            Objects to consider assigning to this tile.
        check_tile_radius : bool
            Boolean denoting whether the
            candidate_targets list needs to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty : bool:
            Boolean denoting whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within FIBRE_EXCLUSION_RADIUS +
            TILE_RADIUS of the tile centre; targets for assignment are
            only within TILE_RADIUS). Defaults to True.
            
        method : string
            The method to use to choose the target to assign (see above).
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering :
            A two-tuple or length-2 list determining
            in which order to sequence potential targets. It must
            contain the integers 1 and 2, which correspond to the
            position of most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (1, 2).
            
        overwrite_existing :
            A Boolean value, denoting whether to overwrite an
            existing target allocation if the best fibre for the chosen target
            already has a target assigned. Defaults to False.

        Returns
        -------    
        candidate_targets :
            The list of candidate_targets originally passed, 
            less the target which has been assigned. If no target is assigned,
            the output matches the input.
            
        tile_former_tgt :
            The target that may have been removed from the tile
            during this procedure. Returns None if this did not occur.
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
                'target list (i.e. that would require check_tile_radius)')

        fibre_former_tgt = None

        # Calculate rest positions for all fibres
        fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
            for fibre in BUGPOS_MM if fibre not in FIBRES_GUIDE}

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
            fibre_dists = {fibre: tgt.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])
            # print 'Done!'

            # Attempt to make assignment
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
                if (overwrite_existing 
                    or self._fibres[permitted_fibres[0]] is None):
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

            if not(candidate_found):
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
                      combined_weight=1.0, sequential_ordering=(1,2),
                      check_tile_radius=True,
                      rank_guides=False):
        """
        Assign guides to this tile.

        Guides are assigned their own special fibres. This function will
        attempt to assign up to GUIDES_PER_TILE to the guide fibres. If
        necessary, it will then remove targets from this tile to allow
        up to GUIDES_PER_TILE_MIN to be assigned. This is more complex than for
        standards, because we are not removing targets to assign standards to
        their fibres - rather, we have to work out the best targets to drop so
        other fibres are no longer within FIBRE_EXCLUSION_RADIUS of the guide
        we wish to assign.

        Parameters
        ----------    
        guide_targets :
            The list of candidate guides for assignment.
            
        target_method :
            The method that should be used to determine the
            lowest-priority target to remove to allow for an extra guide
            star assignment to be made. Values for this input are as for
            the 'method' option in assign_tile/unpick_tile. Defaults to
            'priority'.
            
        combined_weight, sequential_ordering :
            Additional control options
            for the specified target_method. See docs for assign_tile/
            unpick_tile for description. Defaults to 1.0 and (1,2)
            respectively.
            
        check_tile_radius :
            Boolean value, denoting whether to reduce the
            guide_targets list to only those targets within the tile radius.
            Defaults to True.
            
        rank_guides :
            Attempt to assign guides in priority order. This allows
            for 'better' guides to be specified. Defaults to False.

        Returns
        -------    
        removed_targets :
            A list of TaipanTargets that have been removed to
            make way for guide fibres. If no targets are removed, the empty
            list is returned. Targets are *not* separated by type (science or
            standard).
        """

        removed_targets = []

        # Calculate rest positions for all GUIDE fibres
        fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
            for fibre in FIBRES_GUIDE}

        guides_this_tile = guide_targets[:]
        if check_tile_radius:
            guides_this_tile = [g for g in guides_this_tile
                if g.dist_point((self.ra, self.dec, )) < TILE_RADIUS]
        if rank_guides:
            guides_this_tile.sort(key=lambda x: -1 * x.priority)

        # Assign up to GUIDES_PER_TILE guides
        # Attempt to assign guide stars to this tile
        assigned_guides = len([t for t in self._fibres.values() 
            if isinstance(t, TaipanTarget)
            and t.guide])
        while assigned_guides < GUIDES_PER_TILE and len(guides_this_tile) > 0:

            guide = guides_this_tile[0]
            if guide.is_target_forbidden(self.get_assigned_targets()):
                guides_this_tile.pop(0)
                continue

            # Identify the closest fibre to this target
            # print 'Finding available fibres...'
            fibre_dists = {fibre: guide.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])

            # Attempt to make assignment
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
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
            # print 'Having to strip targets for guides...'
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
            excluded_by_guides = [i for i in range(len(guides_this_tile))
                if np.any(map(lambda x: x.guide,
                    problem_targets[i]))]
            guides_this_tile = [guides_this_tile[i] 
                for i in range(len(guides_this_tile))
                if i not in excluded_by_guides]
            problem_targets = [problem_targets[i] 
                for i in range(len(problem_targets))
                if i not in excluded_by_guides]

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
                fibres_for_removal = [f for (f, t) in self._fibres.iteritems()
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
    
    def unpick_tile(self, candidate_targets,
                    standard_targets, guide_targets,
                    overwrite_existing=False,
                    check_tile_radius=True, recompute_difficulty=True,
                    method='priority', combined_weight=1.0,
                    sequential_ordering=(0,1,2),
                    rank_supplements=False,
                    repick_after_complete=True,
                    consider_removed_targets=True):
        """
        Unpick this tile, i.e. make a full allocation of targets, guides etc.

        To 'unpick' a tile is to make the optimal permitted allocation of 
        available targets, standards, guides and 'skies' to a particular tile.
        In the case of using a single target allocation method, it is more
        efficient to have a stand-alone function, rather than make repeated 
        calls to assign_fibre or assign_tile, because the ranking criterion for
        targets need only be computed once (although, note that this function
        does call assign_tile and assign_fibre in certain scenarios). 
        For custom unpicking strategies, which may use different target 
        selection methods at different stages, construct your own routine that 
        uses calls to assign_fibre and/or assign_tile.

        This function will attempt to unpick around any existing fibre
        assignments on this tile. Use the overwrite_existing keyword argument
        to alter this behaviour.

        Target assignment methods are as for assign_tile.

        Parameters
        ----------    
        candidate_targets :  :class:`TaipanTarget` list
            Objects to consider
            assigning to this tile. These are the science targets.
            
        standard_targets, guide_targets : :class:`TaipanTarget` list
            Objects
            to consider assigning to this tile as standards and guides
            respectively. Standards, guides and sky fibres are assigned after
            science targets.
            
        overwrite_existing : bool
            Boolean, denoting whether to remove all existing
            fibre assignments on this tile before attempting to unpick. Defaults
            to False.
            
        check_tile_radius : bool
            Boolean denoting whether the
            input target lists need to be trimmed down such that
            all targets are within the tile. Defaults to True.
            
        recompute_difficulty : bool
            Boolean denoting whether to recompute the
            difficulty of the leftover targets after target assignment.
            Note that, if True, check_tile_radius must be True; if not,
            not all of the affected targets will be available to the
            function (targets affected are within FIBRE_EXCLUSION_RADIUS +
            TILE_RADIUS of the tile centre; targets for assignment are
            only within TILE_RADIUS). Defaults to True.
            
        method : string
            The method to use to choose science targets to assign. See the
            documentation for assign_tile for an explanation of the available
            methods.
            
        combined_weight : float
            A float > 0 describing how to weight between
            difficulty and priority. A value of 1.0 (default) weights
            equally. Values below 1.0 preference difficultly; values
            above 1.0 preference priority. Weighting is linear (e.g.
            a value of 2.0 will cause priority to be weighted twice as
            heavily as difficulty).
            
        sequential_ordering :
            A two-tuple or length-2 list determining
            in which order to sequence potential targets. It must
            contain the integers 1 and 2, which correspond to the
            position of most_difficult (1) and priority (2)
            in the ordering sequence. Defaults to (1, 2).
            
        rank_supplements : bool 
            Boolean value, denoting whether for rank the lists
            of standards and guides by their priority. This allows for the
            preferential selection of 'better' guides and standards, if this
            information is encapsulated in their stored priority values.
            Defaults to False.
            
        repick_after_complete : bool 
            value denoting whether to invoke this
            tile's repick_tile function once unpicking is complete. Defaults to
            True.
            
        consider_removed_targets : bool, optional        
            Boolean value denoting whether to
            place targets removed (due to having overwrite_existing=True) back
            into the candidate_targets list. Defaults to True.

        Returns
        -------    
        remaining_targets :
            The list of candidate_targets, less those targets
            which have been assigned to this tile.
            Updated copies of standard_targets and guide_targets are NOT
            returned, as repeating these objects in other tiles is not an issue.
            Any science targets that are removed from the tile and not
            re-assigned will also be appended to this list.
            
        removed_targets :
            Deprecated - will now always be the empty list. A
            warning will be printed if this list somehow becomes non-empty.
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
                'target list (i.e. that would require check_tile_radius)')

        removed_targets = []

        # If overwrite_existing, burn all fibre allocations on this tile
        if overwrite_existing:
            for f in self._fibres:
                if isinstance(self._fibres[f], TaipanTarget):
                    removed_targets.append(self._fibres[f])
                self._fibres[f] = None

        # Calculate rest positions for all non-guide fibres
        fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
            for fibre in BUGPOS_MM if fibre in FIBRES_NORMAL}

        candidate_targets_return = candidate_targets[:]

        # Return the removed targets to the master candidates lists
        if consider_removed_targets:
            removed_candidates = [t for t in removed_targets
                if isinstance(t, TaipanTarget) 
                and not t.guide and not t.standard]
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
        logging.debug('Assigning targets...')
        assigned_tgts = len([t for t in self._fibres.values() 
            if isinstance(t, TaipanTarget)
            and not t.guide and not t.standard])
        while assigned_tgts < TARGET_PER_TILE and len(
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

            # Identify the closest fibre to this target
            # print 'Finding available fibres...'
            fibre_dists = {fibre: tgt.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])
            # print 'Done!'

            # Attempt to make assignment
            candidate_found = False
            while not(candidate_found) and len(permitted_fibres) > 0:
                # print 'Looking to add to fiber...'
                if self._fibres[permitted_fibres[0]] is None:
                    # Assign the target and 'pop' it from the input list
                    # Target comes from either the input list, or from the
                    # removed targets list we generated earlier
                    tgt = candidates_this_tile.pop(i)
                    burn = ranking_list.pop(i)
                    self._fibres[permitted_fibres[0]] = candidate_targets_return.pop(
                        candidate_targets_return.index(tgt))
                    candidate_found = True
                    assigned_tgts += 1
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

        # Assign guides to this tile
        logging.debug('Assigning guides...')
        removed_for_guides = self.assign_guides(guides_this_tile,
            check_tile_radius=not(check_tile_radius), # don't recheck radius
            target_method=method, combined_weight=combined_weight,
            sequential_ordering=sequential_ordering)
        # Put any science targets back in candidate_targets, and any standards
        # back in standards_this_tile
        candidates_this_tile += [t for t in removed_for_guides 
            if isinstance(t, TaipanTarget) and not t.guide and not t.standard]
        candidate_targets_return += [t for t in removed_for_guides 
            if isinstance(t, TaipanTarget) and not t.guide and not t.standard]
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
            fibre_dists = {fibre: std.dist_point(fibre_posns[fibre])
                for fibre in fibre_posns}
            permitted_fibres = sorted([fibre for fibre in fibre_dists
                if fibre_dists[fibre] < PATROL_RADIUS],
                key=lambda x: fibre_dists[x])

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

        if assigned_standards < STANDARDS_PER_TILE_MIN:
            logging.debug('Having to strip targets for standards...')
            standards_this_tile = [t for t in standard_targets
                if t not in assigned_objs]
            if check_tile_radius:
                standards_this_tile = [t for t in standards_this_tile
                    if t.dist_point((self.ra, self.dec)) < TILE_RADIUS]
                # standards_this_tile = targets_in_range(self.ra, self.dec,
                #   standards_this_tile, TILE_RADIUS)
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

        # All fibres except for sky fibres should now be assigned, unless there
        # are some inaccessible fibres for guides/standards. In this case, 
        # we'll call assign_tile on the science targets list to try to 
        # re-populate those fibres before assigning skies
        # Will need to add any science targets in removed_targets back into
        # the candidate_targets list

        candidate_targets_return += [t for t in removed_targets
            if isinstance(t, TaipanTarget) and not t.standard and not t.guide]
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

        # Assign remaining fibres to sky, up to SKY_PER_TILE fibres
        for f in [f for f in self._fibres 
            if self._fibres[f] is None 
            and f not in FIBRES_GUIDE][:SKY_PER_TILE]:
            self._fibres[f] = 'sky'

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
                < FIBRE_EXCLUSION_RADIUS)],
                full_target_list=candidate_targets_return)

        logging.debug('Made tile with %d science, %d standard '
                      'and %d guide targets' %
                      (len(self.get_assigned_targets_science()),
                       len(self.get_assigned_targets_standard()),
                       len(self.get_assigned_targets_guide()), ))


        return candidate_targets_return, removed_targets

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

        Note that guide fibres are NOT included in the repick process.

        Parameters
        ----------     

        Returns
        -------    
        """

        # Do unpicking separately for guides and science/standards/skies
        for fibres_list in [FIBRES_NORMAL, FIBRES_GUIDE]:
            # Calculate rest positions for all fibres
            fibre_posns = {fibre: self.compute_fibre_posn(fibre) 
                for fibre in fibres_list}
            # print fibre_posns

            # Step through the fibres in reverse order of rest position-target
            # distance, and look for ideal swaps (that is, swaps which reduce
            # the rest position-target distance for both candidates).
            # Identify the fibres with targets assigned
            # Do NOT include the guides in this procedure
            fibres_assigned_targets = [fibre for fibre in fibre_posns
                if isinstance(self._fibres[fibre], TaipanTarget)]
            # print fibres_assigned_targets

            # Keep iterating until all fibres have been 'popped' from the 
            # fibres_assigned_targets list for having no better options
            while len(fibres_assigned_targets) > 0:
                # print 'Within reassign loop'
                fibre_dists = {fibre: self._fibres[fibre].dist_point(
                    fibre_posns[fibre]) 
                    for fibre in fibres_assigned_targets}
                # ID the 'worst' remaining assigned fibre
                wf = max(fibre_dists.iteritems(),
                    key=operator.itemgetter(1))[0]
                i = fibres_assigned_targets.index(wf)
                tgt_wf = self._fibres[wf]
                dist_wf = tgt_wf.dist_point(fibre_posns[wf])
                # ID any other fibres that could potentially take this target
                candidate_fibres = [fibre for fibre in fibre_posns 
                    if tgt_wf.dist_point(fibre_posns[fibre]) < PATROL_RADIUS]
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
                        or self._fibres[fibre] == 'sky')
                        and tgt_wf.dist_point(fibre_posns[fibre]) < dist_wf]
                candidate_fibres_better += [fibre for fibre in candidate_fibres
                    if isinstance(self._fibres[fibre], TaipanTarget)
                    and tgt_wf.dist_point(fibre_posns[fibre]) < dist_wf
                    and self._fibres[fibre].dist_point(
                        fibre_posns[wf]) < dist_wf]
                # print candidate_fibres_better
                # print 'Refined candidates: %d' % len(candidate_fibres_better)
                if len(candidate_fibres_better) == 0:
                    # Remove this fibre from further consideration, 
                    # can't be improved
                    fibres_assigned_targets.pop(i)
                else:
                    candidate_fibres_better.sort(
                        key=lambda x: tgt_wf.dist_point(fibre_posns[x])
                    )
                    # Do the swap
                    swap_to = candidate_fibres_better[0]
                    self._fibres[wf] = self._fibres[swap_to]
                    self._fibres[swap_to] = tgt_wf
                    # print 'Swapped %d and %d' % (wf, swap_to, )
                    fibres_assigned_targets = [fibre for fibre in fibre_posns
                    if isinstance(self._fibres[fibre], TaipanTarget)]

    def save_to_file(self, save_path='', return_filename=False):
        """
        Save configuration information for this tile to a simple text config
        file.

        Parameters
        ----------
        save_path:
            The path to save the file to. Can be relative to present
            working directory or absolute. Details to the empty string
            (i.e. file will be saved in present working directory.) Note that,
            if defined, the destination directory must already exist, or an
            IOError will be raised.
        return_filename:
            Boolean value denoting whether the function should return the
            full path to the written file. Defaults to False.

        Returns
        -------
        Nil, UNLESS return_filename=True, then:
        filename:
            Full path to the file written.
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


