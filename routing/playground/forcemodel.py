"""
Plot out bug paths based on a rudimentary force model
"""

import taipan.core as consts

import scipy.spatial.distance as dist

import logging
import numpy as np
import random

ACTIVE_FIBRES = np.sort(consts.FIBRES_NORMAL + consts.FIBRES_GUIDE)

MAX_STEP_SIZE = 0.2 * consts.FIBRE_EXCLUSION_RADIUS / consts.ARCSEC_PER_MM


def _gen_tile_pos_array(tile, repick_first=False):
    """
    Compute the target position array for a TaipanTile.

    The tile should have had it's :any:`TaipanTile.repick_tile` function
    called before this function is run on it.

    Parameters
    ----------
    tile : :obj:`taipan.core.TaipanTile`
        The TaipanTile to generate the position array for.

    Returns
    -------
    :obj:`numpy.array`
        An array with shape ``(INSTALLED_FIBRES, 2)`` denoting the target
        positions on this tile in x, y mm.
    """

    if repick_first:
        tile.repick_tile()

    tgt_pos = np.zeros((consts.INSTALLED_FIBRES, 2), dtype=float)
    sky_fibres = []

    i = 0
    for bug in ACTIVE_FIBRES:
        if tile._fibres[bug] is None:
            # Set the position values to NaN
            # This will be used as a special value to denote the bug can end
            # up anywhere (i.e can be pushed out of the way)
            tgt_pos[i] = [np.nan, np.nan]
        elif tile._fibres[bug] == 'sky':
            # Put the fibre ID in sky_fibres to assign a random pos later
            sky_fibres.append(bug)
        else:
            # Encode the target position in x, y mm
            # Determine the distance and offset from the tile centre
            dist = consts.dist_points(tile.ra, tile.dec,
                                      tile._fibres[bug].ra,
                                      tile._fibres[bug].dec)
            pa = consts.pa_points(tile.ra, tile.dec,
                                  tile._fibres[bug].ra,
                                  tile._fibres[bug].dec)
            x = dist * np.sin(np.radians(pa)) / consts.ARCSEC_PER_MM
            y = dist * np.cos(np.radians(pa)) / consts.ARCSEC_PER_MM
            tgt_pos[i] = [x, y]
        i += 1

    for bug in sky_fibres:
        # Randomly generate a valid 'sky' position
        i = np.argwhere(ACTIVE_FIBRES == bug)[0]
        x = random.uniform(0., (consts.PATROL_RADIUS / 2.) /
                           consts.ARCSEC_PER_MM)
        y = random.uniform(0., np.sqrt(
            ((consts.PATROL_RADIUS / 2.0) / consts.ARCSEC_PER_MM)**2 - x**2
        ))
        tgt_pos[i] = [x, y]

    return tgt_pos


def _compute_dist_force(arr):
    total_dist = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    if total_dist > MAX_STEP_SIZE:
        return arr / (total_dist / MAX_STEP_SIZE)
    return arr


def _compute_r2_boundary_force(arr):
    total_dist = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    if total_dist < (consts.TILE_RADIUS -
                         consts.FIBRE_EXCLUSION_RADIUS) / consts.ARCSEC_PER_MM:
        return [0., 0., ]
    total_force = MAX_STEP_SIZE / (total_dist - ((consts.TILE_RADIUS -
                                                  consts.FIBRE_EXCLUSION_RADIUS) / consts.ARCSEC_PER_MM) /
                                   (consts.FIBRE_EXCLUSION_DIAMETER /
                                    consts.ARCSEC_PER_MM)) **2
    return arr * (total_force / total_dist)


def findpath_nbody(tile, home_pos=None, repick_first=False):
    """
    Use the n-body pathfinding algorithm to generate tile paths

    This algorithm works by various 'forces' exerted on a fibre, pushing it:

    - Towards the target position;
    - Away from other bugs;
    - Away from the plate edge.

    Every bug is drawn towards its home position by a constant force, :math:`F`.

    Every bug repels every other bug with a force :math:`R`, defined such that
    :math:`R=F` at a distance of :any:`FIBRE_EXCLUSION_RADIUS`.

    The plate edge repels fibres with a force :math:`E`, defined such that
    :math:`E=F` at a distance of one-half of :any:`FIBRE_EXCLUSION_RADIUS` from
    the plate edge.

    A bug subjected to a force equal to :math:`F` will move 1/5th of
    :any:`FIBRE_EXCLUSION_RADIUS` each step. This is the maximum bug speed;
    bugs may move more slowly, but never faster.

    Steps will continue to be computed until one of the following occurs:

    - All bugs reach their home positions;
    - All bugs are commanded to wait (i.e. are in equilibrium) for any step,
      which indicates an impasses has been reached.

    Parameters
    ----------
    tile : :obj:`TaipanTile`
        Tile for pathfinding
    home_pos : :obj:`TaipanTile`
        Tile that we are pathfinding from. Defaults to :obj:`None`, at which
        point ``home_pos`` will be the actual bug home positions

    Returns
    -------
    :obj:`numpy.array`
        A numpy array with dimensions ``((consts.INSTALLED_FIBRES, 2, N))``,
        where ``N`` is the number of steps that were required to reach a
        stable equilibrium.
    :obj:`bool`
        Returns True if target positions have been reached, False if not
    """

    # Generate the target position array
    target_pos = _gen_tile_pos_array(tile, repick_first=repick_first)

    # Generate the start position array
    if home_pos is None:
        home_pos = np.zeros((consts.INSTALLED_FIBRES, 2), dtype=float)
        for i in range(len(ACTIVE_FIBRES)):
            home_pos[i] = list(consts.BUGPOS_MM[ACTIVE_FIBRES[i]])
    else:
        home_pos = _gen_tile_pos_array(tile, repick_first=repick_first)

    # return target_pos, home_pos

    # Stack home_pos with itself as a starting point
    home_pos = np.dstack((home_pos, home_pos))

    i = 0
    while True:
        # Calculate the force exerted on all bugs
        # Note this is all done in x, y mm coordinates for simplicity

        # See how far is left to go to the target positions
        dist_force = target_pos - home_pos[:, :, -1]
        dist_force = np.apply_along_axis(_compute_dist_force,
                                         1, dist_force)

        # Compute the force exerted by each bug on every other bug
        # (except itself)!
        # Distance between all bugs
        distmatrix = dist.cdist(home_pos[:, :, -1], home_pos[:, :, -1],
                                'euclidean')
        # Avoid infinite forces against self
        np.fill_diagonal(distmatrix, -1)
        # Do overall force calculation
        force_matrix = MAX_STEP_SIZE / np.square(
            distmatrix /
            (consts.FIBRE_EXCLUSION_DIAMETER / consts.ARCSEC_PER_MM))
        # No force action on self
        np.fill_diagonal(distmatrix, 0)
        # Calculate aggregate action on each bug
        force_applied = np.dot(force_matrix, distmatrix)

        # return force_applied


        # Compute the force exerted by the plate boundary
        # Easiest thing to do is make this a force acting towards the
        # plate centre position, so home_pos is used directly
        boundary_force = home_pos[:, :, -1]
        # boundary_force = np.array([25, -10]) - home_pos[:, :, -1]
        boundary_force = np.apply_along_axis(_compute_r2_boundary_force,
                                             1, boundary_force)
        # return boundary_force

        total_force = dist_force + boundary_force + force_applied
        # total_force = np.where(total_force > MAX_STEP_SIZE, MAX_STEP_SIZE,
        #                        total_force)

        # Tack on the result of the next step
        home_pos = np.dstack((home_pos,
                              home_pos[:, :, -1] + total_force))

        i += 1
        if i % 200 == 0:
            print('Up to %d steps' % i)

        if np.all(np.abs(home_pos[:, :, -1] - target_pos) < 0.5):
            break
        if np.all(np.abs(home_pos[:, :, -1] - home_pos[:, :, -2]) < 0.1):
            break
        if i == 1000:
            break

    print('Completed pathfinding in %d steps' % i)

    return home_pos[1:, :, :]  # Strip the redundant first move off
