"""
Plot out bug paths based on a rudimentary force model
"""

import taipan.core as consts

import logging
import numpy as np

ACTIVE_FIBRES = np.sort(consts.FIBRES_NORMAL + consts.FIBRES_GUIDE)

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
            pass
        i += 1

    for bug in sky_fibres:
        # Randomly generate a valid 'sky' position
        pass

    return tgt_pos
