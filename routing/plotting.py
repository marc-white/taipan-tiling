"""
Plot paths
"""

from matplotlib.patches import Circle
from matplotlib.pyplot import gcf, gca

import taipan.core as consts

def plot_path_static(path_array):

    fig = gcf()
    fig.clf()
    ax = gca()

    # Plot the tile itself
    tile_circ = Circle((0, 0), radius=consts.TILE_RADIUS / consts.ARCSEC_PER_MM,
                       edgecolor='red', linewidth=1.2,
                       facecolor='none')
    ax.add_artist(tile_circ)

    ax.set_xlim((-1.2 * consts.TILE_RADIUS / consts.ARCSEC_PER_MM,
                 1.2 * consts.TILE_RADIUS / consts.ARCSEC_PER_MM))
    ax.set_ylim((-1.2 * consts.TILE_RADIUS / consts.ARCSEC_PER_MM,
                 1.2 * consts.TILE_RADIUS / consts.ARCSEC_PER_MM))
    ax.set_aspect(1)

    for bug in path_array:
        ax.plot(bug[0, :], bug[1, :], 'b-', lw=0.6)

    # ax.scatter([25, ], [-10, ], marker='x', s=20, c='red')
    # excl_circ = Circle((25, -10), radius=consts.FIBRE_EXCLUSION_RADIUS /
    #                                      consts.ARCSEC_PER_MM,
    #                    edgecolor='red', linewidth=1.2, linestyle='dashed',
    #                    facecolor='none')
    # ax.add_artist(excl_circ)

    return fig
