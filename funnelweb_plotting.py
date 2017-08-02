"""
"""
import taipan.core as tp
import matplotlib.patches as mpatches
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def plot_tiling(test_tiling, remaining_targets, run_settings):
    """
    """
    run_id = run_settings["run_id"]
    description = run_settings["description"]
    time_to_complete = run_settings["mins_to_complete"]
    completeness_target = run_settings["completeness_target"]
    num_targets = run_settings["num_targets"]
    alloc_method = run_settings["alloc_method"]
    combined_weight = run_settings["combined_weight"]
    ranking_method = run_settings["ranking_method"]
    tiling_method = run_settings["tiling_method"]
    #non_standard_targets_per_tile = run_settings["non_standard_targets_per_tile"]
    #targets_per_tile = run_settings["targets_per_tile"]
    #standards_per_tile = run_settings["targets_per_tile"]
    #guides_per_tile = run_settings["targets_per_tile"]
    non_standard_targets_per_tile = [t.count_assigned_targets_science(include_science_standards=False) for t in test_tiling]
    targets_per_tile = [t.count_assigned_targets_science() for t in test_tiling]
    standards_per_tile = [t.count_assigned_targets_standard() for t in test_tiling]
    guides_per_tile = [t.count_assigned_targets_guide() for t in test_tiling]
    
    
    plt.clf()
    fig = plt.gcf()
    plt.suptitle(run_id + ": " + description)
    fig.set_size_inches(12., 18.)

    # Tile positions
    ax1 = fig.add_subplot(421, projection='aitoff')
    ax1.grid(True)
    # for tile in test_tiling:
        # tile_verts = np.asarray([tp.compute_offset_posn(tile.ra, 
        #     tile.dec, tp.TILE_RADIUS, float(p)) for p in range(361)])
        # tile_verts = np.asarray([tp.aitoff_plottable(xy, ra_offset=180.) for xy
        #     in tile_verts])
        # ec = 'k'
        # tile_circ = mpatches.Polygon(tile_verts, closed=False,
        #     edgecolor=ec, facecolor='none', lw=.5)
        # ax1.add_patch(tile_circ)
    ax1.plot([np.radians(t.ra - 180.) for t in test_tiling], [np.radians(t.dec) 
        for t in test_tiling],
        'ko', lw=0, ms=1)
    ax1.set_title('Tile centre positions')

    #plt.show()
    #plt.draw()

    # Box-and-whisker plots of number distributions
    ax0 = fig.add_subplot(423)
    ax0.boxplot([targets_per_tile, standards_per_tile, guides_per_tile],
        vert=False)
    ax0.set_yticklabels(['T', 'S', 'G'])
    ax0.set_title('Box-and-whisker plots of number of assignments')

    #plt.show()
    #plt.draw()

    # Move towards completeness
    ax3 = fig.add_subplot(223)
    targets_per_tile_sorted = sorted(targets_per_tile, key=lambda x: -1.*x)
    xpts = np.asarray(range(len(targets_per_tile_sorted))) + 1
    ypts = [np.sum(targets_per_tile_sorted[:i+1]) for i in xpts]
    ax3.plot(xpts, ypts, 'k-', lw=.9)
    ax3.plot(len(test_tiling), np.sum(targets_per_tile), 'ro',
        label='No. of tiles: %d' % len(test_tiling))
    ax3.hlines(num_targets, ax3.get_xlim()[0], ax3.get_xlim()[1], lw=.75,
        colors='k', linestyles='dashed', label='100% completion')
    ax3.hlines(0.975 * num_targets, ax3.get_xlim()[0], ax3.get_xlim()[1], 
        lw=.75,
        colors='k', linestyles='dashdot', label='97.5% completion')
    ax3.hlines(0.95 * num_targets, ax3.get_xlim()[0], ax3.get_xlim()[1], lw=.75,
        colors='k', linestyles='dotted', label='95% completion')
    ax3.legend(loc='lower right', title='Time to %3.1f comp.: %dm:%2.1fs' % (
        completeness_target * 100., 
        int(np.floor(time_to_complete/60.)), time_to_complete % 60.))
    ax3.set_title('Completeness progression')
    ax3.set_xlabel('No. of tiles')
    ax3.set_ylabel('No. of assigned targets')

    #plt.show()
    #plt.draw()

    # No. of targets per tile
    target_range = max(targets_per_tile) - min(targets_per_tile)
    ax2 = fig.add_subplot(322)
    ax2.hist(targets_per_tile, bins=target_range, align='right')
    ax2.vlines(tp.TARGET_PER_TILE, ax2.get_ylim()[0], ax2.get_ylim()[1], linestyles='dashed',
        colors='k', label='Ideally-filled tile')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('No. of targets per tile')
    ax2.set_ylabel('Frequency')

    #plt.show()
    #plt.draw()

    # No. of standards per tile
    standard_range = max(standards_per_tile) - min(standards_per_tile)
    if standard_range > 0:
        ax4 = fig.add_subplot(324)
        ax4.hist(standards_per_tile, bins=standard_range, align='right')
        ax4.vlines(tp.STANDARDS_PER_TILE, ax4.get_ylim()[0], ax4.get_ylim()[1], 
            linestyles='dashed',
            colors='k', label='Ideally-filled tile')
        ax4.vlines(tp.STANDARDS_PER_TILE_MIN, ax4.get_ylim()[0], ax4.get_ylim()[1], 
            linestyles='dotted',
            colors='k', label='Minimum standards per tile')
        ax4.set_xlabel('No. of standards per tile')
        ax4.set_ylabel('Frequency')
        ax4.legend(loc='upper left')

    #plt.show()
    #plt.draw()

    # No. of guides per tile
    guide_range = max(guides_per_tile) - min(guides_per_tile)

    ax5 = fig.add_subplot(326)
    ax5.hist(guides_per_tile, bins=max(guides_per_tile), align='right')
    ax5.vlines(tp.GUIDES_PER_TILE, ax5.get_ylim()[0], ax5.get_ylim()[1], 
        linestyles='dashed',
        colors='k', label='Ideally-filled tile')
    ax5.vlines(tp.GUIDES_PER_TILE_MIN, ax5.get_ylim()[0], ax5.get_ylim()[1], 
        linestyles='dotted',
        colors='k', label='Minimum guides per tile')
    ax5.set_xlabel('No. of guides per tile')
    ax5.set_ylabel('Frequency')
    ax5.legend(loc='upper left')

    #plt.show()
    #plt.draw()

    supp_str = ''
    if alloc_method == 'combined_weighted':
        supp_str = str(combined_weight)
    elif alloc_method == 'sequential':
        supp_str = str(sequential_ordering)
    ax2.set_title('GREEDY %s - %d targets, %s tiling, %s unpicking (%s)' % (
        ranking_method,num_targets,
        tiling_method, alloc_method, supp_str))
    try:
        plt.tight_layout()
    except:
        pass

    plt.show()
    plt.draw()

    # Save plot
    name = "results/" + run_id + "_results_" + \
        'greedy-%s_%s_%s%s.pdf' % (ranking_method, tiling_method, alloc_method, supp_str)
    fig.savefig(name, fmt='pdf')
    
if __name__ == "__main__":
    # Plot
    plot_tiling(test_tiling, remaining_targets, run_settings)