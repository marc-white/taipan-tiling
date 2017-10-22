"""Plotting functions to visualise/record the results of a FunnelWeb tiling run
"""
import taipan.core as tp
import matplotlib.patches as mpatches
import numpy as np
from collections import OrderedDict
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import cycle
import cv2
import glob

def plot_tiling(tiling, run_settings):
    """Function to plot an overview of a FunnelWeb tiling run.
    
    Plots:
    - Tile centres on sky using aitoff projection
    - Box & whisker plots of target/standard/guide assignments
    - Completeness progression (targets as a function of tiles)
    - Table of run_settings
    - Histograms of targets per tile for all tiles, as well as each magnitude bin
    - Histograms of standards per tile for all tiles, as well as each magnitude bin
    - Histograms of guides per tile for all tiles, as well as each magnitude bin
    
    Parameters
    ----------
    tiling: list
        The list of TaipanTiles from a tiling run.
    
    run_settings: OrderedDict
        An OrderedDict containing input settings and results from the tiling run.
    
    """
    # Initialise plot, use GridSpec to have a NxM grid, write title
    plt.clf()
    gs = gridspec.GridSpec(5,4)
    gs.update(wspace=0.2)
    fig = plt.gcf()
    fig.set_size_inches(24., 24.)
    plt.suptitle(run_settings["run_id"] + ": " + run_settings["description"], fontsize=30)
    
    # Separate targets, standards, and guides by magnitude range
    tiles_by_mag_range = []
    standards_by_mag_range = []
    guides_by_mag_range = []
    
    # Initialise legend label list
    tile_count_labels = ["All"]
        
    for mrange in run_settings["mag_ranges"]:
        tiles_for_range = []
        standards_for_range = []
        guides_for_range = []
        
        for tile in tiling:
            if tile.mag_min == mrange[0] and tile.mag_max == mrange[1]:
                tiles_for_range.append(tile.count_assigned_targets_science())
                standards_for_range.append(tile.count_assigned_targets_standard())
                guides_for_range.append(tile.count_assigned_targets_guide())
        
        tiles_by_mag_range.append(tiles_for_range)
        standards_by_mag_range.append(standards_for_range)
        guides_by_mag_range.append(guides_for_range)
        
        tile_count_labels.append("Mag range: %s-%s (%s tiles)" % (mrange[0], mrange[1],
                                 len(tiles_for_range)))
    
    # Insert the non-magnitude range specific numbers
    tiles_by_mag_range.insert(0, run_settings["targets_per_tile"]) 
    standards_by_mag_range.insert(0, run_settings["standards_per_tile"]) 
    guides_by_mag_range.insert(0, run_settings["guides_per_tile"])
    
    # ------------------------------------------------------------------------------------
    # Tile positions
    # ------------------------------------------------------------------------------------
    # Plot an Aitoff projection of the tile centre positions on-sky
    ax0 = fig.add_subplot(gs[0,0], projection='aitoff')
    ax0.grid(True)
    
    # Count the number of tiles per field
    coords = Counter(["%f_%f" % (tile.ra, tile.dec) for tile in tiling])
    coords = np.array([[float(key.split("_")[0]), float(key.split("_")[1]), coords[key]] 
                      for key in coords.keys()])
    
    ax0_plt = ax0.scatter(np.radians(coords[:,0] - 180.), np.radians(coords[:,1]), c=coords[:,2],
             marker='o', lw=0, s=3, cmap="rainbow")
    ax0.set_title('Tile centre positions', y=1.1)
    ax0.set_axisbelow(True)
    
    # Colour bar
    ax0_plt.set_clim([np.min(coords[:,2]), np.max(coords[:,2])])
    cbar = plt.colorbar(ax0_plt, orientation='horizontal')
    cbar.set_label("# Tiles")
    
    ax0.tick_params(axis='both', which='major', labelsize=6)
    ax0.tick_params(axis='both', which='minor', labelsize=6)

    
    # ------------------------------------------------------------------------------------
    # Box-and-whisker plots of number distributions
    # ------------------------------------------------------------------------------------
    # Plot box-and-whisker plots of the targets, standards, and guides
    ax1 = fig.add_subplot(gs[1,0])
    ax1.boxplot([run_settings["targets_per_tile"], run_settings["standards_per_tile"], 
                 run_settings["guides_per_tile"]], vert=False)
    ax1.set_yticklabels(['T', 'S', 'G'])
    ax1.set_title('Box-and-whisker plots of number of assignments')

    # ------------------------------------------------------------------------------------
    # Number of Tiles vs Completeness
    # ------------------------------------------------------------------------------------
    # Plot tiling completeness as number of targets as a function of number of tiles
    ax2 = fig.add_subplot(gs[2,0])
    targets_per_tile_sorted = sorted(run_settings["targets_per_tile"], 
                                     key=lambda x: -1.*x)
    xpts = np.asarray(range(len(targets_per_tile_sorted))) + 1
    ypts = [np.sum(targets_per_tile_sorted[:i+1]) for i in xpts]
    ax2.plot(xpts, ypts, 'k-', lw=.9)
    ax2.plot(len(tiling), np.sum(run_settings["targets_per_tile"]), 'ro',
             label='No. of tiles: %d' % len(tiling))
    ax2.hlines(run_settings["num_targets"], ax2.get_xlim()[0], ax2.get_xlim()[1], lw=.75,
               colors='k', linestyles='dashed', label='100% completion')
    ax2.hlines(0.975 * run_settings["num_targets"], ax2.get_xlim()[0], ax2.get_xlim()[1], 
               lw=.75, colors='k', linestyles='dashdot', label='97.5% completion')
    ax2.hlines(0.95 * run_settings["num_targets"], ax2.get_xlim()[0], ax2.get_xlim()[1], 
               lw=.75, colors='k', linestyles='dotted', label='95% completion')
    ax2.legend(loc='lower right', title='Time to %3.1f comp.: %dm:%2.1fs' % (
               run_settings["completeness_target"] * 100., 
               int(np.floor(run_settings["mins_to_complete"])), 
               run_settings["mins_to_complete"] % 60.))
    ax2.set_title('Completeness progression')
    ax2.set_xlabel('No. of tiles')
    ax2.set_ylabel('No. of assigned targets')
    ax2.set_xlim([0, 1.05*len(tiling)])
    ax2.set_ylim([0, 1.05*run_settings["num_targets"]])

    # ------------------------------------------------------------------------------------
    # Table of Run Settings
    # ------------------------------------------------------------------------------------
    # Plot a table for referencing the run settings/results
    ax3 = fig.add_subplot(gs[3:5,0])
    col_labels = ("Parameter", "Value")
    ax3.axis("off")
    
    # Prep table
    fmt_table = []
    
    for key, value in zip(run_settings.keys(), run_settings.values()):
        # Show only file name (not full path) for space constraints
        if (type(value) is str or type(value) is np.string_) and "/" in value:
            fmt_table.append([key, value.split("/")[-1]])
        # Format any short lists as strings, don't show long lists
        elif type(value) is list:
            if len(value) <= 10:
                fmt_table.append([key, str(value)])
        # Leave all other values as is
        else:
            fmt_table.append([key, value])
    
    settings_tab = ax3.table(cellText=fmt_table, colLabels=col_labels, loc="center")
    settings_tab.set_fontsize(8)
    settings_tab.scale(1.0,0.7)
    
    # ------------------------------------------------------------------------------------
    # Create plots for all tiles, as well as each magnitude range
    # ------------------------------------------------------------------------------------
    # Initialise axes lists, and colour format
    ax4 = []
    ax5 = []
    ax6 = []
    plt_colours = ["MediumSeaGreen","SkyBlue","Gold","Orange", "Tomato"]
    colour_cycler = cycle(plt_colours)
    
    for i, label in enumerate(tile_count_labels): 
        # Match the colours for each magnitude range
        colour = next(colour_cycler)
        
        # --------------------------------------------------------------------------------
        # Number of Targets per Tile
        # --------------------------------------------------------------------------------
        # Plot a histogram of the number of targets per tile
        ax4.append(fig.add_subplot(gs[i,1]))
        ax4[-1].hist(tiles_by_mag_range[i], bins= max(run_settings["targets_per_tile"]), 
                     color=colour, align='right', label=label)
        ax4[-1].vlines(tp.TARGET_PER_TILE, ax4[-1].get_ylim()[0], ax4[-1].get_ylim()[1], 
                       linestyles='dashed', colors='k', label='Ideally-filled tile')
        ax4[-1].legend(loc='upper center')
        ax4[-1].set_xlabel('No. of targets per tile')
        ax4[-1].set_ylabel('Frequency')
        ax4[-1].set_yscale('log')
        ax4[-1].set_xlim(0, max(run_settings["targets_per_tile"]) + 1)
        
        if len(tiles_by_mag_range[i]) > 0:
            tile_mean = np.mean(tiles_by_mag_range[i])
            tile_median = np.median(tiles_by_mag_range[i])
        else:
            tile_mean = 0
            tile_median = 0
        
        ax4[-1].text(0.5, 0.5, "Mean: %i, Median: %i" % (tile_mean, tile_median),
                     ha="center", transform=ax4[-1].transAxes)

        # --------------------------------------------------------------------------------
        # Number of standards per tile
        # --------------------------------------------------------------------------------
        # Plot a histogram of the number of standards per tile
        ax5.append(fig.add_subplot(gs[i,2]))
        ax5[-1].hist(standards_by_mag_range[i], 
                     bins=max(run_settings["standards_per_tile"]), color=colour, 
                     align='right', label=label)
        ax5[-1].vlines(tp.STANDARDS_PER_TILE, ax5[-1].get_ylim()[0], 
                       ax5[-1].get_ylim()[1], linestyles='dashed', colors='k', 
                       label='Ideally-filled tile')
        ax5[-1].vlines(tp.STANDARDS_PER_TILE_MIN, ax5[-1].get_ylim()[0], 
                       ax5[-1].get_ylim()[1], linestyles='dotted',  colors='k', 
                       label='Minimum standards per tile')
        ax5[-1].set_xlabel('No. of standards per tile')
        ax5[-1].set_ylabel('Frequency')
        ax5[-1].legend(loc='upper center')
        ax5[-1].set_xlim(0, max(run_settings["standards_per_tile"]) + 1)
        
        if len(standards_by_mag_range[i]) > 0:
            standard_mean = np.mean(standards_by_mag_range[i])
            standard_median = np.median(standards_by_mag_range[i])
        else:
            standard_mean = 0
            standard_median = 0
        
        ax5[-1].text(ax5[-1].get_xlim()[1]/2, ax5[-1].get_ylim()[1]/2,
                     "Mean: %i, Median: %i" % (standard_mean, standard_median), 
                     ha="center")            

        # --------------------------------------------------------------------------------
        # Number of guides per tile
        # --------------------------------------------------------------------------------
        # Plot a histogram of the number of guides per fibre
        ax6.append(fig.add_subplot(gs[i,3]))
        ax6[-1].hist(guides_by_mag_range[i], bins=max(run_settings["guides_per_tile"]), 
                     color=colour, align='right', label=label)
        ax6[-1].vlines(tp.GUIDES_PER_TILE, ax6[-1].get_ylim()[0], ax6[-1].get_ylim()[1], 
                       linestyles='dashed', colors='k', label='Ideally-filled tile')
        ax6[-1].vlines(tp.GUIDES_PER_TILE_MIN, ax6[-1].get_ylim()[0], 
                       ax6[-1].get_ylim()[1], linestyles='dotted', colors='k', 
                       label='Minimum guides per tile')
        ax6[-1].set_xlabel('No. of guides per tile')
        ax6[-1].set_ylabel('Frequency')
        ax6[-1].legend(loc='upper center')
        ax6[-1].set_xlim(0, max(run_settings["guides_per_tile"]) + 1)
        
        if len(guides_by_mag_range[i]) > 0:
            guides_mean = np.mean(guides_by_mag_range[i])
            guides_median = np.median(guides_by_mag_range[i])
        else:
            guides_mean = 0
            guides_median = 0
        
        ax6[-1].text(ax6[-1].get_xlim()[1]/2, ax6[-1].get_ylim()[1]/2,
                     "Mean: %i, Median: %i" % (guides_mean, guides_median), ha="center") 

    # Set plot titles
    ax3.set_title("Run Settings & Overview", y=0.96)
    ax4[0].set_title("Stars per Tile")
    ax5[0].set_title("Standards per Tile")
    ax6[0].set_title("Guides per Tile")

    # Save plot
    name = "results/" + run_settings["run_id"] + "_tiling_run_overview.pdf"
    fig.savefig(name, fmt='pdf')
    
def create_tiling_visualisation(tiling, run_settings, increment=1000):
    """
    For increasingly larger slices of the tiling set, run the plotting code.
    """
    # 
    steps = list(np.arange(increment, len(tiling), increment))
    
    if steps[-1] != len(tiling):
        steps.append(len(tiling))
    
    for frame, step in enumerate(steps):
        plot_tiling(tiling[:step], run_settings)
        
        name = "results/visualisation/%s_overview_f%i.png" % (run_settings["run_id"], 
                                                              frame)
        plt.savefig(name)
        
def create_video(run_id):
    """
    """
    image_paths = glob.glob("results/visualisation/*%s*" % run_id)
    image_paths.sort()
    
    height, width, layers = cv2.imread(image_paths[0]).shape
    
    images = [cv2.imread(path) for path in image_paths]
    
    video = cv2.VideoWriter("results/visualisation/%s_tiling_visualisation.avi" % run_id, 
                            -1, 2, (width, height), 1)
    
    for frame in images:
        video.write(frame)
        
    cv2.destroyAllWindows()
    video.release()