"""Plotting functions to visualise/record the results of a FunnelWeb tiling run
"""
import taipan.core as tp
import taipan.fwtiling as fwtl
import matplotlib.patches as mpatches
import numpy as np
from collections import OrderedDict
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from itertools import cycle
import glob

def plot_tiling(tiling, run_settings, plot_other=False):
    """Function to plot an overview of a FunnelWeb tiling run.
    
    Plots:
    - Tile centres on sky using aitoff projection
    - Box & whisker plots of target/standard/guide assignments
    - Completeness progression (targets as a function of tiles)
    - Table of run_settings
    - Histograms of targets per tile for all tiles, plus each magnitude bin
    - Histograms of standards per tile for all tiles, plus each magnitude bin
    - Histograms of guides per tile for all tiles, plus each magnitude bin
    
    Parameters
    ----------
    tiling: list
        The list of TaipanTiles from a tiling run.
    
    run_settings: OrderedDict
        OrderedDict containing input settings and results from the tiling run.
        
    plot_other: boolean
        Whether to plot box-and-whisker and completeness target plots, or 
        instead plot the variable table in a larger format.
    """
    # Initialise plot, use GridSpec to have a NxM grid, write title
    plt.clf()
    gs = gridspec.GridSpec(5,6)
    gs.update(wspace=0.2)
    fig = plt.gcf()
    fig.set_size_inches(28.8, 24.)
    plt.suptitle(run_settings["run_id"] + ": " + run_settings["description"], 
                 fontsize=30)
    
    # Separate targets, standards, and guides by magnitude range
    tiles_by_mag_range = [tiling]
    targets_by_mag_range = []
    standards_by_mag_range = []
    guides_by_mag_range = []
    sky_by_mag_range = []
    
    # Initialise legend label list
    tile_count_labels = ["All"]
        
    for mrange in run_settings["mag_ranges"]:
        tiles_for_range = []
        targets_for_range = []
        standards_for_range = []
        guides_for_range = []
        sky_for_range = []
        
        for tile in tiling:
            if tile.mag_min == mrange[0] and tile.mag_max == mrange[1]:
                tiles_for_range.append(tile)
                targets_for_range.append(tile.count_assigned_targets_science())
                standards_for_range.append(
                    tile.count_assigned_targets_standard())
                guides_for_range.append(tile.count_assigned_targets_guide())
                sky_for_range.append(tile.count_assigned_targets_sky())
        
        tiles_by_mag_range.append(tiles_for_range)
        targets_by_mag_range.append(targets_for_range)
        standards_by_mag_range.append(standards_for_range)
        guides_by_mag_range.append(guides_for_range)
        sky_by_mag_range.append(sky_for_range)
        
        tile_count_labels.append("Mag range: %s-%s" % (mrange[0], mrange[1]))
    
    targets_per_tile = run_settings["targets_per_tile"]
    standards_per_tile = run_settings["standards_per_tile"]
    guides_per_tile = run_settings["guides_per_tile"]
    sky_per_tile = run_settings["sky_per_tile"]
    
    # Insert the non-magnitude range specific numbers
    targets_by_mag_range.insert(0, [t.count_assigned_targets_science() 
                                  for t in tiling]) 
    standards_by_mag_range.insert(0, [t.count_assigned_targets_standard() 
                                      for t in tiling]) 
    guides_by_mag_range.insert(0, [t.count_assigned_targets_guide() 
                                   for t in tiling])
    sky_by_mag_range.insert(0, [t.count_assigned_targets_sky() 
                                for t in tiling])      
                                
    # Now count the number of unique targets per magnitude range
    unique_targets_range =  []
    for mag_range in tiles_by_mag_range:
        unique, tot, dup = fwtl.count_unique_science_targets(mag_range, True) 
        unique_targets_range.append(unique)                   
    
    # -------------------------------------------------------------------------
    # Tile positions
    # -------------------------------------------------------------------------
    # Plot an Aitoff projection of the tile centre positions on-sky
    ax0 = fig.add_subplot(gs[0:2,0:2], projection='aitoff')
    ax0.grid(True)
    
    # Count the number of tiles per field
    coords = Counter(["%f_%f" % (tile.ra, tile.dec) for tile in tiling])
    coords = np.array([[float(key.split("_")[0]), float(key.split("_")[1]), 
                      coords[key]] for key in coords.keys()])
    
    ax0_plt = ax0.scatter(np.radians(coords[:,0] - 180.), 
                          np.radians(coords[:,1]), c=coords[:,2], marker='o',
                          lw=0, s=9, cmap="rainbow")
    ax0.set_title('Tile centre positions', y=1.1)
    ax0.set_axisbelow(True)
    
    # Colour bar
    ax0_plt.set_clim([np.min(coords[:,2]), np.max(coords[:,2])])
    cbar = plt.colorbar(ax0_plt, orientation='horizontal')
    cbar.set_label("# Tiles")
    
    ax0.tick_params(axis='both', which='major', labelsize=10)
    ax0.tick_params(axis='both', which='minor', labelsize=10)

    
    # -------------------------------------------------------------------------
    # Box-and-whisker plots of number distributions
    # -------------------------------------------------------------------------
    # Plot box-and-whisker plots of the targets, standards, and guides
    if plot_other:
        ax1 = fig.add_subplot(gs[1,0])
        ax1.boxplot([targets_per_tile, standards_per_tile, guides_per_tile], 
                    vert=False)
        ax1.set_yticklabels(['T', 'S', 'G'])
        ax1.set_title('Box-and-whisker plots of number of assignments')

    # -------------------------------------------------------------------------
    # Number of Tiles vs Completeness
    # -------------------------------------------------------------------------
    # Plot tiling completeness as # of targets as a function of # tiles
        ax2 = fig.add_subplot(gs[2,0])
        targets_per_tile_sorted = sorted(targets_per_tile, key=lambda x: -1.*x)
        xpts = np.asarray(range(len(targets_per_tile_sorted))) + 1
        ypts = [np.sum(targets_per_tile_sorted[:i+1]) for i in xpts]
        ax2.plot(xpts, ypts, 'k-', lw=.9)
        ax2.plot(len(tiling), np.sum(targets_per_tile), 'ro',
                 label='No. of tiles: %d' % len(tiling))
        ax2.hlines(run_settings["num_targets"], ax2.get_xlim()[0], 
                   ax2.get_xlim()[1], lw=.75, colors='k', linestyles='dashed', 
                   label='100% completion')
        ax2.hlines(0.975 * run_settings["num_targets"], ax2.get_xlim()[0], 
                   ax2.get_xlim()[1], lw=.75, colors='k', linestyles='dashdot', 
                   label='97.5% completion')
        ax2.hlines(0.95 * run_settings["num_targets"], ax2.get_xlim()[0], 
                   ax2.get_xlim()[1], lw=.75, colors='k', linestyles='dotted', 
                   label='95% completion')
        ax2.legend(loc='lower right', 
                   title='Time to %3.1f comp.: %dm:%2.1fs' % (
                   run_settings["tiling_completeness"] * 100., 
                   int(np.floor(run_settings["mins_to_complete"])), 
                   run_settings["mins_to_complete"] % 60.))
        ax2.set_title('Completeness progression')
        ax2.set_xlabel('No. of tiles')
        ax2.set_ylabel('No. of assigned targets')
        ax2.set_xlim([0, 1.05*len(tiling)])
        ax2.set_ylim([0, 1.05*run_settings["num_targets"]])

    # -------------------------------------------------------------------------
    # Table of Run Settings
    # -------------------------------------------------------------------------
    # Plot a table for referencing the run settings/results
    if plot_other:
        ax3 = fig.add_subplot(gs[3:5,0])
    else:
        ax3 = fig.add_subplot(gs[2:5,0:2])
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
        # Leave all other values as is (but don't plot the description, as it
        # is already at the top of the plot and likely to be long
        elif key != "description":
            fmt_table.append([key, value])
    
    settings_tab = ax3.table(cellText=fmt_table, colLabels=col_labels, 
                             loc="center")
    if plot_other:
        settings_tab.set_fontsize(7)
        settings_tab.scale(1.0, 0.6)
    else:
        settings_tab.auto_set_font_size(False)
        settings_tab.set_fontsize(9.5)
        #settings_tab.scale(2.2, 1.2)
    
    # -------------------------------------------------------------------------
    # Create plots for all tiles, as well as each magnitude range
    # -------------------------------------------------------------------------
    # Initialise axes lists, and colour format
    ax4 = []
    ax5 = []
    ax6 = []
    ax7 = []
    plt_colours = ["MediumSeaGreen","SkyBlue","Gold","Orange", "Tomato"]
    colour_cycler = cycle(plt_colours)
    
    for i, label in enumerate(tile_count_labels): 
        # Match the colours for each magnitude range
        colour = next(colour_cycler)
        
        # ---------------------------------------------------------------------
        # Number of Targets per Tile
        # ---------------------------------------------------------------------
        # Plot a histogram of the number of targets per tile
        ax4.append(fig.add_subplot(gs[i,2]))
        ax4[-1].hist(targets_by_mag_range[i], 
                     bins=np.arange(0, max(targets_per_tile)+1, 1), 
                     color=colour, align='right', label=label)
        ax4[-1].vlines(run_settings["TARGET_PER_TILE"], ax4[-1].get_ylim()[0],  
                       ax4[-1].get_ylim()[1], linestyles='dashed', colors='k', 
                       label='Ideally-filled tile')
        ax4[-1].legend(loc='upper center')
        ax4[-1].set_xlabel('No. of targets per tile')
        ax4[-1].set_ylabel('Frequency')
        ax4[-1].set_yscale('log')
        ax4[-1].set_xlim(0, max(targets_per_tile) + 1)
        ax4[-1].xaxis.set_major_locator(ticker.MultipleLocator(10))
        
        ax4[-1].tick_params(axis='both', which='major', labelsize=7)
        ax4[-1].tick_params(axis='both', which='minor', labelsize=7)
        
        if len(targets_by_mag_range[i]) > 0:
            tile_mean = np.mean(targets_by_mag_range[i])
            tile_median = np.median(targets_by_mag_range[i])
        else:
            tile_mean = 0
            tile_median = 0
        
        # Now plot text describing the number of tiles, targets, stats, and
        # completion target for all histogram plots (with the exception of 
        # skipping completion target on the total histogram)
        if i > 0:
            completeness = run_settings["completeness_targets"][i-1] * 100
            ax4[-1].text(0.5, 0.4, "%5.2f %% Completion" % completeness,
                         ha="center", transform=ax4[-1].transAxes)
        
        ax4[-1].text(0.5, 0.5, "Mean: %i, Median: %i" % (tile_mean, 
                                                         tile_median),
                     ha="center", transform=ax4[-1].transAxes)
        
        ax4[-1].text(0.5, 0.6, 
                     "{:,} Unique Targets".format(unique_targets_range[i]),
                     ha="center", transform=ax4[-1].transAxes)
                     
        ax4[-1].text(0.5, 0.7, 
                     "{:,} Tiles".format(len(tiles_by_mag_range[i])),
                     ha="center", transform=ax4[-1].transAxes)
                     
        
         
        # ---------------------------------------------------------------------
        # Number of standards per tile
        # ---------------------------------------------------------------------
        # Plot a histogram of the number of standards per tile
        ax5.append(fig.add_subplot(gs[i,3]))
        ax5[-1].hist(standards_by_mag_range[i], 
                     bins=max(standards_per_tile), color=colour, 
                     align='right', label=label)
        ax5[-1].vlines(run_settings["STANDARDS_PER_TILE"], 
                       ax5[-1].get_ylim()[0], ax5[-1].get_ylim()[1], 
                       linestyles='dashed', colors='k', 
                       label='Ideally-filled tile')
        ax5[-1].vlines(run_settings["STANDARDS_PER_TILE_MIN"],
                       ax5[-1].get_ylim()[0], ax5[-1].get_ylim()[1],
                       linestyles='dotted',  colors='k', 
                       label='Minimum standards per tile')
        ax5[-1].set_xlabel('No. of standards per tile')
        ax5[-1].set_ylabel('Frequency')
        ax5[-1].legend(loc='upper center')
        ax5[-1].set_xlim(0, max(standards_per_tile) + 1)
        #ax5[-1].xaxis.set_major_locator(ticker.MultipleLocator(2))
        
        ax5[-1].tick_params(axis='both', which='major', labelsize=7)
        ax5[-1].tick_params(axis='both', which='minor', labelsize=7)
        
        if len(standards_by_mag_range[i]) > 0:
            standard_mean = np.mean(standards_by_mag_range[i])
            standard_median = np.median(standards_by_mag_range[i])
        else:
            standard_mean = 0
            standard_median = 0
        
        ax5[-1].text(ax5[-1].get_xlim()[1]/2, ax5[-1].get_ylim()[1]/2,
                     "Mean: %i, Median: %i" % (standard_mean, standard_median), 
                     ha="center")             
        
        # ---------------------------------------------------------------------
        # Number of guides per tile
        # ---------------------------------------------------------------------
        # Plot a histogram of the number of guides per tile
        ax6.append(fig.add_subplot(gs[i,4]))
        ax6[-1].hist(guides_by_mag_range[i], bins=max(guides_per_tile), 
                     color=colour, align='right', label=label)
        ax6[-1].vlines(run_settings["GUIDES_PER_TILE"], ax6[-1].get_ylim()[0], 
                       ax6[-1].get_ylim()[1], linestyles='dashed', colors='k', 
                       label='Ideally-filled tile')
        ax6[-1].vlines(run_settings["GUIDES_PER_TILE_MIN"], 
                       ax6[-1].get_ylim()[0], ax6[-1].get_ylim()[1],
                       linestyles='dotted', colors='k', 
                       label='Minimum guides per tile')
        ax6[-1].set_xlabel('No. of guides per tile')
        ax6[-1].set_ylabel('Frequency')
        ax6[-1].legend(loc='upper center')
        ax6[-1].set_xlim(0, max(guides_per_tile) + 1)
        ax6[-1].xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax6[-1].tick_params(axis='both', which='major', labelsize=7)
        ax6[-1].tick_params(axis='both', which='minor', labelsize=7)
        
        if len(guides_by_mag_range[i]) > 0:
            guides_mean = np.mean(guides_by_mag_range[i])
            guides_median = np.median(guides_by_mag_range[i])
        else:
            guides_mean = 0
            guides_median = 0
        
        ax6[-1].text(ax6[-1].get_xlim()[1]/2, ax6[-1].get_ylim()[1]/2,
                     "Mean: %i, Median: %i" % (guides_mean, guides_median), 
                                               ha="center") 
    
        # ---------------------------------------------------------------------
        # Number of sky per tile
        # ---------------------------------------------------------------------
        # Plot a histogram of the number of sky per tile
        ax7.append(fig.add_subplot(gs[i,5]))
        ax7[-1].hist(sky_by_mag_range[i], bins=max(sky_per_tile), 
                     color=colour, align='right', label=label)
        ax7[-1].vlines(run_settings["SKY_PER_TILE"], ax6[-1].get_ylim()[0], 
                       ax6[-1].get_ylim()[1], linestyles='dashed', colors='k', 
                       label='Ideally-filled tile')
        ax7[-1].vlines(run_settings["SKY_PER_TILE_MIN"], 
                       ax6[-1].get_ylim()[0], ax6[-1].get_ylim()[1],
                       linestyles='dotted', colors='k', 
                       label='Minimum sky per tile')
        ax7[-1].set_xlabel('No. of sky per tile')
        ax7[-1].set_ylabel('Frequency')
        ax7[-1].legend(loc='upper center')
        ax7[-1].set_xlim(0, max(sky_per_tile) + 1)
        ax7[-1].xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        if len(sky_by_mag_range[i]) > 0:
            sky_mean = np.mean(sky_by_mag_range[i])
            sky_median = np.median(sky_by_mag_range[i])
        else:
            sky_mean = 0
            sky_median = 0
        
        ax7[-1].text(ax6[-1].get_xlim()[1]/2, ax6[-1].get_ylim()[1]/2,
                     "Mean: %i, Median: %i" % (sky_mean, sky_median), 
                                               ha="center") 

    # Set plot titles
    ax3.set_title("Run Settings & Overview", y=0.96)
    ax4[0].set_title("Stars per Tile")
    ax5[0].set_title("Standards per Tile")
    ax6[0].set_title("Guides per Tile")
    ax7[0].set_title("Sky per Tile")

    # Save plot
    name = "results/" + run_settings["run_id"] + "_tiling_run_overview.pdf"
    fig.savefig(name, fmt='pdf')
    
    
def create_tiling_visualisation(tiling, run_settings, increment=1000):
    """
    For increasingly larger slices of the tiling set, run the plotting code.
    
    To create a video from these:
    ffmpeg -framerate 4 -pattern_type glob -i "*.png" -c:v libx264 
        -pix_fmt yuv420p xx.mp4
    
    Parameters
    ----------
    tiling: list
        The list of TaipanTiles from a tiling run.
    run_settings: OrderedDict
        OrderedDict containing input settings and results from the tiling run.
    increment: int
        The number of new tiles to include in each plot/frame (i.e. 
        increment=1000 means that each iteration of the loop will create a plot
        with 1000 more tiles than the last).
    """
    # Create the list of tile increments, ensuring the final number is the 
    # complete plot
    steps = list(np.arange(increment, len(tiling), increment))
    
    if steps[-1] != len(tiling):
        steps.append(len(tiling))
    
    # Generate a plot for each increment
    for frame, step in enumerate(steps):
        plot_tiling(tiling[:step], run_settings)
        
        name = "results/visualisation/%s_overview_f%04i.png" % (
                                                        run_settings["run_id"], 
                                                        frame)
        plt.savefig(name)