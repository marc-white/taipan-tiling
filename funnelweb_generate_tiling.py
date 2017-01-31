#ipython --pylab
#i.e.
#run -i funnelweb_generate_tiling
# Save with:
# pickle.dump( (test_tiling, tiling_completeness, remaining_targets), open('tiling_big_email.pkl','w'))

# Test the various implementations of assign_fibre

#Test speed with: 
#kernprof -l funnelweb_generate_tiling.py
#python -m line_profiler script_to_profile.py.lprof

import taipan.core as tp
import taipan.tiling as tl
from astropy.table import Table
import matplotlib.patches as mpatches
# from mpl_toolkits.basemap import Basemap
import random
import sys
import datetime
import logging
import numpy as np
plotit = True
if plotit:
    import matplotlib.pyplot as plt
logging.basicConfig(filename='funnelweb_generate_tiling.log',level=logging.INFO)

#Change defaults. NB By changing in tp, we chance in tl.tp also.
tp.TARGET_PER_TILE = 139
tp.STANDARDS_PER_TILE = 4
tp.STANDARDS_PER_TILE_MIN = 3
#Enough to fit a linear trend and get a chi-squared uncertainty distribution with
#4 degrees of freedom, with 1 bad fiber.
tp.SKY_PER_TILE = 7
tp.SKY_PER_TILE_MIN = 7
tp.GUIDES_PER_TILE = 9
tp.GUIDES_PER_TILE_MIN = 3

#Limits for right-ascension and declination
ra_lims = [30,43]
de_lims = [-34,-25]
#Takes a *long* time, i.e. 5 mins. 1000 CPU-mins expected. Parallelization needed.
#ra_lims = [20,60] 
#de_lims = [-40,0]

#Range of magnitudes for the guide stars. Note that this range isn't allowed to be 
#completely within one of the mag_ranges below
guide_range=[8,10.5]
gal_lat_limit = 10 #For Gaia data only

#infile = '/Users/mireland/tel/funnelweb/2mass_AAA_i7-10_offplane.csv'; tabtype='2mass'
#mag_ranges_prioritise = [[7.5,8.5],[8.5,9.5]]
#mag_ranges = [[7.5,9],[8.5,10]]

### Change this below for your path!
infile = '/Users/mireland/Google Drive/funnelweb/FunnelWeb_Gaia_declt3.fits'; tabtype='gaia'

#Magnitude (Gaia) ranges for each exposure time.
mag_ranges = [[5,8],[7,10],[9,12],[11,14]]
#Magnitude ranges to prioritise within each range. We make sure that these are 
#mostly complete (up to completenes_target below)
mag_ranges_prioritise = [[5,7],[7,9],[9,11],[11,12]]

#mag_ranges_prioritise = [[7,9],[9,11]]
#mag_ranges = [[7,10],[9,12]]

#Method for tiling the sky. The following two parameter determine where the field 
#centers are.
tiling_method = 'SH'
tiling_set_size = 100

#Method for prioritising fibers 
alloc_method = 'combined_weighted'
combined_weight = 4.0 #Originally 1.0 - used for 'combined_weighted' fiber priority
sequential_ordering = (1,2) #Not used for combined_weighted. Just for 'sequential'

#Method for prioritising fields
ranking_method = 'priority-expsum'
exp_base = 3.0 #For priority-expsum, one fiber of priority k is worth exp_base fibers of priority k-1

#We move on to the next magnitude range after reaching this completeness on the priority
#targets
completeness_target = 0.99  #Originally 0.999

#Randomly choose this fraction of stars as standards. In practice, we will use colour cuts
#to choose B and A stars. An inverse standard frac of 10 will mean 1 in 10 stars are
#standards.
inverse_standard_frac = 10

#NB We have repick_after_complete set as False below - this can be done at the end.

#-------------- Automatic below here -----------------
try:
    if all_targets:
        pass
except NameError:
    print 'Importing test data...'
    start = datetime.datetime.now()
    #It is important not to duplicate targets, i.e. if a science targets is a guide and
    #a standard, then it should become a single instance of a TaipanTarget, appearing in
    #all lists.
    tabdata = Table.read(infile)
    print 'Generating targets...'
    if tabtype == '2mass':
        all_targets = [tp.TaipanTarget(str(r['mainid']), r['raj2000'], r['dej2000'], 
            priority=1, mag=r['imag'],difficulty=1) for r in tabdata 
            if ra_lims[0] < r['raj2000'] < ra_lims[1] and de_lims[0] < r['dej2000'] < de_lims[1]]
    elif tabtype == 'gaia':
        all_targets = [tp.TaipanTarget(str(r['source_id']), r['ra'], r['dec'], 
            priority=1, mag=r['phot_g_mean_mag'],difficulty=1) for r in tabdata 
            if ra_lims[0] < r['ra'] < ra_lims[1] and de_lims[0] < r['dec'] < de_lims[1] 
            and np.abs(r['b']) > gal_lat_limit]
    else: 
        raise UserWarning("Unknown table type")
    #Take standards from the science targets, and assign the standard property to True
    for t in all_targets:
        if (int(t.mag*100) % int(inverse_standard_frac))==0:
            t.standard=True
    #    if t.mag > 8:
    #        t.guide=True
    end = datetime.datetime.now()
    delta = end - start
    print ('Imported & generated %d targets'
        ' in %d:%02.1f') % (
        len(all_targets),
        delta.total_seconds()/60, 
        delta.total_seconds() % 60.)
    
    print 'Calculating target US positions...'
    #NB No need to double-up for standard targets or guide_targets as these are drawn from all_targets
    burn = [t.compute_usposn() for t in all_targets]

# tp.compute_target_difficulties(all_targets, ncpu=4)
# sys.exit()

# sys.exit()

# Ensure the objects are re type-cast as new instances of TaipanTarget (not needed?)
#for t in all_targets:
#    t.__class__ = tp.TaipanTarget

# Make a copy of all_targets list for use in assigning fibres
candidate_targets = all_targets[:]
random.shuffle(candidate_targets)

#Make sure that standards and guides are drawn from candidate_targets, not our master
#all_targets list.
standard_targets = [t for t in candidate_targets if t.standard==True]
guide_targets = [t for t in candidate_targets if guide_range[0]<t.mag<guide_range[1]]
print '{0:d} science, {1:d} standard and {2:d} guide targets'.format(
    len(candidate_targets), len(standard_targets), len(guide_targets))

print 'Commencing tiling...'
start = datetime.datetime.now()
test_tiling, tiling_completeness, remaining_targets = tl.generate_tiling_funnelweb(
    candidate_targets, standard_targets, guide_targets,
    mag_ranges_prioritise = mag_ranges_prioritise,
    prioritise_extra = 4,
    completeness_priority = 4,
    mag_ranges = mag_ranges,
    completeness_target=completeness_target,
    ranking_method=ranking_method,
    tiling_method=tiling_method, randomise_pa=True, 
    tiling_set_size=tiling_set_size,
    ra_min=ra_lims[0]-1, ra_max=ra_lims[1]+1, dec_min=de_lims[0]-1, dec_max=de_lims[1]+1,
    randomise_SH=True, tiling_file='ipack.3.4112.txt',
    tile_unpick_method=alloc_method, sequential_ordering=sequential_ordering,
    combined_weight=combined_weight,
    rank_supplements=False, repick_after_complete=False, exp_base=exp_base,
    recompute_difficulty=True, disqualify_below_min=True, nthreads=0)
end = datetime.datetime.now()

# Analysis
time_to_complete = (end - start).total_seconds()
non_standard_targets_per_tile = [t.count_assigned_targets_science(include_science_standards=False) for t in test_tiling]
targets_per_tile = [t.count_assigned_targets_science() for t in test_tiling]
standards_per_tile = [t.count_assigned_targets_standard() for t in test_tiling]
guides_per_tile = [t.count_assigned_targets_guide() for t in test_tiling]

print 'TILING STATS'
print '------------'
print 'Greedy/FunnelWeb tiling complete in %d:%2.1f' % (int(np.floor(time_to_complete/60.)),
    time_to_complete % 60.)
print '%d targets required %d tiles' % (len(all_targets), len(test_tiling), )
print 'Average %3.1f targets per tile' % np.average(targets_per_tile)
print '(min %d, max %d, median %d, std %2.1f' % (min(targets_per_tile),
    max(targets_per_tile), np.median(targets_per_tile), 
    np.std(targets_per_tile))

# Plot these results (requires ipython --pylab)
if plotit:
    
    plt.clf()
    fig = plt.gcf()
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
    ax3.hlines(len(all_targets), ax3.get_xlim()[0], ax3.get_xlim()[1], lw=.75,
        colors='k', linestyles='dashed', label='100% completion')
    ax3.hlines(0.975 * len(all_targets), ax3.get_xlim()[0], ax3.get_xlim()[1], 
        lw=.75,
        colors='k', linestyles='dashdot', label='97.5% completion')
    ax3.hlines(0.95 * len(all_targets), ax3.get_xlim()[0], ax3.get_xlim()[1], lw=.75,
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
    ax5.hist(guides_per_tile, bins=guide_range, align='right')
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
        ranking_method,len(all_targets),
        tiling_method, alloc_method, supp_str))
    try:
        plt.tight_layout()
    except:
        pass

    plt.show()
    plt.draw()

    fig.savefig('test_tiling_greedy-%s_%s_%s%s.pdf' % (ranking_method,
        tiling_method, alloc_method, supp_str),
        fmt='pdf')
