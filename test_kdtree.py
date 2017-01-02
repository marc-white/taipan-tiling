#ipython --pylab

# Test the various implementations of assign_fibre

import taipan.core as tp
import taipan.tiling as tl
from astropy.table import Table
import matplotlib.patches as mpatches
# from mpl_toolkits.basemap import Basemap
import random
import datetime
import sys
import time
from scipy.spatial import KDTree, cKDTree

try:
	if all_targets:
		pass
except:
	print 'Importing test data...'
	start = datetime.datetime.now()
	# tabdata = Table.read('TaipanCatalogues/southernstrip/'
	# 	'SCOSxAllWISE.photometry.KiDS.fits')
	tabdata = Table.read('TaipanCatalogues/wholehemisphere/'
		'Taipan.2MASS_selected.fits')
	print 'Generating targets...'
	# all_targets = [tp.TaipanTarget(str(r[0]), r[1], r[2], 
	all_targets = [tp.TaipanTarget(str(r[0]), r[4], r[5],
		priority=random.randint(1,8)) for r in tabdata ]
		# if r[1] > 40 and r[1] < 53 and r[2] > -34 and r[2] < -26]
	print 'Computing US position for all targets...'
	for target in all_targets:
		target.compute_usposn()
	no_targets = len(all_targets)
	end = datetime.datetime.now()
	delta = end - start
	print 'Imported & generated %d targets in %d:%02.1f' % (
		no_targets, delta.total_seconds()/60, 
		delta.total_seconds() % 60.)

# sys.exit()

print 'Computing target difficulties...'
start = datetime.datetime.now()
tp.compute_target_difficulties(all_targets, verbose=True)
end = datetime.datetime.now()
delta = end - start
print 'Time to compute %d difficulties: %d:%02.1f' % (
		no_targets, delta.total_seconds()/60, 
		delta.total_seconds() % 60.)
sys.exit()

# Test single-distance computations
# print 'Computing tile candidates via comprehension...'
# start = datetime.datetime.now()
# candidates = [t for t in all_targets if t.dist_point((40., -32.))
# 	< tp.TILE_RADIUS]
# end = datetime.datetime.now()
# delta = end - start
# print 'Time to compute candidates via comprehension: %d:%02.1f' % (
# 		delta.total_seconds()/60, 
# 		delta.total_seconds() % 60.)

# print 'Computing tile candidates via KDTree...'
# start = datetime.datetime.now()
# candidates = tp.targets_in_range(40., -32., all_targets,
# 	tp.TILE_RADIUS)
# end = datetime.datetime.now()
# delta = end - start
# print 'Time to compute candidates via KDTree: %d:%02.1f' % (
# 		delta.total_seconds()/60, 
# 		delta.total_seconds() % 60.)

# Difficulty speed comparison
# print 'tp.BREAKEVEN_KDTREE = %d' % tp.BREAKEVEN_KDTREE
# for n in [1e3, 2.5e3, 5e3, 1e4, 1e5, 2e5, 3e5]:
# 	n = int(n)
# 	print 'Difficulties via KDTree...'
# 	start = datetime.datetime.now()
# 	tp.compute_target_difficulties(all_targets[:n])
# 	end = datetime.datetime.now()
# 	delta = end - start
# 	print 'Computed %d target difficulties in %d:%02.1f' % (
# 		n, delta.total_seconds()/60, 
# 		delta.total_seconds() % 60.)

	# print 'Difficulties via recursion...'
	# start = datetime.datetime.now()
	# for t in all_targets[:n]:
	# 	t.difficulty = len([c for c in all_targets[:n]
	# 		if t.dist_point((c.ra, c.dec)) < tp.FIBRE_EXCLUSION_RADIUS])
	# end = datetime.datetime.now()
	# delta = end - start
	# print 'Computed %d target difficulties in %d:%02.1f' % (
	# 	n, delta.total_seconds()/60, 
		# delta.total_seconds() % 60.)