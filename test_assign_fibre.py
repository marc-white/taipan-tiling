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

try:
	if all_targets:
		pass
except NameError:
	print 'Importing test data...'
	tabdata = Table.read('TaipanCatalogues/southernstrip/'
		'SCOSxAllWISE.photometry.KiDS.fits')
	print 'Generating targets...'
	all_targets = [tp.TaipanTarget(str(r[0]), r[1], r[2], 
		priority=random.randint(1,8)) for r in tabdata ]
		# if r[1] > 20 and r[1] < 43 and r[2] > -34 and r[2] < -26]
	print 'Computing target difficulties...'
	no_targets = len(all_targets)
	print "Targets: %d" % no_targets
	start = datetime.datetime.now()
	# # Full time calculation
	# for i in range(no_targets):
	# 	all_targets[i].compute_difficulty(all_targets)
	# 	# if i % 100 == 0:
	# 	# 	print 'Done %d' % i
	# # [t.compute_difficulty(all_targets) for t in all_targets]
	# end = datetime.datetime.now()
	# delta = end - start
	# print 'Full time: %f' % (delta.total_seconds())

	# print 'Waiting...'
	# time.sleep(5)

	# # Approx calculation
	# print 'Computing target difficulties...'
	# start = datetime.datetime.now()
	# for i in range(no_targets):
	# 	all_targets[i].compute_difficulty_approx(all_targets)
	# 	# if i % 100 == 0:
	# 	# 	print 'Done %d' % i
	# # [t.compute_difficulty(all_targets) for t in all_targets]
	# end = datetime.datetime.now()
	# delta = end - start
	# print 'Approx time: %f' % (delta.total_seconds())

	# print 'Waiting...'
	# time.sleep(5)

	# # Mixed calculation
	# print 'Computing target difficulties...'
	# start = datetime.datetime.now()
	# for i in range(no_targets):
	# 	all_targets[i].compute_difficulty_mixed(all_targets, dec_cut=30.)
	# 	# if i % 100 == 0:
	# 	# 	print 'Done %d' % i
	# # [t.compute_difficulty(all_targets) for t in all_targets]
	# end = datetime.datetime.now()
	# delta = end - start
	# print 'Mixed time: %f' % (delta.total_seconds())

	# print 'Waiting...'
	# time.sleep(5)

	# KDTree calculation
	print 'Computing target difficulties...'
	start = datetime.datetime.now()
	tp.compute_target_difficulties(all_targets)
	end = datetime.datetime.now()
	delta = end - start
	print 'Mixed time: %d:%2.1f' % (delta.total_seconds()/60, 
		delta.total_seconds())

# del all_targets
# sys.exit()

# Ensure the objects are re type-cast as new instances of TaipanTarget
for t in all_targets:
	t.__class__ = tp.TaipanTarget

# Make a copy of all_targets list for use in assigning fibres
candidate_targets = all_targets[:]
random.shuffle(candidate_targets)

alloc_method = 'sequential'
sequential_ordering = (1,2,0)

clf()
fig = gcf()
fig.set_size_inches(18,9)
ax = fig.add_subplot(121)
# ax = Basemap(projection='gnom', lon_0=45.0, lat_0=-30.0)
ax.set_title(alloc_method)

test_tile_x = 43.5
test_tile_y = -31.0

test_tile = tp.TaipanTile(test_tile_x, test_tile_y)
ax.set_xlim(test_tile_x - 4., test_tile_x + 4.)
ax.set_ylim(test_tile_y - 4., test_tile_y + 4.)
candidate_targets = [t for t in candidate_targets 
	if t.dist_point((test_tile.ra, test_tile.dec)) < tp.TILE_RADIUS]
ax.plot([t.ra for t in candidate_targets], [t.dec for t in candidate_targets],
	marker='o', ms=1, mec='gray', mfc='gray', lw=0)
if alloc_method in ['combined_weighted', 'priority', 'sequential']:
	high_pris = [t for t in candidate_targets 
		if t.priority == tp.TARGET_PRIORITY_MAX]
	ax.plot([t.ra for t in high_pris], [t.dec for t in high_pris],
		marker='x', ms=7, mec='gray', mfc='gray', lw=0)

ax.plot([test_tile.ra], [test_tile.dec], 'kx', ms=12)
# tile_circ = mpatches.Circle((test_tile.ra, test_tile.dec),
# 	radius=tp.TILE_RADIUS / 3600., edgecolor='k', facecolor='none', lw=3)
tile_verts = np.asarray([tp.compute_offset_posn(test_tile.ra, test_tile.dec, tp.TILE_RADIUS, float(p)) for p in range(361)])
tile_circ = mpatches.Polygon(tile_verts, closed=False,
	edgecolor='k', facecolor='none', lw=3)
ax.add_patch(tile_circ)

for fibre in tp.BUGPOS_OFFSET:
	fibre_posn = test_tile.compute_fibre_posn(fibre)
	ax.plot(fibre_posn[0], fibre_posn[1], 'g+', ms=8)
	# ax.plot(test_tile.ra + tp.BUGPOS_ARCSEC[fibre][0]/3600.,
	# 	test_tile.dec + tp.BUGPOS_ARCSEC[fibre][1]/3600.,
	# 	'bx', ms=8)
	# fibre_circ = mpatches.Circle(fibre_posn, radius=tp.PATROL_RADIUS / 3600.,
	# 	edgecolor='b', facecolor='none', ls='dashed', lw=0.5)

	# fibre_targets = [t for t in candidate_targets 
	# 	if t.dist_point(fibre_posn) < tp.PATROL_RADIUS]
	# ax.plot([t.ra for t in fibre_targets], [t.dec for t in fibre_targets],
	# 	'ko', ms=0.6)

# Do a subsample of fibres as a demo
fibres = tp.BUGPOS_OFFSET.keys()
random.shuffle(fibres)
for fibre in fibres:
	print 'Assigning fibre %d' % (fibre, )
	fibre_posn = test_tile.compute_fibre_posn(fibre)
	ax.plot(fibre_posn[0], fibre_posn[1], 'r+', ms=10)
	fibre_verts = np.asarray([tp.compute_offset_posn(fibre_posn[0],
		fibre_posn[1], tp.PATROL_RADIUS, float(p)) for p in range(360)])
	fibre_circ = mpatches.Polygon(fibre_verts, closed=False,
		edgecolor='r', facecolor='none', lw=1.2, ls='dashed')
	# ax.add_patch(fibre_circ)

	# Assign a target
	# print len(test_tile.get_assigned_targets())
	candidate_targets, fibre_former_tgt = test_tile.assign_fibre(fibre, 
		candidate_targets,
		method=alloc_method, sequential_ordering=sequential_ordering,
		order_closest_secondary=False)
	# print len(candidate_targets)
	tgt = test_tile._fibres[fibre]
	if tgt is not None:
		# print tgt.priority
		ax.plot(tgt.ra, tgt.dec, marker='x', ms=20, mec='r', mfc='r', lw=0)
		ax.arrow(fibre_posn[0], fibre_posn[1], 
			tgt.ra-fibre_posn[0], tgt.dec - fibre_posn[1],
			fc='r', ec='r', head_width=0.03, head_length=0.1,
			length_includes_head=True)
		excl_verts = [tp.compute_offset_posn(tgt.ra, tgt.dec,
			tp.FIBRE_EXCLUSION_RADIUS, float(p)) for p in range(360)]
		excl_circ = mpatches.Polygon(np.asarray(excl_verts), closed=False,
			edgecolor='r', facecolor='none', lw=0.8, ls='dotted')
		ax.add_patch(excl_circ)

ax.set_aspect(1.)
show()
draw()

test_tile.repick_tile()

ax2 = fig.add_subplot(122)
ax2.set_xlim(test_tile_x - 4., test_tile_x + 4.)
ax2.set_ylim(test_tile_y - 4., test_tile_y + 4.)
tile_circ2 = mpatches.Polygon(tile_verts, closed=False,
	edgecolor='k', facecolor='none', lw=3)
ax2.add_patch(tile_circ2)
ax2.set_title('repicked')
for fibre in fibres:
	fibre_posn = test_tile.compute_fibre_posn(fibre)
	ax2.plot(fibre_posn[0], fibre_posn[1], 'r+', ms=10)
	fibre_verts = np.asarray([tp.compute_offset_posn(fibre_posn[0],
		fibre_posn[1], tp.PATROL_RADIUS, float(p)) for p in range(360)])
	fibre_circ = mpatches.Polygon(fibre_verts, closed=False,
		edgecolor='r', facecolor='none', lw=1.2, ls='dashed')
	# ax.add_patch(fibre_circ)
	tgt = test_tile._fibres[fibre]
	if tgt is not None:
		# print tgt.priority
		ax2.plot(tgt.ra, tgt.dec, marker='x', ms=20, mec='r', mfc='r', lw=0)
		ax2.arrow(fibre_posn[0], fibre_posn[1], 
			tgt.ra-fibre_posn[0], tgt.dec - fibre_posn[1],
			fc='r', ec='r', head_width=0.03, head_length=0.1,
			length_includes_head=True)
		excl_verts = [tp.compute_offset_posn(tgt.ra, tgt.dec,
			tp.FIBRE_EXCLUSION_RADIUS, float(p)) for p in range(360)]
		excl_circ = mpatches.Polygon(np.asarray(excl_verts), closed=False,
			edgecolor='r', facecolor='none', lw=0.8, ls='dotted')
		ax2.add_patch(excl_circ)

ax2.set_aspect(1)
show()
draw()

fig.savefig('unpick-%s.png' % (alloc_method), fmt='png', dpi=600)