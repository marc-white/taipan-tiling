#!ipython --pylab

# Test the generation of an even, full-coverage tiling system

import taipan.core as tp
import taipan.tiling as tl
import sys
import numpy as np

import matplotlib.patches as mpatches

ra_min = 60.
ra_max = 160.
dec_min = -60
dec_max = 20.

# Generate the tiling
print 'Making tiling...'
# tile_list = tl.generate_SH_tiling('ipack.3.402.txt',
tile_list = tl.generate_SH_tiling('ipack.3.8192.txt',
	randomise_seed=True,
	randomise_pa=True)
print 'Done!'

tile_list = [tile for tile in tile_list if tile.ra < 53.]
tile_list = [tile for tile in tile_list if tile.dec > -36. and tile.dec < -26.]
print len(tile_list)
# sys.exit()

clf()
fig = gcf()
ax = fig.add_subplot(111, projection='aitoff')
ax.grid(True)

print 'Commencing plotting...'
i = 1
for tile in tile_list:
	tile_verts = np.asarray([tp.compute_offset_posn(tile.ra, 
		tile.dec, tp.TILE_RADIUS, float(p)) for p in range(361)])
	tile_verts = np.asarray([tp.aitoff_plottable(xy, ra_offset=180.) for xy
		in tile_verts])
	ec = 'k'
	if i == 1:
		ec = 'r'
	tile_circ = mpatches.Polygon(tile_verts, closed=False,
		edgecolor=ec, facecolor='none', lw=.7)
	ax.add_patch(tile_circ)
	# ax.plot([np.radians(t.ra - 180.) for t in tile_list], [np.radians(t.dec) for t in tile_list],
	# 	'ko', lw=0, ms=1)
	i += 1

show()
draw()
# fig.savefig('tiling.png', fmt='png')