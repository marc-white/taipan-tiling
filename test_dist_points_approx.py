#!ipython --pylab
import numpy as np
import taipan.core as tp

# Gauge the accuracy of the dist_points_approx function

# Generate a grid of RA, Dec points
points_list = []
for i in np.arange(360):
	points_list.append([])
	for j in np.arange(180.):
		points_list[i].append((i, j - 90.))

# Compute displacements of FIBRE_EXCLUSION_RADIUS
no_disp = 8
delta_disp = 360. / float(no_disp)
points_disps = np.zeros((len(points_list), len(points_list[0]), no_disp))
points_disps = points_disps.tolist()

for i in range(len(points_list)):
	for j in range(len(points_list[0])):
		for k in range(no_disp):
			# Compute the positions
			points_disps[i][j][k] = tp.compute_offset_posn(
				points_list[i][j][0],
				points_list[i][j][1],
				tp.FIBRE_EXCLUSION_RADIUS,
				k * delta_disp)

clf()
fig = gcf()
fig.set_size_inches(12,12)

ax1 = fig.add_subplot(221)
ax1.set_title('Average of all offsets')

dists_full = np.zeros((len(points_list), len(points_list[0])))
dists_approx = np.zeros((len(points_list), len(points_list[0])))

for i in range(len(points_list)):
	for j in range(len(points_list[0])):
		dists_full[i, j] = np.average([tp.dist_points(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in points_disps[i][j]])
		dists_approx[i, j] = np.average([tp.dist_points_approx(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in points_disps[i][j]])
	print 'Done RA %d' % (i)

dists_relerrs = np.abs(dists_approx - dists_full) * 100. / dists_full
# Transpose for plotting
dists_relerrs = np.transpose(dists_relerrs)

xpts = np.arange(361.) - 0.5
ypts = np.arange(181.) - 90. - 0.5
X, Y = np.meshgrid(xpts, ypts)

dataplot = ax1.pcolor(X, Y, dists_relerrs,
	cmap=matplotlib.cm.get_cmap('jet'),
	vmin=0., vmax=100.0)
ax1.contour(xpts[1:]-0.5, ypts[1:]-0.5, dists_relerrs,
	levels=[1.0, 2.0, 5.0, 10.0, 50.0],
	colors='w')

fig.colorbar(dataplot, label='Relative error (%)')

ax1.set_xlim(0., 360.)
ax1.set_ylim(-90., 90.)
ax1.set_xlabel('RA')
ax1.set_ylabel('Dec')

show()
draw()

# dec displacements
ax2 = fig.add_subplot(222)
ax2.set_title('Average of dec offsets')

dists_full = np.zeros((len(points_list), len(points_list[0])))
dists_approx = np.zeros((len(points_list), len(points_list[0])))

for i in range(len(points_list)):
	for j in range(len(points_list[0])):
		dists_full[i, j] = np.average([tp.dist_points(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][0], points_disps[i][j][4]]])
		dists_approx[i, j] = np.average([tp.dist_points_approx(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][0], points_disps[i][j][4]]])
	print 'Done RA %d' % (i)

dists_relerrs = np.abs(dists_approx - dists_full) * 100. / dists_full
# Transpose for plotting
dists_relerrs = np.transpose(dists_relerrs)

ax2.set_xlim(0., 360.)
ax2.set_ylim(-90., 90.)
ax2.set_xlabel('RA')
ax2.set_ylabel('Dec')

dataplot = ax2.pcolor(X, Y, dists_relerrs,
	cmap=matplotlib.cm.get_cmap('jet'),
	vmin=0., vmax=100.0)
ax2.contour(xpts[1:]-0.5, ypts[1:]-0.5, dists_relerrs,
	levels=[1.0, 2.0, 5.0, 10.0, 50.0],
	colors='w')

show()
draw()


# RA displacements
ax3 = fig.add_subplot(223)
ax3.set_title('Average of RA offsets')

dists_full = np.zeros((len(points_list), len(points_list[0])))
dists_approx = np.zeros((len(points_list), len(points_list[0])))

for i in range(len(points_list)):
	for j in range(len(points_list[0])):
		dists_full[i, j] = np.average([tp.dist_points(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][2], points_disps[i][j][6]]])
		dists_approx[i, j] = np.average([tp.dist_points_approx(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][2], points_disps[i][j][6]]])
	print 'Done RA %d' % (i)

dists_relerrs = np.abs(dists_approx - dists_full) * 100. / dists_full
# Transpose for plotting
dists_relerrs = np.transpose(dists_relerrs)

ax3.set_xlim(0., 360.)
ax3.set_ylim(-90., 90.)
ax3.set_xlabel('RA')
ax3.set_ylabel('Dec')

dataplot = ax3.pcolor(X, Y, dists_relerrs,
	cmap=matplotlib.cm.get_cmap('jet'),
	vmin=0., vmax=100.0)
ax3.contour(xpts[1:]-0.5, ypts[1:]-0.5, dists_relerrs,
	levels=[1.0, 2.0, 5.0, 10.0, 50.0],
	colors='w')

show()
draw()


# Diagonal displacements
ax4 = fig.add_subplot(224)
ax4.set_title('Average of "diagonal" offsets')

dists_full = np.zeros((len(points_list), len(points_list[0])))
dists_approx = np.zeros((len(points_list), len(points_list[0])))

for i in range(len(points_list)):
	for j in range(len(points_list[0])):
		dists_full[i, j] = np.average([tp.dist_points(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][k] for k in [1,3,5,7]]])
		dists_approx[i, j] = np.average([tp.dist_points_approx(
			points_list[i][j][0], points_list[i][j][1],
			test_point[0], test_point[1]) for test_point
			in [points_disps[i][j][k] for k in [1,3,5,7]]])
	print 'Done RA %d' % (i)

dists_relerrs = np.abs(dists_approx - dists_full) * 100. / dists_full
# Transpose for plotting
dists_relerrs = np.transpose(dists_relerrs)

ax4.set_xlim(0., 360.)
ax4.set_ylim(-90., 90.)
ax4.set_xlabel('RA')
ax4.set_ylabel('Dec')

dataplot = ax4.pcolor(X, Y, dists_relerrs,
	cmap=matplotlib.cm.get_cmap('jet'),
	vmin=0., vmax=100.0)
ax4.contour(xpts[1:]-0.5, ypts[1:]-0.5, dists_relerrs,
	levels=[1.0, 2.0, 5.0, 10.0, 50.0],
	colors='w')

show()
draw()

fig.savefig('dist_points_approx_error.pdf', fmt='pdf')