from mayavi import mlab
import numpy as np

# data = np.uint8(np.load('/media/pranathi/DATA/zoomedIndownsampledless.npy'))
data = np.uint8(np.load('/media/pranathi/DATA/maskDownsampled.npy'))
# mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

src = mlab.pipeline.scalar_field(data)
# Our data is equally spaced in all directions if not mention so
src.spacing = [1, 1, 1]
src.update_image_data = True


# Extract some inner structures: the ventricles and the inter-hemisphere
# fibers. We define a volume of interest (VOI) that restricts the
# iso-surfaces to the inner of the brain. We do this with the ExtractGrid
# filter.
blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi.set(x_min=1, x_max=3, y_min=1, y_max=3, z_min=1, z_max=3)

mlab.pipeline.iso_surface(voi, colormap='Spectral')

# Add two cut planes to show the raw MRI data. We use a threshold filter
# to remove cut the planes outside the brain.
thr = mlab.pipeline.threshold(src, low=50)
cut_plane = mlab.pipeline.scalar_cut_plane(thr,
                                           plane_orientation='y_axes',
                                           colormap='black-white',
                                           vmin=1400,
                                           vmax=2600)
cut_plane.implicit_plane.origin = (0, 0, 0)
cut_plane.implicit_plane.widget.enabled = False

cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
                                            plane_orientation='z_axes',
                                            colormap='black-white',
                                            vmin=1400,
                                            vmax=2600)
cut_plane2.implicit_plane.origin = (0, 0, 0)
cut_plane2.implicit_plane.widget.enabled = False

# Extract two views of the outside surface. We need to define VOIs in
# order to leave out a cut in the head.
voi2 = mlab.pipeline.extract_grid(src)
voi2.set(y_min=112)
outer = mlab.pipeline.iso_surface(voi2, contours=[1776, ],
                                  color=(0.8, 0.7, 0.6))

voi3 = mlab.pipeline.extract_grid(src)
voi3.set(y_max=112, z_max=53)
outer3 = mlab.pipeline.iso_surface(voi3, contours=[1776, ],
                                   color=(0.8, 0.7, 0.6))


mlab.view(-125, 54, 326, (145.5, 138, 66.5))
mlab.roll(-175)

mlab.show()

min = data.min()
max = data.max()
vol = mlab.pipeline.volume(data, vmin=min + 0.65 * (max - min),
                           vmax=min + 0.9 * (max - min))
