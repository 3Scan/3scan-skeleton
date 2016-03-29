import numpy as np
from mayavi.api import Engine
from mayavi import mlab
# Create a new mayavi scene.
threshold = np.load(input("enter a path to thresholded volume"))
skeleton = np.load(input("enter a path to skeleton you want to superimpose over"))
e = Engine()
e.start()
s = e.new_scene()
mlab.contour3d(np.uint8(threshold), contours=[108.50]).actor.property.representation = 'points'
mlab.contour3d(np.uint8(skeleton), colormap='gray')
mlab.options.offscreen = True
# Make an animation:
for i in range(36):
    # Rotate the camera by 10 degrees.
    s.scene.camera.azimuth(10)
    # Resets the camera clipping plane so everything fits and then
    # renders.
    s.scene.reset_zoom()
    # Save the scene. magnification=2 gives saves as an image when seen in fullscreen
    mlab.savefig("anim%d.png" % i, magnification=2)

# use imagemagick to create video from this frames
# convert -set delay 20 -loop 0 -quality 1000 -scale 100% *.png /home/pranathi/animExp.mpg
    # for coord in list(d.keys()):
    #     x = coord[0]; y = coord[1]; z = coord[2];
    #     mlab.text3d(x, y, z, d[coord], color=(0, 0, 0), scale=4.0)
# def _getSort(cycles):
#     """
#        sort one or two or three dimensional coordinates from least
#        first coordinate, second coordinate, third coordinate
#        to highest only if it is not sorted
#     """
#     nodeDim = len(cycles[0])
#     if nodeDim == 3:
#         ilist = [0, 2, 1]
#     else:
#         ilist = [1, 0]
#     cycles.sort(key=lambda x: [x[i] for i in ilist], reverse=True)
#     return cycles

