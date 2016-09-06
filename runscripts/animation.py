import numpy as np
import os

from mayavi.api import Engine
from mayavi import mlab

"""
Create an animation using mayavi, will create a series of images.
use KESMAnalysis.cli.makemp4.py to create a video or
use imagemagick to create video from this frames
convert -set delay 20 -loop 0 -quality 1000 -scale 100% *.png /home/pranathi/animExp.mpg
"""


def getFrames(pathThresh, pathSkel, totalTime, fps=24, totalRotation=360):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    pathThresh : str
        path of the .npy thresholded 3D Volume

    pathSKel : str
        path of the .npy skeleton 3D Volume

    totalTime : integer
        in seconds, duration of the video

    fps : integer
        frames per second, number of input frames per second

    totalRotation : integer
        angle in degrees frames should be captured in, integer between 0 and 360

    Returns
    -------
    frames of png images in the same directory as pathThresh
        mayavi scenes are saved as png images at different
        angle of rotations as anim%i.png i is the ith frame

    Notes
    -----
    threshold and skeletonized volume are overlapped,
    thresholded volume's isosurface is transparent and
    in grey and the skeletonized volume can be seen through
    it and is in red
    """
    # total frames
    totalFrameCount = fps * totalTime
    # degree of rotation after each frame
    degreePerFrame = totalRotation / totalFrameCount
    # load the threshold and skeleton paths
    threshold = np.load(pathThresh)
    skeleton = np.load(pathSkel)
    e = Engine()
    e.start()
    # Create a new mayavi scene.
    s = e.new_scene()
    s.scene.background = (1, 1, 1)
    # thresholded image in transparent grey
    g = mlab.contour3d(np.uint8(threshold), colormap='gray', contours=[0.7525])
    g.actor.property.opacity = 0.3025
    # skeleton in red
    f = mlab.contour3d(np.uint8(skeleton), contours=[0.9901])
    f.actor.property.representation = 'points'
    f.actor.property.point_size = 6.448
    mlab.options.offscreen = True
    mlab.outline(f).actor.property.color = (0, 0, 0)
    # extract rootDir of pathThresh
    rootDir = ""
    separator = os.sep
    directories = pathThresh.split(separator)[1:-1]
    for directory in directories:
        rootDir = rootDir + separator + directory
    rootDir = rootDir + separator
    # Make an animation:
    for i in range(totalFrameCount):
        # Rotate the camera by 10 degrees.
        s.scene.camera.azimuth(degreePerFrame)
        s.scene.camera.elevation(102.04192512233053)
        # Resets the camera clipping plane so everything fits and then
        # renders.
        s.scene.reset_zoom()
        # Save the scene. magnification=4 gives saves as an image when seen in fullscreen
        s.scene.magnification = 4
        s.scene.save(rootDir + "anim%d.png" % i)
