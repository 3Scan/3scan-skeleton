import numpy as np
from mayavi.api import Engine
from mayavi import mlab

"""
Create an animation using mayavi, will create a series of images.
use KESMAnalysis.cli.makemp4.py to create a video or
use imagemagick to create video from this frames
convert -set delay 20 -loop 0 -quality 1000 -scale 100% *.png /home/pranathi/animExp.mpg
"""


def getFrames(path1, path2, totalTime, fps=24, totalRotation=360):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """

    totalFrameCount = fps * totalTime
    degreePerFrame = totalRotation / totalFrameCount
    threshold = np.load(path1)
    skeleton = np.load(path2)
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
        s.scene.save("anim%d.png" % i)
