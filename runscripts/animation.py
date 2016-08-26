import numpy as np
from mayavi.api import Engine
from mayavi import mlab

"""
    Create an animation using mayavi, will create a series of images. use makemp4.py to 
    create a video
"""
# Create a new mayavi scene.

totalTime = 4  # total duration of animation in seconds
framesPerSecond = 24  # the sampling rate of the images we make. Higher is smoother, max at 24
totalDegree = 360  # total degrees rotation, default clockwise
totalFrameCount = framesPerSecond * totalTime
degreePerFrame = totalDegree / totalFrameCount

threshold = np.load("./figECCB.npy")
skeleton = np.load("./figECCB_pruned.npy")
e = Engine()
e.start()
s = e.new_scene()
s.scene.background = (1, 1, 1)
g = mlab.contour3d(np.uint8(threshold), colormap='gray', contours=[0.7525])
g.actor.property.opacity = 0.3025
f = mlab.contour3d(np.uint8(skeleton), contours=[0.9901])
f.actor.property.representation = 'points'
f.actor.property.point_size = 6.448
mlab.options.offscreen = True
mlab.outline(f).actor.property.color = (0,0,0)
# Make an animation:
for i in range(totalFrameCount):
    # Rotate the camera by 10 degrees.
    s.scene.camera.azimuth(degreePerFrame)
    s.scene.camera.elevation(102.04192512233053)
    # Resets the camera clipping plane so everything fits and then
    # renders.
    s.scene.reset_zoom()
    # Save the scene. magnification=2 gives saves as an image when seen in fullscreen
    s.scene.magnification = 4
    s.scene.save("anim%d.png" % i)

# use imagemagick to create video from this frames
# convert -set delay 20 -loop 0 -quality 1000 -scale 100% *.png /home/pranathi/animExp.mpg
    # for coord in list(d.keys()):
    #     x = coord[0]; y = coord[1]; z = coord[2];
    #     mlab.text3d(x, y, z, d[coord], color=(0, 0, 0), scale=4.0)

# import Image

# background = Image.open("bg.png")
# overlay = Image.open("ol.jpg")

# # background = background.convert("RGBA")
# # overlay = overlay.convert("RGBA")

# new_img = Image.blend(background, overlay, 0.5)
# new_img.save("new.png", "PNG")

# fig, ax = plt.subplots(1)
# p = ax.pcolormesh(overlay)
# fig.colorbar(p)

