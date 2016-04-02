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

