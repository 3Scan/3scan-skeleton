import cv2
import numpy as np

"""
Test library to create 3D raster volume of vessels
default settings use a light background with dark vessels, mimicking
the output of india-ink perfused brightfield microscopy images

ex:
>>> bstack = vessel_tree()
>>> stack = realistic_filter(bstack)

Using mayavi is a useful way to visualize these test stacks
Install mayavi via:
    conda install -c menpo mayavi=4.5.0
This version works on python 3. Versions < 4.5 don't work on python 3.x
mayavi also requires that QT4 (pyqt) is installed. This may be in conflict
with matplotlib >=1.5.3, which started to use qt5

View the stack contours:
>>> import mayavi as mlab
>>> mlab.contour3d(stack, colormap='gray')


This library is the start of taking vectorized inputs and creating raster
representations of that data in a large block (possibly gigabytes) in memory.
Use caution in allocating large volumes, especially in larger data types.

Todo: import VTK as dependency, use to create primitives and create mesh
booleans over a ndarray raster volume

As of 2016.01.10, mayavi 4.4.x works in python 3, but only with pyqt4. Newer
matplotlib (>=1.5.3) defaults to pyqt5 and qt5. It is difficult to have an
environment that supports both dependencies.
mayavi requires a newer version of VTK as well.
See https://github.com/enthought/mayavi/issues/84 for more info
Still unclear which branch of VTK works on python 3. Often requires special builds
The 'menpo' anaconda package releases may work better than most
"""


def realistic_filter(stack,
                     foreground=40,
                     background=200,
                     edge_filter_sigma=7,
                     speckle_level=20,
                     random_seed=1):
    """
    Add "realistic" noise to the binary vessel representation
    Inverts the binary mask so that vessels are now dark and background is light,
    matching the expected output of india ink stained vessels

    Input parameters:
    stack : ndarray, binary mask, expecting ndim==3, may work for 4
    foreground : int, within stack dtype, value for foreground
    background : int, within stack dtype, value for background
    edge_filter_sigma : stdev of gaussian to blur the binmask edges
    speckle_level : range +/- to add pixel noise
    random_seed : int, used to seed the random number generator for consistent results
    """
    np.random.seed(random_seed)
    # squish the signal into more realistic bounds
    noise_stack = scale_binarymask_update(stack.copy(), foreground=foreground, background=background)
    # blur the edges of the vessels
    noise_stack = blur_volume(noise_stack, sigma=edge_filter_sigma)
    # add blurred uniform noise to volume
    noise_stack = add_speckle(noise_stack, level=speckle_level, sigma=5)
    # add gussian noise to volume
    noise_stack = add_gaussian_noise(noise_stack, sigma=10)
    return noise_stack.astype(np.uint8)


def scale_binarymask_update(stack, foreground=40, background=200):
    """
    rescale a binary mask (two classes, identified as min and max)
    to the given foreground and background values.
    Enforces that the input array consists of only 2 values
        eg (0, 255), or (0, 1), or (30, 2354).
    Inputs:
        stack : ndarray, two values present
        foreground: signal value, ie vessels, must be within dtype of stack
        background: background value, must be within dtype of stack

    Returns: stack updated in place with new value mapping
    """
    np.testing.assert_array_equal(len(np.unique(stack)), 2)
    # find the max and min before updating
    mmin = stack.min()
    mmax = stack.max()
    stack[stack == mmin] = background
    stack[stack == mmax] = foreground
    return stack


def blur_volume(stack, sigma=21, out=None):
    """
    smooth out the edges of the vessels in 2D
    Uses a 2D kernel to blur on each face, with size=`sigma`

    Inputs :
        stack, 3D array of dtype of np.uint8 or float
        sigma : standard deviation of the Gaussian kernel used, in both (x, y)
        out : None, return allocated array
              ndarray, same shape as stack, to save output to.
    Returns :
        blurred volume, allocated

    NOTE: blur_volume() only works on types restricted within cv2.GaussianBlur(),
        which appears to be undocumented.
        uint8 and float64 work, int16 or int32 does not.
    """
    np.testing.assert_array_equal(sigma % 2, 1)

    if out is None:
        out = np.empty_like(stack)
    else:
        np.testing.assert_array_equal(out.shape, stack.shape)
    for i in range(stack.shape[2]):
        out[:, :, i] = cv2.GaussianBlur(stack[:, :, i], (sigma, sigma), 0)
    return out


def add_gaussian_noise(stack, sigma=5):
    """
    adds gaussian noise to the input stack, with standard deviation `sigma`
    """
    gnoise = (sigma * np.random.randn(*stack.shape))  # returns as float64
    return (stack + gnoise).clip(0, 256).astype(np.uint8)


def add_speckle(stack, level=10, sigma=5):
    """
    Adds multiple layers of noise from uniform distribution
    1) create layer of uniform noise defined by level (+- value, within 255)
    2) Filter layer with gaussian kernel with sigma (std of gaussian kernel)
    3) add back to original stack, and return as uint8

    Inputs:
        stack : ndrray
        sigma : int, stdev of gaussian kernel used in first convolution; only odd values allowed
        out : if not None, puts output in this array, asserts that has same shape as stack

    Returns a new ndarray of the volume with "speckle" applied, same shape as stack
    Returns assertion error if input `stack` is not np.uint8

   """
    np.testing.assert_array_equal(sigma % 2, 1)

    noise = np.random.randint(0, level * 2, stack.shape, dtype=np.uint8)
    blur = blur_volume(noise, sigma=sigma)  # blur the first level

    # we need to boost the dtype to signed to avoid twos complement rollover in uint8
    # When .astype() is used, it reallocates the array in memory, so should be used sparingly
    # Unfortunately, blur_volume() only works on types restricted within cv2.GaussianBlur(),
    # which is undocumented. uint8 and float64 work, int16 or int32 do not.
    stack2 = stack.astype(np.int16) + blur - level  # shift back down to center on distribution

    return stack2.clip(0, 255).astype(np.uint8)


def _add_cylinder_update(stack, point1, point2, radius, color=255):
    """
    Given two 3-tuples with the start and end points, create a cylinder between
    these points with radius r.
    This method updates the stack in place.

    This is a sloppy way to create a cylinder. Should be able to take the
    two input points and find the ellipse axis lengths for a horizontal plane
    cut through the cylinder at a fixed angle
    """
    # Some math that will improve the quality of the planar intersection with the cylinder:
    # line goes through point (x0, y0, z0) and in direction of unit vector (u1, u2, u3)
    # point on line closest to (x, y, z) is
    # (X0,Y0,Z0) + ((X-X0)u1+(Y-Y0)u2+(Z-Z0)u3) (u1,u2,u3)
    # (x-x0)^2 + (y-y0)^2 = r^2

    if point1[2] > point2[2]:  # points need to have the first point have a lower Z
        point1, point2 = point2, point1  # swap
    z = np.arange(point1[2], point2[2])
    n = len(z)
    x, y = np.linspace(point1[0], point2[0], n, dtype=int), np.linspace(point1[1], point2[1], n, dtype=int)
    thickness = -1  # line thickness, -1 == filled circle
    for i, zz in enumerate(z):
        # need to pass in copy of image, for unknown arcane openCV reasons
        stack[:, :, zz] = cv2.circle(stack[:, :, zz].copy(), (x[i], y[i]), radius, color, thickness)


def add_cylinders_update(stack, cylinders, color=255):
    """
    Input:
        stack : ndarray, 3 dimensions of uint8.
        cylinders : list of tuples, as [(point1, point2, radius), ...]
                    where point1 = (px, py, pz), bounded by the shape of stack

    """
    for p1, p2, radius in cylinders:
        _add_cylinder_update(stack, p1, p2, radius, color=color)


def vessel_diagonal(cube_edge=512,
                    radius=30,
                    background=0,
                    foreground=1,
                    dtype=np.uint8):
    """
    Create a simple cylinder going from the origin to the far corner
    Inputs:
        cube_edge : int, the length of the cube along each edge
        radius : int, radius of the cylinder about the line
            from (0, 0, 0) to (cube_edge, cube_edge, cube_edge)
        background : int, should be within dtype, (default=0)
        foreground : int, vessel fill level, within dtype (default=1)
        dtype : data type to allocate
    Returns:
        ndarray of type dtype
    """
    if 2 * radius > cube_edge:
        raise ValueError("Given radius '{}' does not fit in cube edge length {}"
                         .format(radius, cube_edge))
    stack = np.ones((cube_edge, cube_edge, cube_edge), dtype=dtype) * background
    cylinder = [
        ((0, 0, 0), (cube_edge, cube_edge, cube_edge), radius)
    ]
    add_cylinders_update(stack, cylinder, color=foreground)
    return stack


def _scale_point(point, scale):
    """
    scales a point tuple (x0, y0, z0) by scale, a float
    """
    return tuple(int(p * scale) for p in point)


def _scale_cylinders(cylinders, scale, scale_radius=True):
    """
    load the list of tuples associated with cylinders, and scale them to edgelen
    """
    new_cylinders = []
    for p1, p2, r in cylinders:
        r = int(r * scale) if scale_radius else r
        new_cylinders.append((_scale_point(p1, scale),
                             _scale_point(p2, scale),
                             r))
    return new_cylinders


def vessel_loop(cube_edge=512, scale_radius=True):
    """
    Create a simple loop of vessels inside of cube (uint8), (cube_edge, cube_edge, cube_edge)

    Input:
    cube_edge : size of cube in voxels on one edge
    scale_radius : bool, scale radius away from biological scale, default True

    Returns 3D ndarray,(cube_edge, cube_edge, cube_edge) dtype = np.uint8
    """
    stack = np.zeros((cube_edge, cube_edge, cube_edge), dtype=np.uint8)

    cylinders = [
        ((5, 15, 0), (480, 400, 512), 35),
        ((154, 138, 160), (94, 380, 400), 20),
        ((122, 258, 282), (254, 454, 459), 17),
        ((280, 276, 470), (325, 276, 40), 17),
        ((234, 419, 429), (319, 272, 108), 17),
    ]
    if cube_edge != 512:
        cylinders = _scale_cylinders(cylinders, cube_edge / 512, scale_radius)
    add_cylinders_update(stack, cylinders, color=1)
    return stack


def vessel_tree(cube_edge=512, scale_radius=True):
    """
    Create a simple tree of vessels inside of cube_edge cube
    This graph looks rather bad, but it works.
    Currently has two unconnected sets of branching tubes (two disjoint graphs)

    Input:
    cube_edge : size of cube in voxels on one edge
    scale_radius : bool, scale radius away from biological scale, default True

    Returns 3D cube of (cube_edge, cube_edge, cube_edge) np.uint8, with cylinders
    """
    stack = np.zeros((cube_edge, cube_edge, cube_edge), dtype=np.uint8)

    cylinders = [
        ((5, 15, 0), (480, 400, 512), 35),
        ((154, 138, 160), (94, 380, 400), 20),
        ((122, 270, 292), (250, 400, 500), 15),
        ((243, 185, 241), (424, 190, 475), 12),
        ((359, 311, 368), (208, 303, 200), 12),
        ((318, 280, 345), (322, 459, 510), 18),
        ((105, 91, 512), (446, 200, 0), 15),
        ((86, 380, 50), (35, 30, 512), 20),
        ((69, 158, 325), (230, 100, 512), 15)
    ]
    if cube_edge != 512:
        cylinders = _scale_cylinders(cylinders, cube_edge / 512, scale_radius)
    add_cylinders_update(stack, cylinders, color=1)
    return stack


def binmask_accuracy(original_binary_mask, predicted_binary_mask):
    """
    test the similarity of the predicted mask (aka classification)
    to the original binary mask
    Inputs:
        original_binary_mask : ndarray, values should be 0 or 1
        predicted_binary_mask : ndarray, values 0 or 1

    Asserts that the inputs are both binary masks, only accepting values of 0 or 1

    Returns:
        Accuracy, defined as (true_positive + true_negative) / total_instance_count
            Is returned as fractional percentage

    NOTE: accuracy is a standard definition of classification error measurement
        from a confusion matrix.
        See https://en.wikipedia.org/wiki/Confusion_matrix for more info
    """
    np.testing.assert_array_equal(original_binary_mask.shape, predicted_binary_mask.shape)
    np.testing.assert_array_equal(tuple(np.unique(original_binary_mask)), (0, 1))
    np.testing.assert_array_equal(tuple(np.unique(predicted_binary_mask)), (0, 1))

    diff = original_binary_mask.astype(np.int16) - predicted_binary_mask

    # everything in diff that is a zero is a true positive or a true negative
    n_matches = (diff == 0).sum()
    return n_matches / original_binary_mask.size


def assert_binmask_similar(expected, observed, accuracy=0.95):
    similarity = binmask_accuracy(expected, observed)
    assert similarity >= accuracy, \
        "Binmask similarity {} < target accuracy {}".format(similarity, accuracy)
