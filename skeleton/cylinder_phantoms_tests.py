import numpy as np

import vessels.cylinder_phantoms as cylinder_phantoms


def test_single_vessel():
    mask = cylinder_phantoms.vessel_diagonal(cube_edge=50, radius=10)
    np.testing.assert_array_equal(tuple(np.unique(mask)), (0, 1))
    noise_stack = cylinder_phantoms.realistic_filter(mask)
    np.testing.assert_array_equal(noise_stack.shape, (50, 50, 50))


def test_vessel_loop():
    cube_edge = 128
    stack = cylinder_phantoms.vessel_loop(cube_edge, scale_radius=False)
    np.testing.assert_array_equal(stack.shape, (cube_edge, cube_edge, cube_edge))
    np.testing.assert_array_equal(tuple(np.unique(stack)), (0, 1))


def test_vessel_tree():
    cube_edge = 128
    stack = cylinder_phantoms.vessel_tree(cube_edge)
    np.testing.assert_array_equal(stack.shape, (cube_edge, cube_edge, cube_edge))
    np.testing.assert_array_equal(tuple(np.unique(stack)), (0, 1))


def test_binmask_similarity():
    mask = cylinder_phantoms.vessel_diagonal(cube_edge=50, radius=5)

    # force errors in mask
    predicted_mask = mask.copy()
    predicted_mask[15:20, 35:40, 2:4] = 1  # add false positives
    predicted_mask[18:22, 18:22, 20] = 0  # add false negatives

    cylinder_phantoms.assert_binmask_similar(predicted_mask, mask, accuracy=.9990)


def test_blur_volume():
    stack = cylinder_phantoms.vessel_diagonal(cube_edge=20, radius=5)
    blur = cylinder_phantoms.blur_volume(stack, sigma=3)
    np.testing.assert_array_equal(blur.shape, stack.shape)

    # check the in-place operation
    cylinder_phantoms.blur_volume(stack, sigma=3, out=stack)
    np.testing.assert_array_equal(blur, stack)


def test_add_noise():
    stack = cylinder_phantoms.vessel_diagonal(cube_edge=20, radius=5)
    stack = cylinder_phantoms.scale_binarymask_update(stack)

    noisy = cylinder_phantoms.add_speckle(stack, level=15, sigma=5)
    np.testing.assert_array_equal(noisy.shape, stack.shape)

    gauss_noisy = cylinder_phantoms.add_gaussian_noise(stack, sigma=10)
    np.testing.assert_array_equal(gauss_noisy.shape, stack.shape)
