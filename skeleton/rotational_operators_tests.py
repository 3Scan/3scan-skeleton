import nose.tools

import numpy as np

import skeleton.rotational_operators as rotational_operators

RAND_ARR = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)


def test_column():
    # test if column returned by the function _column is as expected
    expected_summations = [1, 2, 0]
    for index, summation in enumerate(expected_summations):
        nose.tools.assert_equal(rotational_operators._column(RAND_ARR, index).sum(), summation)


def test_rotate_3D_90():
    # test if rot_3D_90 raises assertion error correctly
    try:
        rotational_operators.rot_3D_90(RAND_ARR[0:1])
    except AssertionError:
        print("error raised correctly")
    expected_sum = RAND_ARR.sum()
    obtained_sum = rotational_operators.rot_3D_90(RAND_ARR).sum()
    nose.tools.assert_equal(expected_sum, obtained_sum)


def test_get_directions_list():
    # test if the border point after is rotated to the expected direction
    test_array = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    directions_list = rotational_operators.get_directions_list(test_array)
    # expected index where 1 occurs after one of the rotation in 12 directions
    expected_results = [1, 23, 15, 5, 9, 25, 3, 19, 17, 21, 11, 7]
    for expected_result, direction in zip(expected_results, directions_list):
        nose.tools.assert_true(direction.reshape(27).tolist()[expected_result])
        nose.tools.assert_true(direction.sum())
