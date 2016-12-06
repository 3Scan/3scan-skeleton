import numpy as np


import kesm.projects.KESMAnalysis.skeleton.rotational_operators as rotational_operators

RAND_ARR = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)


def raise_assertion_error(call, arg):
    """ helper function to raise error and print if error is raised correctly"""
    try:
        return call(arg)
    except AssertionError:
        print("error raised correctly")


def test_column():
    expected_summations = [1, 2, 0]
    for index, summation in enumerate(expected_summations):
        assert rotational_operators._column(RAND_ARR, index).sum() == summation


def test_rotate3D():
    expected_sum = RAND_ARR.sum()
    raise_assertion_error(rotational_operators.rot_3D_90, RAND_ARR[0:1])
    obtained_sum = rotational_operators.rot_3D_90(RAND_ARR).sum()
    assert obtained_sum == expected_sum, "expeceted {}, obtained {}".format(expected_sum, obtained_sum)


def test_get_directions_list():
    test_array = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    directions_list = rotational_operators.get_directions_list(test_array)
    expected_results = [1, 23, 15, 5, 9, 25, 3, 19, 17, 21, 11, 7]
    for expected_result, direction in zip(expected_results, directions_list):
        assert direction.reshape(27).tolist()[expected_result]
        assert direction.sum() == 1
