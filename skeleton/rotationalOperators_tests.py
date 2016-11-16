import numpy as np

import skeleton.rotationalOperators as rotationalOperators

randArr = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                   [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)


def test_column():
    assert rotationalOperators.column(randArr, 0).sum() == 1
    assert rotationalOperators.column(randArr, 1).sum() == 2
    assert rotationalOperators.column(randArr, 2).sum() == 0


def test_flipLrInX():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipLrInX(randArr).sum() == randArr.sum()


def test_flipUdInY():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipUdInY(randArr).sum() == randArr.sum()


def test_flipFbInZ():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipFbInZ(randArr).sum() == randArr.sum()


def test_rot3D90():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.rot3D90(randArr).sum() == randArr.sum()


def test_iters():
    assert len(rotationalOperators.TRANSFORMATIONS_LIST) == 12
    for iterFunction in rotationalOperators.TRANSFORMATIONS_LIST:
        try:
            iterFunction(randArr[0:1])
        except AssertionError:
            print("error raised correctly")
        finally:
            assert sum(iterFunction(randArr)) + 1 == randArr.sum()


def test_directions():
    assert len(rotationalOperators.DIRECTIONS_LIST) == 12
