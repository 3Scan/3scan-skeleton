import numpy as np
import math


def F(n, d0):
    if n > 0:
        params = calculateBifurcation(d0)
        d0 = params['d1']
        N = n - 1
        return 'f' + ' [ ' + ' +(' + params['th1'] + ') ' + F(N, params['d1']) + ' ] ' + ' −(' + params['th2'] + ') ' + F(N, params['d2']) + ' ] '
    else:
        return 'F'


def calculateBifurcation(d0):
    params = {}
    params['d1'] = 0.7937 * d0
    params['d2'] = 0.7937 * d0
    params['th1'] = '45.47'
    params['th2'] = '45.47'
    return params


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    N-D Bresenham line algo
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def generatePointsOnskeleton(stringVar):
    from math import radians
    splitString = stringVar.split()
    pointsList = []
    origin = [1, 1, 1]; count = 0; currentState = []; pointsList = []
    pointsList.append(origin)
    currentState.append(origin)
    for branchFunction in splitString:
        print(branchFunction, count)
        if branchFunction == '[':
            print(currentState)
            # restore the past state and get them =(list of points in previous state)
            # ready for new branch
            # rotations
            if count != 0:
                currentState = pointsList[count - 1]
                print("after changing:", currentState, count)
        elif branchFunction == '+(45.47)' or '(-45.47)':
            stateList = []
            print(currentState)
            if len(currentState) == 1:
                for axis in currentState:
                    print("in for loop")
                    stateList.append(np.dot(rotation_matrix(axis, radians(360 - 45.47)), axis))
                    stateList.append(np.dot(rotation_matrix(axis, radians(-45.47)), axis))
            else:
                print("current state is origin")
                currentState = origin
                stateList.append(np.dot(rotation_matrix(axis, radians(360 - 45.47)), axis))
                stateList.append(np.dot(rotation_matrix(axis, radians(-45.47)), axis))
            print("statelist", currentState)
            print("stateList modified", stateList)
            currentState = stateList
            count += 1
            print("before modification", pointsList)
            pointsList.append(currentState)
            print("points list", pointsList)
        else:
            continue
            print(currentState)
    return pointsList
    # bresenham line algorithm
    # b-splineInterpolate the points list
    # set these points to 1 in an array


if __name__ == '__main__':
    F(1, 20)
