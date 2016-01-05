import numpy as np
import random
import skimage


def makeFakeVessels(imgsize=(2048, 1024), background=230):
    """
    create and save a matrix with whitish background and randomly selected vessel sizes and save matrices generated as images of format png
    """
    nVes = 9
    mu = 20
    sigma = 5
    minw = 5
    sx, sy = imgsize
    vasc = np.ones((sx, sy), dtype=np.uint8) * background

    for i in range(nVes):
        cx, cy = random.uniform(0, sx), random.uniform(0, sy)
        r1, r2 = 0, 0
        while (r1 < minw) or (r2 < minw):
            np.random.seed(20)
            r1 = np.random.normal(mu, sigma)
            r2 = np.random.normal(mu, sigma)
        print(r1, r2)

        rr, cc = skimage.draw.ellipse(cy, cx, r1, r2)
        if np.any(rr >= sy):
            ix = rr < sy
            rr, cc = rr[ix], cc[ix]
        if np.any(cc >= sx):
            ix = cc < sx
            rr, cc = rr[ix], cc[ix]
        vasc[rr, cc] = 1  # make circle blackish
    return vasc


def getPhantom(slices):
    vessels = _getCrosssection()
    phantom = np.zeros((slices, vessels.shape[0], vessels.shape[1]), dtype=np.uint8)
    for i in range(0, slices):
        phantom[i, :, :] = vessels
    return phantom


def _getCrosssection():
    s = np.array((512, 512))
    vessels = makeFakeVessels(s, background=0)
    return vessels


if __name__ == '__main__':
    from convOptimize import getSkeletonize3D
    from radiusOfNodes import getRadiusByPointsOnCenterline
    from unitwidthcurveskeleton import getShortestPathskeleton
    from radiusOfNodes import _getBouondariesOfimage
    from density import getRadisuStatistics
    phantom = getPhantom(424)
    np.save('/Users/3scan_editing/records/phantom.npy', phantom)
    phantomSkel = getSkeletonize3D(phantom)
    phantomShort = getShortestPathskeleton(phantomSkel)
    np.save('/Users/3scan_editing/records/phantomShort.npy', phantomShort)
    phantomBound = _getBouondariesOfimage(phantom)
    np.save('/Users/3scan_editing/records/phantomBound.npy', phantomBound)
    dict1, dist = getRadiusByPointsOnCenterline(phantomShort, phantomBound, phantom)
    getRadisuStatistics(dict1, dist)
