import numpy as np
from rotationalOperators import firstSubIter, secondSubIter, thirdSubIter, fourthSubIter, fifthSubIter, sixthSubIter
from rotationalOperators import seventhSubIter, eighthSubIter, ninthSubIter, tenthSubIter, eleventhSubIter, twelvethSubIter

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""


def generateLookuparray(stop, iterationNumber):
    """
       to generate a look up aray the all the 2 ** 26 possible configurations are
       checked if they satisy one of the 14 templates
       iterationNumber is the rotation of the cube into one of the directions
       refer to the paper Parallel 3D thinning algorithm with 12 directions for more details
    """
    lookuparray = np.zeros(stop, dtype=bool)
    print(iterationNumber)
    for item in range(0, stop):
        neighborValues = [(item >> digit) & 0x01 for digit in range(26)]
        neighborValues.insert(13, 1)
        neighborMatrix = np.reshape(neighborValues, (3, 3, 3))
        if np.sum(neighborValues) == 1:
            lookuparray[item] = 0
        else:
            neighborValues = iterationNumber(neighborMatrix)
            a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = tuple(neighborValues)
            shouldVoxelBedeleted = (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & p & (d | e | f | m | n | u | v | w | g | h | i | o | q | x | y | z)) | \
                                   (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & v & (r | s | t | j | k | l | m | n | u | w | o | p | q | x | y | z)) | \
                                   (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & y & (m | n | u | w | o | q | x | z)) | \
                                   (~(a) & ~(b) & ~(c) & ~(k) & ~(e) & ~(d & j) & ~ (l & f) & p & v) | \
                                   (~(a) & ~(b) & ~(k) & ~(e) & c & v & p & ~(j & d) & (l ^ f)) | \
                                   (a & v & p & ~(b) & ~(c) & ~(k) & ~(e) & ~(l & f) & (j ^ d)) | \
                                   (~(a) & ~(b) & ~(k) & ~(e) & n & v & p & ~(j & d)) | \
                                   (~(b) & ~(c) & ~(k) & ~(e) & m & v & p & ~(l & f)) | \
                                   (~(b) & ~(k) & ~(e) & a & n & v & p & (j ^ d)) | \
                                   (~(b) & ~(k) & ~(e) & c & m & v & p & (l ^ f)) | \
                                   (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(g) & ~(h) & q & y) | \
                                   (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(e) & ~(f) & ~(h) & ~(i) & o & y) | \
                                   (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(r) & ~(s) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & w & y) | \
                                   (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & ~(k) & ~(l) & ~(s) & ~(t) & u & y)
            lookuparray[item] = shouldVoxelBedeleted
    return lookuparray


directionList = [firstSubIter, secondSubIter, thirdSubIter, fourthSubIter,
                 fifthSubIter, sixthSubIter, seventhSubIter, eighthSubIter,
                 ninthSubIter, tenthSubIter, eleventhSubIter, twelvethSubIter]

if __name__ == '__main__':
    for i, item in enumerate(directionList):
        lookuparray = generateLookuparray(2 ** 26, item)
        np.save("lookuparray%i.npy" % (i + 1), lookuparray)

