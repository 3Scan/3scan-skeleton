import numpy as np

"""
   finding permutaions without using inbuilt function
   and getting proper permutations - the permutaions with
   even number of transitions
"""


def all_perms(elements):
    if len(elements) <= 1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]


def perm_parity2(a):
    """
    Using algorithm from http://stackoverflow.com/questions/337664/counting-inversions-in-an-array/6424847#6424847
    But substituting Pythons in-built TimSort
    """

    a = list(a)
    b = sorted(a)
    inversions = 0
    while a:
        first = a.pop(0)
        inversions += b.index(first)
        b.remove(first)
    return -1 if inversions % 2 else 1


# calculating proper permutations
connectElements = list(range(2, 27))
properPerms = []
thefile = open('perms.txt', 'w')
for firstPerm in all_perms(connectElements):
    p = perm_parity2(firstPerm)
    if p == 1:
        properPerms.append(firstPerm)
        thefile.write("%s\n\n\n" % firstPerm)
    else:
        continue
thefile.close()

print(len(properPerms))


def borderDecide(image):
    faces = [image[0][1][1], image[2][1][1], image[1][0][1], image[1][2][1], image[1][1][0], image[1][1][2]]
    if sum(faces) in list(range(1, 7)):
        border = True
    else:
        border = False
    return border


def edge(image):
    z, m, n = np.shape(image)
    paddedShape = z, m, n
    borderIm = np.ones((paddedShape), dtype=np.uint8)
    a = image.copy()
    for k in range(1, z - 1):
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if a[k, i, j] != 1:
                    continue
                else:
                    validateMatrix = a[k - 1: k + 2, j - 1: j + 2, i - 1: i + 2]
                    borderOrNot = borderDecide(validateMatrix);
                    if borderOrNot == 1:
                        borderIm[k, i, j] = 2
    return borderIm


# from enum import Enum


# class Directions(Enum):
#     usw = 2
#     us = 3
#     usea = 4
#     uw = 5
#     u = 6
#     ue = 7
#     unw = 8
#     un = 9
#     une = 10
#     sw = 11
#     n = 12
#     es = 13
#     w = 14
#     o = 1
#     e = 15
#     nw = 16
#     s = 17
#     ne = 18
#     sdw = 19
#     sd = 20
#     sde = 21
#     wd = 22
#     d = 23
#     ed = 24
#     ndw = 25
#     nd = 26
#     nde = 27
# referenceArray = np.array([[[Directions.usw.name, Directions.us.name, Directions.usea.name],
#                           [Directions.sw.name, Directions.n.name, Directions.es.name], [Directions.sdw.name, Directions.sd.name, Directions.sde.name]],
#                           [[Directions.uw.name, Directions.u.name, Directions.ue.name], [Directions.w.name, Directions.o.name, Directions.e.name],
#                           [Directions.wd.name, Directions.d.name, Directions.ed.name]], [[Directions.unw.name, Directions.un.name, Directions.une.name],
#                           [Directions.nw.name, Directions.s.name, Directions.ne.name], [Directions.ndw.name, Directions.nd.name, Directions.nde.name]]])
#  a by default cubic array

