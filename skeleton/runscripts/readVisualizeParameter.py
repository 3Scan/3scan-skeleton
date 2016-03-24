import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

root = '/media/pranathi/KINGSTON/subsubVolumestat/'
formatOfFiles = 'txt'
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfNpys.sort()
dictPercetages = {}
dictLength = {}; dictTortuosity = {}
for f in listOfNpys:
    file_contents = open(f, 'r')
    npyIndex = f.replace(".txt", "")
    fileC = [_ for _ in file_contents]
    file_contents.close()
    strs = npyIndex.split('/')[-1].split('_')
    strs[0] = strs[0].replace('stat', '')
    strs[2] = strs[2].replace('~', '')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    dictPercetages[(i, j, k)] = float(fileC[0].replace('\n', ''))
    # dictLength[(i, j, k)] = float(fileC[3].replace('\n', ''))
    # dictTortuosity[(i, j, k)] = float(fileC[4].replace('\n', ''))

maxVal = max(list(dictPercetages.values()))
ilist = list(range(60, 799 - 10, 78))
# dictSlice = {'saggital': 0, 'transverse': 2, 'coronal': 1}
for i in ilist:
    dictPercetagesFiltz = {((key[1] / 280, key[2] / 280)): (255 * dictPercetages[key] / maxVal) for key, value in dictPercetages.items() if key[0] == i + 160}

    samplePoints = np.zeros((17480 / 280, 8026 / 280), dtype=np.uint8)
    for coords, values in dictPercetagesFiltz.items():
        samplePoints[int(coords[0]), int(coords[1])] = values
    imsave("saggitalSlicePercent%i.png" % (i + 160), samplePoints)
    # plt.imshow(samplePoints, cmap='gray')
    # plt.savefig("saggitalSlicePercent%i.png" % (i + 160), samplePoints)

for i in ilist:
    dictPercetagesFiltz = {((key[0] / 280, key[1] / 280)): (255 * dictPercetages[key] / maxVal) for key, value in dictPercetages.items() if key[2] == 3707}

    samplePoints = np.zeros((17480 / 280, 8026 / 280), dtype=np.uint8)
    for coords, values in dictPercetagesFiltz.items():
        samplePoints[int(coords[0]), int(coords[1])] = values
    plt.imshow(samplePoints, cmap='gray')
    plt.savefig("transverseSliceParam%i.png" % i)
