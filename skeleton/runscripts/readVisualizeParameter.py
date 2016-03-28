import os
import numpy as np
from scipy.misc import imsave

root = '/media/pranathi/KINGSTON/RESULTS_1sttransverseslice/subsubVolumestat/'
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
    strs[2] = strs[2].replace('~', '')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    dictPercetages[(i, j, k)] = float(fileC[0].replace('\n', ''))
    dictLength[(i, j, k)] = float(fileC[2].replace('\n', ''))
    dictTortuosity[(i, j, k)] = float(fileC[3].replace('\n', ''))

dictList = [dictPercetages, dictLength, dictTortuosity]
imNames = ['percentVasc', 'length', 'tortuosity']
badKeys = [(260, 564, 2767), (260, 2055, 2767), (270, 16042, 2767)]
for index, dictStat in enumerate(dictList):
    maxVal = max(list(dictStat.values()))
    if index < 1:
        dictPercetagesFiltz = {((key[0] - 160), key[1] / 7): (255 * dictStat[key] / maxVal) for key, value in dictStat.items() if key not in badKeys}
        samplePoints = np.zeros((799, 17480 / 7), dtype=np.uint8)
        for coords, values in dictPercetagesFiltz.items():
            samplePoints[int(coords[0]) - 1: int(coords[0]) + 2, int(coords[1]) - 1: int(coords[1]) + 2] = values
    else:
        minVal = min(list(dictStat.values()))
        dictPercetagesFiltz = {((key[0] - 160), key[1] / 7): ((dictStat[key] - minVal) / (maxVal - minVal)) for key, value in dictStat.items() if key not in badKeys}
        samplePoints = np.zeros((799, 17480 / 7), dtype=np.uint8)
        maxVal = max(list(dictPercetagesFiltz.values()))
        for coords, values in dictPercetagesFiltz.items():
            samplePoints[int(coords[0]) - 1: int(coords[0]) + 2, int(coords[1]) - 1: int(coords[1]) + 2] = (255 * values / maxVal)
    imsave("transverseSlice" + imNames[index] + "%i.png" % 2767, samplePoints)

