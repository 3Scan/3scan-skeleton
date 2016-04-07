import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

root = '/home/pranathi/subsubVolumestatNew_28/'
formatOfFiles = 'txt'
vol = 95 * 95 * 95
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if os.path.getsize(os.path.join(root, files)) != 26]
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if os.path.getsize(os.path.join(root, files)) != 0]
listOfNpys.sort()
dictPercetages = {}
dictLength = {}; dictTortuosity = {}
for f in listOfNpys:
    file_contents = open(f, 'r')
    fileC = [_ for _ in file_contents]
    npyIndex = f.replace(".txt", "")
    file_contents.close()
    strs = npyIndex.split('/')
    strs[4] = strs[4].replace('stat', '')
    strs = strs[4].split('_')
    strs = strs[1:]
    i, j, k = [int(s) for s in strs if s.isdigit()]
    dictPercetages[(i, j, k)] = float(fileC[0].replace('\n', ''))
    dictLength[(i, j, k)] = (float(fileC[3].replace('\n', '')) * 0.7) / vol
    dictTortuosity[(i, j, k)] = float(fileC[4].replace('\n', '')) / vol

dictList = [dictPercetages, dictLength, dictTortuosity]
imNames = ['percentVasc', 'length', 'tortuosity']
badKeys = [(240, 2694, 3037), (250, 635, 3307), (350, 9510, 7357), (340, 8587, 7357), (870, 9155, 3577), (240, 4753, 5737),
           (400, 10859, 7087), (250, 635, 2632), (230, 7309, 7357), (550, 7806, 7087), (260, 1913, 2902), (360, 3120, 5737),
           (580, 4185, 2632), (260, 16184, 2767), (260, 10930, 6007), (450, 8587, 3172), (270, 10362, 6007), (320, 10859, 2632),
           (470, 8587, 7087), (470, 8800, 7087), (380, 7593, 7087), (420, 8161, 7222), (350, 9013, 7222), (250, 564, 2767),
           (700, 8232, 2767), (300, 11498, 7222), (430, 7735, 6007), (340, 15332, 3982), (500, 4327, 2767), (350, 12208, 6682),
           (550, 8374, 3577), (260, 16042, 2767), (390, 9226, 7087), (280, 15261, 3307), (220, 8658, 7357), (310, 8658, 7357),
           (490, 8303, 3577), (300, 10362, 7357), (270, 12918, 3307), (880, 8232, 3712), (250, 15971, 2767), (560, 6457, 5872),
           (520, 10007, 3982), (240, 4682, 3982), (360, 8161, 7357), (600, 12563, 2902), (320, 12847, 3037), (250, 15119, 2632),
           (820, 6315, 3037), (880, 8161, 3712), (530, 7593, 5872), (620, 11214, 6277), (280, 1984, 3307), (310, 8303, 7357),
           (360, 9013, 7357), (240, 848, 3712), (350, 8374, 7357), (700, 9865, 2632), (400, 8871, 7087), (250, 15758, 2767),
           (340, 8871, 7357), (800, 8445, 5872), (350, 9013, 7357), (340, 9297, 7357)]
iskpx = 135; iskpz = 10; iskpy = 71
klist = [x for x in range(2632, 8026 - 68, iskpx) if (x > 2420 and x < 4000) or (x > 5500 and x < (7267 + 135))]
# dictSlice = {'saggital': 0, 'transverse': 2, 'coronal': 1}
for i in klist[:-1]:
    for index, dictStat in enumerate(dictList):
        dictStat = {key: value for key, value in dictStat.items() if key[2] == i and key not in badKeys}
        maxVal = max(list(dictStat.values()))
        print(maxVal, [key for key, value in dictStat.items() if value == maxVal])
        dictPercetagesFiltz = {((key[0] - 160) / iskpz, key[1] / iskpy): (255 * dictStat[key] / maxVal) for key, value in dictStat.items() if key not in badKeys}
        samplePoints = np.zeros((799 / iskpz, 17480 / iskpy), dtype=np.uint8)
        for coords, values in dictPercetagesFiltz.items():
            samplePoints[int(coords[0]), int(coords[1])] = values
        imsave("transverseSlice" + imNames[index] + "%i.png" % i, samplePoints)
        plt.imshow(samplePoints, cmap='winter')
        plt.savefig("transverseSlicecmap" + imNames[index] + "%i.png" % i)
