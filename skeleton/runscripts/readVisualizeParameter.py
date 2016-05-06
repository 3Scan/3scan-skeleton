import os
import numpy as np
# import matplotlib.pyplot as plt
from scipy.misc import imsave, imread
import pandas
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


root = '/media/pranathi/DATA/subsubVolumestatNew_28/'
vol = 95 * 95 * 95
formatOfFiles = 'txt'
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if os.path.getsize(os.path.join(root, files)) != 82]
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if os.path.getsize(os.path.join(root, files)) != 0]
listOfNpys.sort()
dictTortuosity2 = {}; dictTortuosity1 = {}
dictPercetages = {}; dictLength = {}
for f in listOfNpys:
    file_contents = open(f, 'r')
    fileC = [_ for _ in file_contents]
    npyIndex = f.replace(".txt", "")
    file_contents.close()
    strs = npyIndex.split('/')
    strs[5] = strs[5].replace('stat', '')
    strs = strs[5].split('_')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    dictPercetages[(i, j, k)] = float(fileC[0].replace('\n', ''))
    # dictLength[(i, j, k)] = (float(fileC[0].replace('\n', '')) * 0.7) / vol
    # dictTortuosity1[(i, j, k)] = float(fileC[1].replace('\n', ''))
    # dictTortuosity2[(i, j, k)] = float(fileC[2].replace('\n', ''))
# intlist = [int(x) for x in l if x.isdigit()]
dictList = [dictPercetages]
imNames = ['percentVasc']
# root = '/media/pranathi/KINGSTON/Pictures/Pictures_ts/'
# formatOfFiles = 'png'
# listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]

# for fileName in listOfNpys:
#     image = imread(fileName)
#     image = np.rot90(image, k=3)
#     imsave(fileName, image)
badKeys = [(240, 2694, 3037), (250, 635, 3307), (350, 9510, 7357), (340, 8587, 7357), (870, 9155, 3577), (240, 4753, 5737), (4100, 10859, 7087),
           (250, 635, 2632), (230, 7309, 7357), (550, 7806, 7087), (260, 1913, 2902), (360, 3120, 5737), (580, 4185, 2632), (260, 16184, 2767),
           (260, 10930, 6007), (450, 8587, 3172), (270, 10362, 6007), (320, 10859, 2632), (470, 8587, 7087), (470, 8800, 7087), (380, 7593, 7087),
           (420, 8161, 7222), (350, 9013, 7222), (250, 564, 2767), (700, 8232, 2767), (300, 11498, 7222), (430, 7735, 6007), (340, 15332, 3982),
           (500, 4327, 2767), (350, 12208, 6682), (550, 8374, 3577), (260, 16042, 2767), (390, 9226, 7087), (280, 15261, 3307), (220, 8658, 7357),
           (310, 8658, 7357), (490, 8303, 3577), (300, 10362, 7357), (270, 12918, 3307), (880, 8232, 3712), (250, 15971, 2767), (560, 6457, 5872),
           (520, 10007, 3982), (240, 4682, 3982), (360, 8161, 7357), (600, 12563, 2902), (320, 12847, 3037), (250, 15119, 2632), (820, 6315, 3037),
           (880, 8161, 3712), (530, 7593, 5872), (620, 11214, 6277), (280, 1984, 3307), (310, 8303, 7357), (360, 9013, 7357), (240, 848, 3712),
           (350, 8374, 7357), (700, 9865, 2632), (400, 8871, 7087), (250, 15758, 2767), (340, 8871, 7357), (800, 8445, 5872), (350, 9013, 7357),
           (340, 9297, 7357), (250, 15190, 2632), (250, 15261, 2632), (250, 15332, 2632), (250, 15403, 2632), (250, 15474, 2632), (250, 15545, 2632),
           (250, 15687, 2632), (250, 15758, 2632), (250, 15829, 2632), (250, 15971, 2632), (250, 16042, 2632), (270, 14551, 2632), (550, 6528, 2632),
           (770, 3546, 2632), (770, 3617, 2632), (770, 3688, 2632), (770, 3759, 2632), (770, 3830, 2632), (770, 3901, 2632), (770, 3972, 2632),
           (550, 1771, 2632), (550, 1842, 2632), (550, 1913, 2632), (550, 1984, 2632), (550, 2055, 2632), (550, 2126, 2632), (550, 2197, 2632),
           (550, 2268, 2632), (550, 2339, 2632), (550, 2410, 2632), (470, 1203, 2632), (470, 1274, 2632), (470, 1345, 2632), (250, 564, 2632),
           (250, 777, 2632), (240, 848, 2632), (250, 14835, 2767), (250, 14906, 2767), (250, 14977, 2767), (250, 15048, 2767), (250, 15119, 2767),
           (250, 15616, 2767), (250, 15900, 2767), (250, 16113, 2767), (250, 16184, 2767), (260, 15687, 2767), (270, 14338, 2767), (270, 14409, 2767),
           (270, 14480, 2767), (270, 14551, 2767), (270, 14622, 2767), (270, 14693, 2767), (270, 14764, 2767), (890, 6883, 2767), (800, 4398, 2767),
           (520, 8374, 2767), (800, 4398, 2767), (550, 1700, 2767), (550, 1771, 2767), (550, 1842, 2767), (550, 1913, 2767), (550, 1984, 2767),
           (550, 2126, 2767), (470, 1061, 2767), (470, 1132, 2767), (470, 1203, 2767), (400, 1416, 2767), (250, 564, 2767), (260, 564, 2767),
           (250, 635, 2767), (250, 706, 2767), (370, 1061, 2767), (370, 1345, 2767), (380, 1132, 2767), (370, 1132, 2767), (370, 1203, 2767),
           (550, 2055, 2767), (370, 1274, 2767), (360, 919, 2767), (240, 990, 2767), (240, 1061, 2767), (330, 11569, 2902), (470, 6528, 6952),
           (270, 4966, 6952), (490, 9723, 6952), (630, 9794, 6952), (330, 3120, 6952), (380, 2623, 6817), (380, 2694, 6817), (380, 2765, 6817),
           (630, 10646, 6817), (640, 10646, 6817), (260, 4895, 6817), (580, 10859, 6817), (580, 11143, 6682), (730, 7593, 6682), (730, 7735, 6682),
           (730, 7806, 6682), (730, 7877, 6682), (730, 7948, 6682), (260, 1842, 6547), (380, 2268, 6547), (430, 2481, 6547), (430, 2552, 6547),
           (440, 4540, 6547), (300, 3262, 6547), (300, 3191, 6412), (430, 2268, 6412), (430, 2339, 6412), (380, 2055, 6412), (320, 1913, 6412),
           (440, 4682, 6412), (440, 5250, 6412), (720, 6173, 6412), (280, 4611, 6277), (490, 5605, 6277), (500, 5605, 6277), (320, 1700, 6277),
           (380, 1913, 6277), (430, 2197, 6277), (480, 3191, 6277), (500, 4043, 6277), (770, 6386, 6277), (640, 12563, 6277), (640, 12776, 6142),
           (640, 12705, 6142), (630, 12705, 6142), (290, 4611, 6007), (560, 3333, 6007), (740, 5321, 6007), (810, 6599, 6007), (810, 6031, 5872),
           (320, 1345, 5872), (310, 15261, 5872), (310, 15190, 5872), (380, 1061, 5737), (630, 12989, 5737), (640, 12918, 5737), (310, 15474, 5737),
           (250, 10859, 5602), (380, 1061, 5602), (710, 12563, 3982), (850, 12492, 3982), (680, 2836, 3982), (440, 8942, 3982), (700, 7238, 3982),
           (590, 13202, 3847), (590, 13273, 3847), (630, 13131, 3847), (680, 12847, 3847), (920, 6457, 3847), (680, 2765, 3847), (440, 5818, 3847),
           (220, 3759, 3712), (240, 635, 3712), (680, 2907, 3712), (580, 5037, 3712), (600, 4327, 3712), (940, 7238, 3712), (940, 7309, 3712),
           (940, 7380, 3712), (940, 7451, 3712), (940, 7522, 3712), (940, 7593, 3712), (680, 12847, 3712), (680, 12776, 3712), (260, 1842, 3037),
           (260, 1913, 3037), (240, 635, 3037), (250, 635, 3037), (250, 706, 3037), (250, 777, 3037), (250, 848, 3037), (250, 919, 3037),
           (360, 919, 3037), (470, 1132, 3037), (550, 1629, 3037), (600, 2197, 3037), (430, 8303, 3037), (620, 8800, 3037), (610, 7096, 3037),
           (330, 11569, 2902), (250, 635, 2902), (250, 706, 2902), (240, 706, 2902), (220, 1274, 2902), (470, 1132, 2902), (470, 1203, 2902),
           (550, 1629, 2902), (550, 1700, 2902), (530, 2126, 2902), (490, 2055, 2902), (490, 2197, 2902), (710, 4469, 2902), (540, 4895, 2902),
           (430, 6031, 2902), (630, 12492, 2902), (630, 12563, 2902), (590, 12847, 2902), (590, 12918, 2902)]
lut = {}
lut[(255, 0, 0)] = ["red", "hypothalamus"]
lut[(255, 0, 128)] = ["pink", "medulla"]
lut[(128, 0, 0)] = ["brown", "olfactory"]
lut[(0, 255, 0)] = ["green", "cortex"]
lut[(128, 128, 0)] = ["moss", "thalamus"]
lut[(0, 0, 255)] = ["blue", "cerebralNuclei"]
lut[(0, 255, 255)] = ["sky", "midbrain"]
lut[(128, 0, 128)] = ["purple", "hippocampus"]
lut[(255, 255, 0)] = ["yellow", "cerebellum"]
lut[(255, 128, 0)] = ["orange", "pons"]

table = np.zeros((22, 10, 4))
table1 = np.zeros((22, 10, 4))
table2 = np.zeros((22, 10, 4))
iskpx = 135; iskpz = 10; iskpy = 71
klist = [x for x in range(2632, 8026 - 68, iskpx) if (x > 2420 and x < 4000) or (x > 5500 and x < (7267 + 135))]
# dictSlice = {'saggital': 0, 'transverse': 2, 'coronal': 1}
for kIndex, i in enumerate(klist[:-3]):
    for index, dictStat in enumerate(dictList):
        dictStat = {key: value for key, value in dictStat.items() if key not in badKeys}
        maxVal = max(list(dictStat.values()))
        keyMax = [key for key, value in dictStat.items() if value == maxVal]
        print(maxVal, keyMax)
        dictPercetagesFiltz = {((key[0] - 160) / iskpz, key[1] / iskpy): dictStat[key] for key, value in dictStat.items() if key[2] == i}
        samplePoints = np.zeros((799 / iskpz, 17480 / iskpy))
        for coords, values in dictPercetagesFiltz.items():
            samplePoints[int(coords[0]), int(coords[1])] = values
        colorAnnotate = imread("/media/pranathi/DATA/annotate/transverseSliceAnnotate%i.png" % i)
        t = colorAnnotate.any(axis=-1)
        t = samplePoints * t
        listNZI = list(set(map(tuple, list(np.transpose(np.array(np.where(t != 0)))))))
        count = 0
        for key, value in lut.items():
            region = [samplePoints[tuple((listNZI[index][0], listNZI[index][1]))] for index in range(0, len(listNZI)) if tuple(colorAnnotate[(listNZI[index][0], listNZI[index][1])]) == key]
            if len(region) != 0:
                avg = sum(region) / len(region)
                table[kIndex, count, 0] = sum(region)
                table1[kIndex, count, 0] = len(region)
                table2[kIndex, count, 0] = avg
            count += 1
            # f = open(value[1] + '.txt', 'a')
            # f.writelines(str(avg) + "\n")
            # f.close()
        # rgb = np.zeros(79, 246, 3)
        # rgb[:, :, 0] = samplePoints
        # rgb[:, :, 1] = samplePoints
        # rgb[:, :, 2] = samplePoints
        # imsave("transverseSlice" + imNames[index] + "%i.png" % i, samplePoints)

# order of array
# (0, 255, 255) ['sky', 'midbrain']
# (128, 0, 0) ['brown', 'olfactory']
# (128, 128, 0) ['moss', 'thalamus']
# (255, 0, 0) ['red', 'hypothalamus']
# (128, 0, 128) ['purple', 'hippocampus']
# (255, 255, 0) ['yellow', 'cerebellum']
# (0, 0, 255) ['blue', 'cerebralNuclei']
# (0, 255, 0) ['green', 'cortex']
# (255, 0, 128) ['pink', 'medulla']
# (255, 128, 0) ['orange', 'pons']
regionName = ['Midbrain', 'Olfactory Bulb', 'Thalamus', 'Hypothalamus', 'Hippocampus', 'Cerebellum', 'Cerebral Nuclei', 'Cortex', 'Medulla', 'Pons']
slices = [str(i) for i in klist[:-3]]
writer = pandas.ExcelWriter('output.xlsx', engine='xlsxwriter')
df1 = pandas.DataFrame(table[:, :, 0], slices, regionName)
df1.to_excel(writer, 'Sheet1')
df2 = pandas.DataFrame(table[:, :, 1], slices, regionName)
df2.to_excel(writer, 'Sheet2')
df3 = pandas.DataFrame(table[:, :, 2], slices, regionName)
df3.to_excel(writer, 'Sheet3')
df4 = pandas.DataFrame(table[:, :, 3], slices, regionName)
df4.to_excel(writer, 'Sheet4')
writer.save()
colors = [(0, 255, 255), (128, 0, 0), (128, 128, 0), (255, 0, 0), (128, 0, 128), (255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 128), (255, 128, 0)]
colors = [(0, 1, 1), (0.5, 0, 0), (0.5, 0.5, 0), (1, 0, 0), (0.5, 0, 0.5), (1, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0.5), (1, 0.5, 0)]
colorsarr = np.zeros((10, 3))
for xdim in range(0, colorsarr.shape[0]):
    colorsarr[xdim, :] = colors[xdim]
regionName = ['Midbrain', 'Olfactory Bulb', 'Thalamus', 'Hypothalamus', 'Hippocampus', 'Cerebellum', 'Cerebral Nuclei', 'Cortex', 'Medulla', 'Pons']
x = []; y = []; z = []
rnames = []; correctColors = []
for index in range(0, 22):
    xlist = [i for i in table[index, :, 0].tolist() if i != 0.0]
    ylist = [i for i in table[index, :, 1].tolist() if i != 0.0]
    zlist = [i for i in table[index, :, 3].tolist() if i != 0.0]
    rlist = [name for name, i in zip(regionName, table[index, :, 3].tolist()) if i != 0.0]
    clist = [color for color, i in zip(colors, table[index, :, 3].tolist()) if i != 0.0]
    for xitem, yitem, zitem, citem, ritem in zip(xlist, ylist, zlist, clist, rlist):
        x.append(xitem)
        y.append(yitem)
        z.append(zitem)
        correctColors.append(citem)
        rnames.append(ritem)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colorsarr, depthshade=False)
ax.xaxis.set_label_text("Percentage microasculature")
ax.yaxis.set_label_text("Total length per volume")
ax.zaxis.set_label_text("Average voxelized tortuosity")
recs = []
colorList = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0.5, 0, 0.5], [1, 1, 0], [1, 0.5, 0], [0.5, 0.5, 1], [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0, 0]]
regionName = ["First TP", "Second TP", "Third TP", "Fourth TP", "Fifth TP", "Sixth TP", "Seventh TP", "Eighth TP", "Ninth TP", "Tenth TP", "Eleventh TP"]
for i in range(0, len(colorList)):
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colorList[i]))
ax.legend(recs, regionName, loc='upper left', fontsize='small')
fig.tight_layout()
fig.savefig("statVtotal.png")

colors = [(0, 1, 1), (0.5, 0, 0), (0.5, 0.5, 0), (1, 0, 0), (0.5, 0, 0.5), (1, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0.5), (1, 0.5, 0)]
colorsarr = np.zeros((10, 3))
for xdim in range(0, colorsarr.shape[0]):
    colorsarr[xdim, :] = colors[xdim]
regionName = ['Midbrain', 'Olfactory Bulb', 'Thalamus', 'Hypothalamus', 'Hippocampus', 'Cerebellum', 'Cerebral Nuclei', 'Cortex', 'Medulla', 'Pons']
x = []; y = [];
rnames = []; correctColors = []
for index in range(0, 22):
    xlist = [i for i in table[index, :, 0].tolist() if i != 0.0]
    ylist = [i for i in table[index, :, 3].tolist() if i != 0.0]
    rlist = [name for name, i in zip(regionName, table[index, :, 3].tolist()) if i != 0.0]
    clist = [color for color, i in zip(colors, table[index, :, 3].tolist()) if i != 0.0]
    for xitem, yitem, citem, ritem in zip(xlist, ylist, clist, rlist):
        x.append(xitem)
        y.append(yitem)
        correctColors.append(citem)
        rnames.append(ritem)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, s=30, c=colors)
ax.xaxis.set_label_text("Percentage microvasculature")
ax.yaxis.set_label_text("Average voxelized Tortuosity")
recs = []
for i in range(0, len(colors)):
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))
ax.legend(recs, regionName, loc='upper left', fontsize='small')
fig.tight_layout()
fig.savefig("statPV3.png")

iskpx = 135; iskpz = 10; iskpy = 71
klist = [kv for kv in range(2632, 8026 - 68, iskpx) if (kv > 2420 and kv < 4000) or (kv > 5500 and kv < (7267 + 135))]
# # dictSlice = {'saggital': 0, 'transverse': 2, 'coronal': 1}
image = imread("sagittalUpsampledAllen.png")
for i in klist[:-3]:
    image[(i / 7), :, :] = 0
imsave("all.png", image)

topIm = imread("Mouse_brain_sagittal__582x279.png")
klist = [kv for kv in range(2632, 8026 - 68, 135) if (kv > 2420 and kv < 4000) or (kv > 5500 and kv < (6952 + 135))]
klist1 = [kv for kv in klist if kv < 5602]
klist1.reverse()
klist2 = [kv for kv in klist if kv >= 5602]
klist2.reverse()
imNames = ['length', 'totTortuosity', 'voxTortuosity']
im = np.ones((917, 582), dtype=np.uint8) * 255
for imName in imNames:
    for i in range(1, 12):
        im[4 * (i) + 79 * (i - 1): (79 + 4) * (i), 45: 246 + 45] = imread("transverseSlice" + imName + "%i.png" % klist2[i - 1])
        im[4 * (i) + 79 * (i - 1): (79 + 4) * (i), 246 + 90: 246 + 246 + 90] = imread("transverseSlice" + imName + "%i.png" % klist1[i - 1])

    rows, cols = im.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    colorList = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0.5, 0, 0.5], [1, 1, 0], [1, 0.5, 0], [0.5, 0.5, 1], [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0, 0]]
    for i, colorn in enumerate(colorList):
        i = i + 1
        color_mask[(25) + 83 * (i - 1): (55) + 83 * (i - 1), 10:40] = colorn
        color_mask[(25) + 83 * (i - 1): (55) + 83 * (i - 1), 246 + 55: 246 + 85] = colorn

    # Construct RGB version of grey-level image
    img_color = np.dstack((im, im, im))
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.5

    img_masked = color.hsv2rgb(img_hsv)
    imsave("result" + imName + ".png", img_masked)

# klist = [x for x in range(2632, 8026 - 68, 135) if (x > 2420 and x < 4000) or (x > 5500 and x < (7267 + 135))]
regionName = ['Midbrain', 'Olfactory Bulb', 'Thalamus', 'Hypothalamus', 'Hippocampus', 'Cerebellum', 'Cerebral Nuclei', 'Cortex', 'Medulla', 'Pons']
columnName = ["Samples", "Sum(F1)", "Average(F1)", "Sum(F2)", "Average(F2)", "Sum(F3)", "Average(F3)", "Sum(F4)", "Average(F4)"]
slices = [str(kval) for kval in klist[:-3]]
writer = pandas.ExcelWriter('output.xlsx', engine='xlsxwriter')
for index in range(0, 10):
    df1 = pandas.DataFrame(np.vstack((table1[:, index, 0], table[:, index, 0], table2[:, index, 0], table[:, index, 1], table2[:, index, 1], table[:, index, 2], table2[:, index, 2], table[:, index, 3], table2[:, index, 3])).T, slices, columnName)
    df1.to_excel(writer, regionName[index])
writer.save()


# In [14]:         print(colorAnnotate[int((keyMax[0][0] - 160) / iskpz), int(keyMax[0][1] / iskpy), :])
# [255 255   0]

# In [15]: keyMax
# Out[15]: [(300, 2694, 3577)]
# 0.0936540161561 [(300, 2694, 3577)]

# 0.0037841488014591046 [(890, 7451, 3982)]
# [  0 255   0]
# 7.20494207827 [(290, 9226, 5602)]
# [128   0 128]
# 6.24894418926 [(280, 11072, 5872)]
# [0 255 0]
AvgFeature = np.zeros((table.shape[1], table.shape[2]))
for i in range(0, table.shape[2]):
    for j in range(0, table.shape[1]):
        AvgFeature[j, i] = np.sum(table2[:, j, i]) / np.sum(table[:, j, i])
regionName = ['Midbrain', 'Olfactory Bulb', 'Thalamus', 'Hypothalamus', 'Hippocampus', 'Cerebellum', 'Cerebral Nuclei', 'Cortex', 'Medulla', 'Pons']
columnName = ["Percentage Microvasculature", "Segment Length per Volume", "Total Tortuosity", "Voxelized Tortuosity"]
writer = pandas.ExcelWriter('finalTable_May4.xlsx', engine='xlsxwriter')
df1 = pandas.DataFrame(AvgFeature[:, :], regionName, columnName)
df1.to_excel(writer)
writer.save()
