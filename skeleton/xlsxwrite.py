import xlsxwriter

import numpy as np


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

header = ['Branch point', 'Radius', 'Segment Count', 'Segment Lengths', 'Segment tortuosities']


def xlsxWrite(listOfDicts, path):
    dictR = listOfDicts[0]
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    row = 0
    for numColumn in range(0, 4):
        worksheet.write(row, numColumn, header[numColumn])
    for key in list(dictR.keys()):  # for each of the nodes with radius
        row += 1; col = 0
        worksheet.write(row, col, str(key))
        for i in range(0, 4):    # increment a column and copy the corresponding lengths and tortuosities at the branch node
            d = listOfDicts[i]
            if i not in [0, 1]:   # for dictionaries with a key that includes end points
                listOfKeys = [allKeys for allKeys in list(d.keys()) if key == tuple(list(allKeys[1]))]
                visKeys = []
                if len(listOfKeys) > 1:
                    listOfVals = []
                    for j in listOfKeys:  # multiple segments at the same branch node
                        listOfVals.append(d[j])
                        visKeys.append(j)
                    worksheet.write(row, col + i + 1, str(listOfVals))
                elif key not in visKeys:  # if only one branch at the node
                    for allKeys in list(d.keys()):
                        if key == tuple(list(allKeys[1])):
                            worksheet.write(row, col + i + 1, d[allKeys])
            else:
                worksheet.write(row, col + i + 1, d[key])
    workbook.close()


def excelWrite():
    from skeleton.BifurcatedsegmentLengths import getBifurcatedSegmentsAndLengths
    from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    boundaryIm = np.load(input("enter a path to boundary of thresholded volume------"))
    path = input("enter a path to save analysis xlsx file at------")
    # segmentdict, disjointgraphDict = getStatsDisjoint(shskel)
    d1, d2, d3, t = getBifurcatedSegmentsAndLengths(shskel)
    d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
    dictR = {your_key: d[your_key] for your_key in d1.keys()}
    listOfDicts = [dictR, d1, d2, d3]
    xlsxWrite(listOfDicts, path)
    d = {}
    for keys in list(dictR.keys()):
        d[keys] = (str(round(dictR[keys], 2)), str(d1[keys]))
        d[keys] = '   '.join(d[keys])
    return d


if __name__ == '__main__':
    d = excelWrite()
    from mayavi import mlab
    # to plot text on an mlab figure
    for coord in list(d.keys()):
        x = coord[0]; y = coord[1]; z = coord[2];
        mlab.text3d(x, y, z, d[coord], color=(0, 0, 0), scale=2.0)
