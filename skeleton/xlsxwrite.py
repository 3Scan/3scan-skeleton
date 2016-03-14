import xlsxwriter


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
    lenXls = len(dictR)
    footer = [str(sum(dictR.values())), str(sum(listOfDicts[1].values)), str(sum(listOfDicts[2].values)), str(sum(listOfDicts[3].values))]
    for numColumn in range(0, 5):
        worksheet.write(row, numColumn, header[numColumn])
        worksheet.write(lenXls + 1, numColumn, footer[numColumn])
        worksheet.write(lenXls + 2, numColumn, (footer[numColumn] / lenXls))
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


def excelWrite(shskel, boundaryIm, path):
    from skeleton.segmentLengths import getSegmentsAndLengths
    from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline
    d1, d2, d3, t, typeGraph = getSegmentsAndLengths(shskel)
    d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
    dictR = {your_key: d[your_key] for your_key in d1.keys()}
    listOfDicts = [dictR, d1, d2, d3]
    xlsxWrite(listOfDicts, path)
    d = {}
    for keys in list(dictR.keys()):
        d[keys] = str(d1[keys])
        d[keys] = '   '.join(d[keys])
    return d


def excelRead():
    import xlrd
    # Open the workbook
    xl_workbook = xlrd.open_workbook('/media/pranathi/KINGSTON/resultSlices-Badregion/xlsx_test/Cross.xlsx')

    # List sheet names, and pull a sheet by name
    #
    sheet_names = xl_workbook.sheet_names()
    print('Sheet Names', sheet_names)

    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # Or grab the first sheet by index
    #  (sheets are zero-indexed)
    #
    xl_sheet = xl_workbook.sheet_by_index(0)
    print ('Sheet name: %s' % xl_sheet.name)

    # Pull the first row by index
    #  (rows/columns are also zero-indexed)
    #
    row = xl_sheet.row(0)  # 1st row
    # Print 1st row values and types
    #
    from xlrd.sheet import ctype_text

    print('(Column #) type:value')
    for idx, cell_obj in enumerate(row):
        cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
        print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))

    # Print all values, iterating through rows and columns
    #
    num_cols = xl_sheet.ncols   # Number of columns
    for row_idx in range(0, xl_sheet.nrows):    # Iterate through rows
        print ('-' * 40)
        print ('Row: %s' % row_idx)   # Print row number
        for col_idx in range(0, num_cols):  # Iterate through columns
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            print ('Column: [%s] cell_obj: [%s]' % (col_idx, cell_obj))


if __name__ == '__main__':
    d = excelWrite()
    from mayavi import mlab
    # to plot text on an mlab figure
    for coord in list(d.keys()):
        x = coord[0]; y = coord[1]; z = coord[2];
        mlab.text3d(x, y, z, d[coord], color=(0, 0, 0), scale=4.0)
