import numpy as np
from xml.dom import minidom

doc = minidom.parse("/media/pranathi/A336-5F43/image1/tree_structure.xml")
nonzerotuples = []
staffs = doc.getElementsByTagName("tup")
for i, staff in enumerate(staffs):
    print(i)
    floats = staff.getElementsByTagName("float")
    nonzerotuples.append(tuple((float(floats[0].firstChild.nodeValue), float(floats[1].firstChild.nodeValue), float(floats[2].firstChild.nodeValue))))


im = np.zeros((101, 101, 101), dtype=bool)
for index in nonzerotuples:
    im[index] = True
