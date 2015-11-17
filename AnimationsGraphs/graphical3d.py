import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
# mlab.contour3d(skeletonStackAlloc)
listOfNonzeros = np.transpose(np.nonzero(skeletonIm))
listOfNonzero = listOfNonzeros.tolist()
xList = []; yList = []; zList = [];
for items in listOfNonzero:
        xList.append(items[2])
        yList.append(items[1])
        zList.append(items[0])
dataDict = {'x': xList, 'y': yList, 'z': zList}
trace1 = Scatter3d(x=dataDict['x'], y=dataDict['y'], z=dataDict['z'], mode='markers', marker=Marker(
    size=12,
    line=Line(
        color='rgba(217, 217, 217, 0.14)',
        width=0.5
    ),
    opacity=0.8
)
)
data = Data([trace1])
py.plot(data)


# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


X, Y, Z = xList, yList, zList

ax.plot_wireframe(X, Y, Z)

plt.show()
