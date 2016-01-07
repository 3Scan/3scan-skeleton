# 3scan-skeleton
3D Image Skeletonization Tools

This repository contains programs needed to obtain
a 3D skeleton using python 3 and quantify the 
skeleotnized array to statistics like radius at each 
node, orientation.
The skeletonized 3D volume is converted to a graph 
using networkx module of python 3 
to further  find statistics like
number of segments, lengths and tortuoisty of the
disjoint components in a graph
To view the 3D volume (input or skeleton output) and save
use Mayavi with python 2.6 (conda create -n mayavi python=2.6)
with insturctions executed in a virtual environment with python 2.6 as below  
mlab.contour3d(anynpyarray)
mlab.options.offscreen = True
mlab.savefig("arrayName.obj")
