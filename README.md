# 3scan-skeleton
3D Image Skeletonization Tools
REQUIREMENTS:
python > 3 https://www.python.org/downloads/
miniconda http://conda.pydata.org/miniconda.html
and then use conda to install the following packages
scipy
numpy
skfmm - (conda install -c pranathi scikit-fmm=0.0.8)
networkx
skimage
matplotlib

This repository contains programs needed to obtain
a 3D skeleton using python 3 and quantify the 
skeleotnized array to statistics like radius at each 
node, orientation in the folder skeleton.
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


The folder tests contains an __init__.py file which explains what this
folder is useful for
Install nosetests to execute programs in this folder
conda install nose
pip install nose

The folder notebook contains ipynb(ipython notebook file)
created to see how the graph reconstruction and quantification
of segments is working

conda install notebook
to install notebook
to open up a network based interactive python shell type the following 
command in terminal
ipython notebook
