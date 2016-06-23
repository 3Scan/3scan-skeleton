# 3scan-skeleton
3D Image Skeletonization Tools
REQUIREMENTS:
python > 3 https://www.python.org/downloads/
miniconda http://conda.pydata.org/miniconda.html
and then use conda to install the following packages

*scipy
*numpy
*skfmm - (conda install -c pranathi scikit-fmm=0.0.8)
*networkx
*skimage
*matplotlib

This repository contains programs needed to obtain a 3D skeleton using python 3 and quantify the 
skeleotnized array to statistics with the help of function present in skeleton.segmentStats

To use import functions of this repo follow runscripts.getMetrics program

To view the 3D volume (input or skeleton output) and save
use Mayavi with python 2.6 (conda create -n mayavi python=2.6)
with insturctions executed in a virtual environment with python 2.6 as below  
*mlab.contour3d(anynpyarray)
*mlab.options.offscreen = True
*mlab.savefig("arrayName.obj")


Install nosetests to execute programs in this folder
*conda install nose
*pip install nose
Run nose tests from within the directory
pranathi@pranathi-3Scan:~/src/3scan-skeleton$ nosetests -xs "/home/pranathi/src/3scan-skeleton/tests/"


The folder notebook contains ipynb(ipython notebook file)
created to see how the graph reconstruction and quantification
of segments is working

*conda install notebook
to install notebook
to open up a network based interactive python shell type the following 
command in terminal
*ipython notebook
