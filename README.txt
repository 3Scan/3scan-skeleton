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

Input must be a binary array with z in its first dimension

Thinning is cythonized for fast execution and pyximport is used to automatically build and use cythonized function
(reference - http://docs.cython.org/src/reference/compilation.html)

This repository contains programs needed to obtain a 3D skeleton using python 3 and quantify the 
skeletonized array to statistics with the help of function present in metrics.segmentStats

To use import functions of this repo follow runscripts.getMetrics program

To view the 3D volume (input or skeleton output) and save
use Mayavi with python 2.6 (conda create -n mayavi python=2.6)
with insturctions executed in a virtual environment with python 2.6 as below  
*mlab.contour3d(anynpynonbooleanarray)
*mlab.options.offscreen = True
*mlab.savefig("arrayName.png")


Install nosetests to run tests in 3scan-skeleton/skeleton folder under the name submodule_tests
*conda install nose
*pip install nose
nosetests -xs path to + "3scan-skeleton/skeleton/"
To run tests with coverage
*conda install coverage
nosetests -sxv --with-coverage --cover-package=3scan-skeleton/tests


