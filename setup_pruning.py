# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:09:37 2019

@author: skyscan
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import skeleton.io_tools
#directives = {'linetrace': False, 'language_level': 3}
path = skeleton.io_tools.module_relative_path('skeleton/pruning.pyx')
ext_modules = [
    Extension("skeleton.pruning",
              [path],
              )]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}


setup(ext_modules=cythonize(ext_modules))
   