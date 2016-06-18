from distutils.core import setup
from Cython.Build import cythonize

setup(name='convolve app', ext_modules=cythonize("conv.pyx"),)
