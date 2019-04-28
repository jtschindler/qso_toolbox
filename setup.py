#!/usr/bin/env python

from distutils.core import setup, Command

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='qsotoolbox',
      version='0.1',
      description='A library of useful tools for quasar selection',
      author='Jan-Torge Schindler, Eduardo Banados',
      author_email='schindler@mpia.de, banados@mpia.de',
      license='MIT',
      url='http://github.com/jtschindler/qso_sel_toolbox',
      packages=['qsotoolbox'],
      provides=['qsotoolbox'],
      package_dir={'qsotoolbox':'qsotoolbox'},
      package_data={'qsotoolbox':['data/*.*','data/passbands/*.*']},
      requires=['numpy', 'matplotlib', 'scipy', 'astropy', 'pandas'],
      keywords=['Scientific/Engineering'],
     )
