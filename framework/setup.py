#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

module1 = Extension('_iea_fortran', sources=['src/ieacase1/iea_bp_model.f90',
                                                   # 'src/gaussianwake/gaussianwake_bv.f90',
                                                   'src/ieacase1/iea_bp_model_dv.f90',
                                                   'src/ieacase1/adStack.c',
                                                   'src/ieacase1/adBuffer.f'],
                    extra_compile_args=['-O2', '-c'])

setup(
    name='IEACase1',
    version='0.0.1',
    description='Gaussian wake model for IEA 37 optimization case studies, case 1',
    install_requires=['openmdao>=1.7'],
    package_dir={'': 'src/'},
    ext_modules=[module1],
    dependency_links=['http://github.com/OpenMDAO/OpenMDAO.git@master'],
    packages=['ieacase1'],
    license='Apache License, Version 2.0',
)