from setuptools import find_packages
from distutils.core import setup, Extension
import os
import numpy
import subprocess

from distutils.command.install import install



VERSION = '0.0.0' 
DESCRIPTION = 'pyexspec'

fh = open('README.md', 'r')
LONG_DESCRIPTION = fh.read()
fh.close()


# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pyexspec",
        version=VERSION,
        author="Jiao Li",
        author_email="lijiao@bao.ac.cn",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        url ='https://github.com/lidihei/pyexspec',
        install_requires=['numpy', 'astropy', 'matplotlib',
                         ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        entry_points={
             'console_scripts': ['pyexspec=pyexspec.main:main'],
                     },
        classifiers= [
            "Development Status :: beta",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Astronomy"
        ],
        #cmdclass={'install': _install},
        #include_package_data=True,
)

