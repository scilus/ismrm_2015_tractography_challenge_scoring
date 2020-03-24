#!/usr/bin/env python
from setuptools import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

from glob import glob
import os

try:
    import numpy
except ImportError as e:
    e.args += ("Try running pip install numpy",)
    raise e

try:
    import scipy
except ImportError as e:
    e.args += ("Try running pip install scipy",)
    raise e


class deactivate_default_build_ext(build_ext):

    def run(self):
        print("Please use one of the custom commands to build this project.\n" +
              "To see the list of commands, check the 'Extra commands' section of\n" +
              "   python setup.py --help-commands")


# Will try to build all extension modules
# Forced to be inplace for ease of import.
# TODO change this
class build_inplace_all_ext(build_ext):

    description = "build optimized code (.pyx files) " +\
                  "(compile/link inplace)"

    # Override to keep only the stats extension.
    def finalize_options(self):
        # Force inplace building for ease of importation
        self.inplace = True

        build_ext.finalize_options(self)


ext_modules = []
ext_modules.append(Extension('challenge_scoring.tractanalysis.robust_streamlines_metrics',
                             ['challenge_scoring/tractanalysis/robust_streamlines_metrics.pyx'],
                             include_dirs=[numpy.get_include(),
                                           os.path.join(
                                               os.path.dirname(
                                                   os.path.realpath(__file__)),
                                               'challenge_scoring/c_src')]))

dependencies = ['dipy', 'nibabel']

setup(name='ismrm_2015_tractography_challenge_scoring', version='1.0.1',
      description='Scoring system used for the ISMRM 2015 Tractography Challenge',
      url='https://github.com/scilus/ismrm_2015_tractography_challenge_scoring',
      ext_modules=ext_modules, author='The challenge team',
      author_email='jean-christophe.houde@usherbrooke.ca',
      scripts=glob('scripts/*.py'), install_requires=dependencies,
      cmdclass={'build_ext': deactivate_default_build_ext,
                'build_all': build_inplace_all_ext})
