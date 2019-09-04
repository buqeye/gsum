from distutils.core import setup, Extension

# To build files manually, run: python setup.py build_ext --build-lib ./

# extensions = [Extension(
#     name="gsum.cutils",
#     sources=["gsum/cutils.pyx"],
#     # depends=['gsl/gsl_linalg.h', 'gsl/gsl_permutation.h'],
#     # libraries=[*cython_gsl.get_libraries()],
#     # library_dirs=[cython_gsl.get_library_dir()],
#     # include_dirs=[numpy.get_include(), cython_gsl.get_cython_include_dir()],
# )]

# Do not import packages if they are not yet installed via install_requires
try:
    from Cython.Build import cythonize, build_ext
except ImportError:
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    from setuptools.command.build_ext import build_ext
    ext_modules = [Extension("gsum.cutils", ["gsum/cutils.c"])]
else:
    # If we successfully imported Cython, look for a .pyx file
    ext_modules = cythonize([Extension("gsum.cutils", ["gsum/cutils.pyx"])])


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy/cython_gsl headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        import cython_gsl

        # Add the headers
        self.libraries = cython_gsl.get_libraries()
        self.library_dirs.append(cython_gsl.get_library_dir())

        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append(cython_gsl.get_include())
        self.include_dirs.append(cython_gsl.get_cython_include_dir())

        # Call original build_ext command
        build_ext.run(self)


setup(
    name='gsum',
    packages=['gsum'],
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=ext_modules,
    version='0.1',
    description='A Bayesian model of series convergence using Gaussian sums',
    author='Jordan Melendez',
    author_email='jmelendez1992@gmail.com',
    license='MIT',
    url='https://github.com/jordan-melendez/gsum.git',
    download_url='',
    keywords='EFT nuclear model gaussian process uncertainty quantification buqeyemodel buqeye',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
        ],
    install_requires=[
        'Cython',
        'CythonGSL',
        'docrep',
        'gsl',
        'numpy>=1.12.0',
        'pandas',
        'scipy',
        'seaborn',
        'statsmodels',
        'matplotlib',
        'scikit-learn',
    ]
)
