from distutils.core import setup, Extension
from Cython.Build import cythonize, build_ext
import numpy
import cython_gsl

# Run: python setup.py build_ext --build-lib ./

extensions = [Extension(
    name="gsum.cutils", 
    sources=["gsum/cutils.pyx"],
    # depends=['gsl/gsl_linalg.h', 'gsl/gsl_permutation.h'],
    libraries=[*cython_gsl.get_libraries()],
    library_dirs=[cython_gsl.get_library_dir()],
    include_dirs=[numpy.get_include(), cython_gsl.get_cython_include_dir()],
)]

setup(
    name='gsum',
    packages=['gsum'],
    # py_modules=[''],
    include_dirs=[cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, build_dir='gsum'),
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
        ]
)
