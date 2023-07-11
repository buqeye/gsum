from distutils.core import setup

setup(
    name='gsum',
    packages=['gsum'],
    version='0.3',
    description='A Bayesian model of series convergence using Gaussian sums',
    author='Jordan Melendez',
    author_email='jmelendez1992@gmail.com',
    license='MIT',
    url='https://github.com/buqeye/gsum.git',
    download_url='',
    keywords='EFT nuclear model gaussian process uncertainty quantification buqeyemodel buqeye',
    classifiers=[
        'Development Status :: 4 - Beta',
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
        'docrep',
        'gsl',
        'numpy>=1.12.0',
        'pandas',
        'scipy>=1.4.0',
        'seaborn',
        'statsmodels',
        'matplotlib',
        'scikit-learn',
    ]
)
