
.. raw:: html

   <p style="text-align:center;">
        <img src="https://cdn.rawgit.com/jordan-melendez/gsum/c85d832b/docs/logos/gsum.png" width="30%" alt="gsum logo" hspace="20"/>
        <img src="https://cdn.rawgit.com/jordan-melendez/buqeyemodel/af4da985/BUQEYE_fig.png" width="30%" alt="BUQEYE logo" hspace="20"/>
   </p>

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/jordan-melendez/gsum/master

The `gsum` package provides convenient classes that allow one to analyze the convergence pattern of Effective Field Theory (EFT) observables.
Specifically, this is a conjugacy-based implementation of the statistical model developed in `this paper <https://arxiv.org/abs/1904.10581>`_.
Although the package is fully functional, we are still working on filling in all of the documentation.
Examples of how it was used to generate results can be found in the `documentation <https://buqeye.github.io/gsum>`_.
The notebooks used in the documentation can be found in `docs/notebooks <https://github.com/buqeye/gsum/tree/master/docs/notebooks>`_.


Installation
============

The current working branch can be installed via

.. code-block:: console

  $ pip install git+https://github.com/jordan-melendez/gsum

Additionally, one can clone the repository and install using

.. code-block:: console

  $ git clone https://github.com/jordan-melendez/gsum.git
  $ cd gsum
  $ pip install .

This will install the package as is, but if you want to make edits to the code for testing you can exchange the last line with ``pip install -e .``, which will allow you to edit the package without reinstalling.

For some Mac users, the build might fail when using the ``pip install .` method. This may be due to the removal of the `usr/include` from new MacOS versions.
Running the following command will fix this problem:
```
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

Soon, an up-to-date version of this package will be added to `pip`, for an easier installation process.

Dependencies
============

gsum is compatible with python 3, but has not been tested with python 2.
See the requirements.txt for the full list of dependencies.

Citing gsum
============

If you have found this package helpful, please cite our paper Melendez et al. (2019), "Quantifying Correlated Truncation Errors in Effective Field Theory" `arXiv:1904.10581 <https://arxiv.org/abs/1904.10581>`_.

Contact
=======

To report an issue please use the `issue tracker <https://github.com/jordan-melendez/gsum/issues>`_.

License
=======

`MIT License <https://github.com/jordan-melendez/gsum/blob/master/LICENSE.txt>`_.
