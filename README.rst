.. # BUQEYE Model

.. <!-- <img src="./BUQEYE_fig.pdf?raw=true" width="30%"/> -->

.. <img src="./BUQEYE_fig.png?raw=true" width="30%"/>

.. .. image : : ./docs/logos/gsum.png

.. raw:: html

   <p style="text-align:center;">
        <img src="https://cdn.rawgit.com/jordan-melendez/gsum/c85d832b/docs/logos/gsum.png" width="30%" alt="gsum logo"/>
        <img src="https://cdn.rawgit.com/jordan-melendez/buqeyemodel/af4da985/BUQEYE_fig.png" width="30%" alt="BUQEYE logo"/>
   </p>

.. raw : : html

   <p style="text-align:center;"><img src="https://cdn.rawgit.com/jordan-melendez/buqeyemodel/af4da985/BUQEYE_fig.png" width="30%" alt="BUQEYE logo"/></p>


.. .. image : : https://cdn.rawgit.com/jordan-melendez/buqeyemodel/af4da985/BUQEYE_fig.png
..   :height: 150px
..   :align: center
..   :alt: BUQEYE logo

The gsum package provides two classes that allow one to analyze the convergence pattern of Effective Field Theory (EFT) observables.
Specifically, this is a conjugacy-based implementation of the statistical model developed in `this paper <https://arxiv.org/abs/1506.01343>`_.

.. The heavy lifting is done by the ``PyMC3`` package, which can be downloaded `here <https://github.com/pymc-devs/pymc3>`_.
.. Some working knowledge of ``PyMC3`` is recommended before reading the usage information below.

Installation
============

The latest release of gsum can be installed from PyPI using ``pip``:

.. code-block:: console

  $ pip install gsum

The current working branch can be installed via

.. code-block:: console

  $ pip install git+https://github.com/jordan-melendez/gsum

Additionally, one can clone the repository and install using

.. code-block:: console

  $ git clone -b develop https://github.com/jordan-melendez/gsum.git
  $ cd gsum
  $ pip install .

Change ``-b {branch}`` to ``-b master`` etc. if desired.
This will install the package as is, but if you want to make edits to the code for testing you can exchange the last line with ``pip install -e .``, which will allow you to edit the package without reinstalling.

Dependencies
============

gsum has only been tested with python 3.5.
.. Additionally, some functionality relies on the ``StatsModels`` package.

Citing gsum
============


Contact
=======

To report an issue please use the `issue tracker <https://github.com/jordan-melendez/gsum/issues>`_.

License
=======

`MIT License <https://github.com/jordan-melendez/gsum/blob/master/LICENSE.txt>`_.


.. # Usage

.. BUQEYE Model provides two classes: `ObservableModel` and `ExpansionParameterModel`.
.. `ObservableModel` takes coefficients for some generic observable and models the coefficients as draws from a Gaussian process (GP) with some specified covariance function.
.. The `ExpansionParameterModel` is meant to be provided as an (optional) argument to one or many `ObservableModel` instances.
.. If provided, this will allow the model to also learn the expansion parameter that best allows the coefficients to look like draws from a GP.


.. They can be defined inside a model context as follows:
.. ```python
.. import pymc3 as pm
.. from buqeyemodel import *

.. # Import data, etc. below
.. # ...

.. # Now set up model
.. with pm.Model() as gp_model:
..     Q = ExpansionParameterModel(breakdown_eval, breakdown_dist, name='Q')
..     cross_section = ObservableModel(coeff_data, X, index_list,
..                                     expansion_parameter=Q, name='cross_section')
.. ```
.. The arguments must be of the following form:
.. * `ExpansionParameterModel`
..   - `breakdown_eval`: The breakdown scale that was used to extract the coefficients
..   - `breakdown_dist`: A prior for the breakdown scale. Must be a distribution object, such as `pm.Lognormal.dist(mu=0, sd=10, testval=600.0)`, _**not**_ a random variable like `pm.Lognormal('breakdown', mu=0, sd=10, testval=600.0)`. Also, a `testval` must be given to begin sampling in a reasonable location. Presumably `breakdown_eval` would be as good as any, or else why did you choose that `breakdown_eval` in the first place?
..   - `name`: The name of the model context created by the classes. All RVs defined in the classes will have names `'name_*'`.
.. * `ObservableModel`
..   - `coeff_data`: A matrix with rows of coefficients, whose entries contain a coefficient evaluated along the domain
..   - `X`: The domain values where the coefficients are observed. The rows are points and columns are the dimension. In the 1D case, this must be a column vector.
..   - `index_list`: A list of the powers of the expansion parameter from which the coefficients were extracted, i.e., the subscripts of the coefficients. Must be in one-to-one correlation to the rows of `coeff_data`.
..   - `expansion_parameter`: An `ExpansionParameterModel` object, whose RVs will be learned on the basis of the values that will most make `coeff_data` look like draws from the specified GP.
..   - `name`: The name of the model context created by the classes. All RVs defined in the classes will have names `'name_*'`.

.. While we have created observable and expansion parameter instances and tied them together under `gp_model`, we must still build the covariance structure for the `cross_section`.
.. No defaults are provided, since the covariance should be built specifically for the given application.
.. To provide the most flexibility in the build, it is recommended that all RVs be built in a model context for each observable.
.. In the case of one observable, we only need:
.. ```python
.. with cross_section as model:
..     sd = pm.Lognormal('sd', mu=0, sd=100)
..     ls = pm.Normal('ls', mu=50, sd=20)
..     cov = sd**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls)
..     model.setup_model(cov=cov)
.. ```
.. This creates the RVs `cross_section_sd` and `cross_section_ls`, which are then combined into an exponentiated quadratic covariance function.
.. By feeding the covariance into the `setup_model` method, this completes the initialization of the model by relating the coefficient data to the model RVs.
.. Without this `setup_model` step, the model is useless!
.. * I really would like to add this `setup_model` step in the cleanup code of the `ObservableModel` class (i.e. `__exit__`) so that it happens automatically behind the scenes, but `PyMC3` enters additional contexts behind the scenes for various reasons, thus calling `__exit__` more times than I would like. Still thinking about if/how I can do this.


.. Now all that is left to do is sample:
.. ```python
.. with gp_model:
..     trace = pm.sample(1000)
.. ```
.. Plots can now be made with `pm.traceplot(trace)`, etc. Again, see `PyMC3` documentation.

.. Examples of distributions for the priors can be found [here](http://docs.pymc.io/api/distributions.html).
.. See [this page](http://docs.pymc.io/notebooks/GP-MeansAndCovs.html) on kernels and covariance functions.
