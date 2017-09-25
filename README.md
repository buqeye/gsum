# BUQEYE Model

<!-- ![picture](https://www.dropbox.com/s/3a78gnhhkl6mzjv/BUQEYE2_v5.pdf?raw=1)

<img src="https://www.dropbox.com/s/3a78gnhhkl6mzjv/BUQEYE2_v5.pdf?dl=1" />

<img src="https://www.dropbox.com/s/3a78gnhhkl6mzjv/BUQEYE2_v5.pdf?raw=1" />
sadfd -->

The BUQEYE Model package provides two classes that allow one to analyze the convergence pattern of Effective Field Theory (EFT) observables.
Specifically, this is a MCMC-based implementation of the statistical model developed in [this paper](https://arxiv.org/abs/1506.01343).
The heavy lifting is done by the `PyMC3` package, which can be downloaded [here](https://github.com/pymc-devs/pymc3).

# Installation

The latest release of BUQEYE Model can be installed from PyPI using `pip`:
```
pip install buqeyemodel
```

# Dependencies

BUQEYE Model has only been tested with python 3.5, and currently relies only on `PyMC3>=3.1rc3`.


# Usage

This section is under construction.
BUQEYE Model provides two classes: `ObservableModel` and `ExpansionParameterModel`.
ObservableModel takes coefficients for some generic observable and sets up the relevant hyperparameters to model the coefficients as draws from a Gaussian process (GP).
The `ExpansionParameterModel` is meant to be provided as an (optional) argument to one or many `ObservableModel` instances.
If provided, this will allow the model to also learn the expansion parameter that best allows the coefficients to look like draws from a GP.


They can be used inside a model context as follows:
```python
with pm.Model() as gp_model:
    Q = ExpansionParameterModel(name='Q', breakdown_eval, breakdown_kwargs)
    cross_section = ObservableModel(coeff_data, inputs, index_list, cov_kwargs,
                                    expansion_parameter=Q, name='cross_section')
```
The arguments must be of the following form:
* `ExpansionParameterModel`
  - `name`: The name of the model context created by the classes. All RVs defined in the classes will have names `'name_*'`.
  -
  -  
* `ObservableModel`
  - `name`: The name of the model context created by the classes. All RVs defined in the classes will have names `'name_*'`.
  - `coeff_data`: A matrix with rows of coefficients, whose entries contain a coefficient evaluated along the domain
  - `input`: The domain values where the coefficients are observed. The rows are points and columns are the dimension. In the 1D case, this must be a column vector.
  - `index_list`: A list of the powers of the expansion parameter from which the coefficients were extracted. Must be in one-to-one correlation to the rows of `coeff_data`.
  - `*_kwargs`: arguments are to be dictionaries containing information to set up the priors for the `*` variable. The format of the dictionaries must contain a key `'dist'`, which could be, for example `pm.Normal`, and the remaining entries in the dictionary are kwargs to be fed into the `dist` as in `dist(**kwargs)`.

Examples of distributions for the priors can be found [here](http://docs.pymc.io/api/distributions.html).
I am currently thinking about how to allow GP priors for the hyperparameters.
This may require that the method of passing priors through dictionaries be reconsidered.
See [this page](http://docs.pymc.io/notebooks/GP-covariances.html) on kernels and covariance functions.
