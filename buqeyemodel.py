import pymc3 as pm
import numpy as np
import theano.tensor as tt
import collections


__all__ = ['ObservableModel', 'ExpansionParameterModel']

# def bn_cov(cov, Lambda_b, n, cov_dim):
#     ones = np.ones((cov_dim, cov_dim))
#     return pm.math.exp(pm.math.log(cov) - 2 * n * ones * pm.math.log(Lambda_b))


def update(d, u):
    """Update a nested dictionary d using u without overwriting subdicts"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class ObservableModel(pm.Model):
    """A statistical model for the convergence pattern of observables in an EFT.

    Parameters
    ----------
    data   :
    inputs :
    name   :
    model  :

    """

    # Could use defaults like "dist": pm.LogNormal
    # ls_default_kwargs = {"name": "ls", "dist": pm.Lognormal}
    # sd_default_kwargs = {"name": "sd", "dist": pm.Lognormal}
    # cov_default_kwargs = {
    #     "cov": pm.gp.cov.ExpQuad,
    #     "noise": 1e-10,
    #     "sd": sd_default_kwargs,
    #     "ls": ls_default_kwargs
    #     }

    def __init__(self, coeff_data, inputs, index_list,
                 # cov_kwargs={},
                 cov=None,
                 noise=1e-10,
                 expansion_parameter=None,
                 name='', model=None, **kwargs):
        # name will be prefix for all variables here
        # if no name specified for model there will be no prefix
        super(ObservableModel, self).__init__(name, model)

        # Overwrite any default parameters
        # self.cov_kwargs = self.cov_default_kwargs
        # update(self.cov_kwargs, cov_kwargs)

        # self.cov_kwargs["input_dim"] = self.input_dim

        # Store parameters
        self.data = coeff_data
        self.inputs = inputs
        self.input_dim = len(inputs[0])
        self.index_list = index_list
        self.noise = noise
        self.expansion_parameter = expansion_parameter

        # The number of coefficient (functions)
        self.num_coeffs = len(coeff_data)

        assert len(coeff_data) == len(index_list), \
            "Indices must match number of coefficients"

        # if "custom" in self.cov_kwargs:
        #     self.noise = self.cov_kwargs['noise']
        #     cov = self.cov_kwargs["custom"]
        # else:
        #     cov = self.setup_covariance(**self.cov_kwargs)

        if cov is not None:
            self.setup_model(cov)

    def setup_hyperparameter(self, **kwargs):
        temp_kwargs = kwargs
        dist = temp_kwargs.pop("dist")
        return dist(**temp_kwargs)

    def setup_covariance(self, **kwargs):
        temp_kwargs = kwargs
        # Convert dicts to hyperparameters, i.e., pymc3 objects
        for k, v in temp_kwargs.items():
            if isinstance(v, dict):
                temp_kwargs[k] = self.setup_hyperparameter(**v)
        # Save for later: Put noise directly into gp
        self.noise = temp_kwargs.pop("noise")
        # Setup covariance function
        cov_func = temp_kwargs.pop("cov")
        self.sd = temp_kwargs.pop("sd")
        cov = self.sd**2 * cov_func(**temp_kwargs)
        return cov

    def setup_model(self, cov, noise=None):
        """Once cov is set up, relate it to the coefficients and other RVs.
        Provides a chance to feed a noise model that may have been
        created in the ObservableModel context before setup completes.
        """
        if noise is not None:
            self.noise = noise

        scale = 1
        if self.expansion_parameter is not None:
            scale = self.expansion_parameter.scale

        for n, cn in zip(self.index_list, self.data):
            # Create a cov that handles an uncertain expansion parameter
            scaled_cov = scale**(-2*n) * cov

            # Treat the coefficients as draws from a GP
            # Constrain the model by the data:
            gp = pm.gp.Marginal(cov_func=scaled_cov)
            cnobs = gp.marginal_likelihood(
                'c{}obs'.format(n),
                X=self.inputs,
                y=cn,
                noise=self.noise
                )

            # Scale fixed cnobs due to possibly uncertain expansion parameter
            pm.Deterministic("c{}".format(n), scale**n * cnobs)


class ExpansionParameterModel(pm.Model):
    """
    """

    # breakdown_default_kwargs = {"name": "breakdown", "dist": pm.Lognormal}

    def __init__(self, breakdown_eval,
                 breakdown=None,
                 # breakdown_kwargs={},
                 name='',
                 model=None, **kwargs):
        super(ExpansionParameterModel, self).__init__(name, model)

        # Setup kwargs for breakdown scale random variable
        # self.breakdown_kwargs = self.breakdown_default_kwargs
        # Issues can occur if sampling doesn't begin in a reasonable region
        # self.breakdown_kwargs["testval"] = breakdown_eval
        # Override defaults
        # self.breakdown_kwargs.update(breakdown_kwargs)

        # Setup random variable for the breakdown scale
        # if "custom" in self.breakdown_kwargs:
        #     self.breakdown = self.breakdown_kwargs["custom"]
        # else:
        #     self.breakdown = self.setup_hyperparameter(**self.breakdown_kwargs)

        self.breakdown_eval = breakdown_eval

        if breakdown is not None:
            self.breakdown = breakdown
            # The scaling factor for coefficients: c_n ~ scale^n
            self.scale = self.breakdown/self.breakdown_eval

    def setup_hyperparameter(self, **kwargs):
        temp_kwargs = kwargs
        dist = temp_kwargs.pop("dist")
        return dist(**temp_kwargs)

    def setup_scale(self, breakdown):
        # The scaling factor for coefficients: c_n ~ scale^n
        self.scale = breakdown/self.breakdown_eval
        return self.scale
