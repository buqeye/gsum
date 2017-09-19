import pymc3 as pm
import numpy as np


__all__ = ['ObservableModel', 'ExpansionParameterModel']

def bn_cov(cov, Lambda_b, n, cov_dim):
    ones = np.ones((cov_dim, cov_dim))
    return pm.math.exp(pm.math.log(cov) - 2 * n * ones * pm.math.log(Lambda_b))


class ObservableModel(pm.Model):
    """A statistical model for the convergence pattern of observables in an EFT.

    Parameters
    ----------
    data   :
    inputs :
    name   :
    model  :

    """

    def __init__(self, coeff_data, inputs, index_list,
                 corr_length_obs_list=None,
                 expansion_parameter=None,
                 name='', model=None, **kwargs):
        # name will be prefix for all variables here
        # if no name specified for model there will be no prefix
        super(ObservableModel, self).__init__(name, model)

        noise = 1e-10
        cbar_lower = 1e-3
        cbar_upper = 1e3

        # The number of coefficient (functions)
        num_coeffs = len(index_list)

        # Domain scaling factors (account for scales other than [0, 1])
        domain_scales = np.amax(inputs, axis=0) - np.amin(inputs, axis=0)

        # Observations shape
        input_dim = len(inputs[0])
        # print(input_dim)

        assert len(coeff_data) == len(index_list), \
            "Indices must match number of coefficients"

        # Create a log-uniform distribution for cbar
        logcbar_lower = pm.math.log(cbar_lower)
        logcbar_upper = pm.math.log(cbar_upper)
        logcbar = pm.Uniform(
            'logcbar', logcbar_lower, logcbar_upper,
            # transform=None,
            # testval=pm.math.log(cbar_true)
            )
        self.cbar = pm.Deterministic('cbar', pm.math.exp(logcbar))

        cov = self.cbar**2

        # Gamma prior for length scales
        # Domain scaling: If X ~ Gamma(a, b), then cX ~ Gamma(a, b/c)
        # lengthscales = pm.Gamma("lengthscales",
        #                         alpha=2, beta=1/domain_scales[0],
        #                         observed=corr_length_obs_list,
        #                         shape=input_dim)

        # Let the covariance factor by dimension, each with ExpQuad
        for dim in range(input_dim):
            if corr_length_obs_list is None:
                ls_obs = None
            else:
                ls_obs = corr_length_obs_list[dim]
            # Gamma prior for length scales
            # Domain scaling: If X ~ Gamma(a, b), then cX ~ Gamma(a, b/c)
            # ls = pm.Gamma("length{}".format(dim),
            #               alpha=5, beta=5/domain_scales[dim]/3,
            #               observed=ls_obs
            #               )
            # print("ds", domain_scales[dim])
            # ls = pm.Normal("length{}".format(dim),
            #                mu=.5*domain_scales[dim],
            #                sd=.2*domain_scales[dim],
            #                # observed=ls_obs
            #                )
            ls = pm.Normal("length{}".format(dim),
                           mu=300,
                           sd=100,
                           # observed=ls_obs
                           )

            cov = cov * pm.gp.cov.ExpQuad(
               input_dim=input_dim, ls=ls, active_dims=[dim]
               )

        for n, cn in zip(index_list, coeff_data):
            scale = 1
            if expansion_parameter is not None:
                scale = (expansion_parameter.scale())**n

            scaled_cov = 1 / scale**2 * cov
            # ckvecset.append(
            #     pm.MvNormal(
            #         'c{}'.format(n),
            #         mu=0,
            #         cov=scaled_cov + cov_noise,
            #         shape=obs_shape,
            #         observed=cn
            #         )
            #     )
            gp = pm.gp.Marginal(cov_func=scaled_cov)
            cnobs = gp.marginal_likelihood(
                'c{}obs'.format(n),
                X=inputs,
                y=cn,
                noise=noise
                )

            pm.Deterministic("c{}".format(n), scale * cnobs)


class ExpansionParameterModel(pm.Model):
    """
    """

    def __init__(self, breakdown_eval, name='', model=None,
                 breakdown_lower=1, breakdown_upper=10000,
                 **kwargs):
        super(ExpansionParameterModel, self).__init__(name, model)

        # Define hard breakdown scale \Lambda_b
        logbreakdown_lower = pm.math.log(breakdown_lower)
        logbreakdown_upper = pm.math.log(breakdown_upper)
        self.logbreakdown = pm.Uniform(
            'logbreakdown', logbreakdown_lower, logbreakdown_upper
            )
        self.breakdown = pm.Deterministic(
            'breakdown',
            pm.math.exp(self.logbreakdown)
            )

        self.breakdown_eval = breakdown_eval

    def scale(self):
        return self.breakdown/self.breakdown_eval

# Obs = ObservableModel([], [], name="test")

# print(Obs['v1'], Obs.cbar_lower)

# print(Obs['v2'])

# print(Obs.model)

# x1, x2 = np.meshgrid(np.linspace(0,1,10), np.arange(1,4))
# X2 = np.concatenate((x1.reshape((30,1)), x2.reshape((30,1))), axis=1)

# print(x1, x2)
# print(X2)
