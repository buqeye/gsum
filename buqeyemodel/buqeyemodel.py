import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
import warnings
from buqeyemodel.pymc3_additions import MatNormal


__all__ = ['ObservableModel', 'ExpansionParameterModel']


class ObservableModel(pm.Model):
    """A statistical model for the convergence pattern of observables in an EFT.

    Parameters
    ----------
    coeff_data          : ndarray
                          Holds arrays of coefficients: c_i, c_j, ...
                          which have been extracted for many parameter values
    X                   : ndarray (2D)
                          The parameter values for which the coefficients have
                          been extracted. Rows correspond to data points
                          and columns to the dimensions. For 1D observations,
                          this becomes a column vector.
    index_list          : list
                          A list containing the powers of the expansion
                          parameter from which the coefficients were extracted,
                          i.e., the subscripts of the coefficients.
                          E.g.: for coefficients [c_i, c_j, c_k], then
                          index_list = [i, j, k]
    cov                 : pymc3.cov object
                          A covariance function object to be used for the
                          coefficients. Unless using the exact same cov across
                          multiple observables, this is not the preferred
                          method of defining a cov. Instead, a cov and its
                          relevant RVs should be defined in a model context
                          for each individual observable.
    noise               : float or pymc3.RV object
                          The noise to be added to the cov. This is required
                          for numerical stability, but can also model real
                          noise in the data. This noise is *not* scaled by the
                          expansion_parameter!
    expansion_parameter : pymc3.Model object
                          An expansion parameter that can be learned _across_
                          various observables. See ExpansionParameterModel.
    name                : str
                          The name of the observable. This name will be
                          placed before all RV names defined within this model,
                          i.e. 'name_sd'.
    model               : pymc3.Model object
                          The parent model. If defined within a model context,
                          it will use that one.

    """

    def __init__(self, coeff_data, X, index_list, cov=None,
                 noise=1e-10, expansion_parameter=None,
                 X_star=None,
                 name='', model=None, **kwargs):
        # name will be prefix for all variables here
        # if no name specified for model there will be no prefix
        super(ObservableModel, self).__init__(name, model)

        # Store parameters
        self.data = coeff_data
        self.X = X
        self.X_dim = len(X[0])
        self.index_list = index_list
        self.noise = noise
        self.expansion_parameter = expansion_parameter
        self.X_star = X_star

        # The number of coefficients
        self.num_coeffs = len(coeff_data)

        # Ensure that everything looks right (add more tests!)
        assert self.num_coeffs == len(index_list), \
            "Indices must match number of coefficients"

        # Finish setup without entering model context if cov is given
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

    def setup_cn(self, name, X, y, cov, noise, scale, order, **kwargs):
        scaled_cov = scale**(-2*order)*cov
        gp = pm.gp.Marginal(cov_func=scaled_cov)
        obs = gp.marginal_likelihood(
                name=name + str(order) + 'obs',
                X=X,
                y=y,
                noise=noise,
                **kwargs
                )
        scaled_obs = pm.Deterministic(name + str(order), scale**order * obs)
        return [gp, obs, scaled_obs]

    def setup_model(self, cov, noise=None):
        """Once cov is set up, relate it to the coefficients and other RVs.
        Provides a chance to feed a noise model, which may have been
        created in the ObservableModel context, before setup completes.
        """
        if cov is None:
            raise AttributeError(
                "No covariance function provided to {}".format(self.name)
                )
        if noise is not None:
            self.noise = noise

        if self.expansion_parameter is None:
            # Create a multi-observed GP
            self.gp = pm.gp.Marginal(cov_func=cov)
            obs = self.gp.marginal_likelihood(
                    'obs',
                    X=self.X,
                    y=self.data,
                    noise=self.noise,
                    shape=self.data.shape
                    )
        else:
            # Expand cov as kronecker product and learn all data at once
            scale = self.expansion_parameter.scale
            scales = [scale**(-2*n) for n in self.index_list]
            # scales_diag = tt.nlinalg.diag(scales) + \
            #     np.diag(self.noise*np.ones(len(scales)))
            scales_diag = tt.nlinalg.diag(scales)
            cov_mat = cov(self.X) + np.diag(self.noise*np.ones(len(self.X)))
            obs = MatNormal(
                    'obs', mu=0, rcov=cov_mat, lcov=scales_diag,
                    observed=self.data,
                    shape=(len(self.index_list), len(self.X))
                    )

            # obs = [pm.MvNormal('obs{}'.format(n), mu=0, cov=scale**(-2*n)*cov_mat + np.diag(self.noise*np.ones(len(self.X))),
            #                    observed=self.data[i])
            #        for i, n in enumerate(self.index_list)]

            self.gp = []
            # cn_scales = [scale**n for n in self.index_list]
            # cn_true = [cn_scales[i] * obs[i] for i in range(len(cn_scales))]
            # pm.Deterministic('cn', cn_true)

            # scaled_cov = ScaledCov(cov, scale, self.index_list)
            # X_concat = np.concatenate(tuple(self.X for n in self.index_list))
            # data_flat = self.data.flatten()

            # self.gp = pm.gp.Marginal(cov_func=scaled_cov)
            # obs = self.gp.marginal_likelihood(
            #         'obs',
            #         X=X_concat,
            #         y=data_flat,
            #         noise=self.noise,
            #         )

            # self.gp = []
            # for n, cn in zip(self.index_list, self.data):
            #     # Create a cov that handles an uncertain expansion parameter
            #     scaled_cov = scale**(-2*n) * cov

            #     # Treat the observed coefficients as draws from a GP
            #     # Constrain the model by the data:
            #     gp_cn = pm.gp.Marginal(cov_func=scaled_cov)
            #     cnobs = gp_cn.marginal_likelihood(
            #         'c{}obs'.format(n),
            #         X=self.X,
            #         y=cn,
            #         noise=self.noise
            #         )

            #     self.gp.append(gp_cn)

        # cov_array = np.array([scale**(-2*n)*cov for n in self.index_list])

        # gp = pm.gp.Marginal(cov_func=cov)
        # cnobs = gp.marginal_likelihood(
        #         'cnobs',
        #         X=self.X,
        #         y=self.data,
        #         # is_observed=False,
        #         noise=self.noise,
        #         shape=self.data.shape
        #         )

        # self.gp = gp

        # names = theano.tensor.vector("names")

        # ([gp, obs, scaled_obs], _) = theano.scan(
        #                                 fn=self.setup_cn,
        #                                 outputs_info=None,
        #                                 sequences=[y, order],
        #                                 non_sequences=[name, X, cov, noise]
        #                                 )

        # # Compile a function
        # setup_ckvec = theano.function(
        #                       inputs=[name, X, y, cov, noise, scale, order],
        #                       outputs=[gp, obs, scaled_obs])

        # name_list = ['c{}'.format(n) for n in self.index_list]

        # X_obs = self.X
        # if self.X_star is not None:
        #     X_obs = self.X_star
        # X_obs_tuple = tuple(X_obs for n in self.index_list)
        # print(X_obs_tuple)
        # X_obs = np.concatenate(X_obs_tuple, axis=1)
        # print(X_obs, X_obs.shape)
        # pm.Deterministic("cn", cnobs)
        # gp.conditional('cn', X_obs, givens={'y': self.data[0]})
        # for i, n in enumerate(self.index_list):
        #     gp.conditional("c{}".format(n), Xnew=X_obs,
        #                    given={'y': self.data[i], 'X': self.X, 'noise': self.noise}
        #                    )

        # self.gp = []

        # for n, cn in zip(self.index_list, self.data):
        #     # Create a cov that handles an uncertain expansion parameter
        #     scaled_cov = scale**(-2*n) * cov

        #     # Treat the observed coefficients as draws from a GP
        #     # Constrain the model by the data:
        #     gp_cn = pm.gp.Marginal(cov_func=scaled_cov)
        #     cnobs = gp_cn.marginal_likelihood(
        #         'c{}obs'.format(n),
        #         X=self.X,
        #         y=cn,
        #         # is_observed=False,
        #         noise=self.noise
        #         )

        #     self.gp.append(gp_cn)

        #     # Scale fixed cnobs due to possibly uncertain expansion parameter
        #     pm.Deterministic("c{}".format(n), scale**n * cnobs)
        #     pm.Deterministic("c{}".format(n), cnobs)


class ExpansionParameterModel(pm.Model):
    """A model for the EFT expansion parameter: Q ~ low_energy_scale / breakdown

    Parameters
    ----------
    breakdown_eval: float
                    The breakdown scale used to extract the observable
                    coefficients
    breakdown_dist: pymc3.distributions object
                    The prior distribution for the breakdown scale
                    (sometimes denoted \Lambda). Must be a distribution! i.e.
                    pm.Lognormal.dist(mu=0, sd=10, testval=600.0)
                    but *not*
                    pm.Lognormal('breakdown', mu=0, sd=10, testval=600.0).
                    Allows the prior to be set up without entering the model
                    context (though that is permitted as well).
                    *Must* contain a testval kwarg to start sampling in a
                    region where you believe the *true* value probably is,
                    otherwise sampling issues can occur!
                    (breakdown_eval is probably a good place to start.)
    name          : str
                    The name of the expansion parameter. This name will be
                    placed before all RV names defined within this model,
                    i.e. 'name_breakdown'.
    model         : pymc3.Model object
                    The parent model. If defined within a model context,
                    it will use that one.
    """

    def __init__(self, breakdown_eval, breakdown_dist=None,
                 name='', model=None, **kwargs):
        super(ExpansionParameterModel, self).__init__(name, model)

        self.breakdown_eval = breakdown_eval
        self.breakdown_dist = breakdown_dist

        if breakdown_dist is not None:
            self.Var('breakdown', self.breakdown_dist)

    # ------------------------
    # RVs with special setters
    # ------------------------

    @property
    def breakdown(self):
        return self._breakdown

    @breakdown.setter
    def breakdown(self, value):
        # Sampling issues can occur if it does not start in reasonable location
        assert value.distribution.testval is not None, \
            "breakdown must be given an appropriate testval. " + \
            "Possibly around breakdown_eval."
        self._breakdown = value
        self.setup_model()

    @property
    def numscale(self):
        return self._numscale

    @numscale.setter
    def numscale(self, value):
        self._numscale = value
        self.setup_model()

    def setup_model(self):
        """Defines the scaling parameter for the coefficients: c_n ~ scale^n.
        Automatically called when breakdown or numscale are set."""
        try:
            bdown = self.breakdown
        except AttributeError:
            bdown = 1
        try:
            nscale = self.numscale
        except AttributeError:
            nscale = 1
        # The scaling factor for coefficients: c_n ~ scale^n
        self.scale = bdown/(self.breakdown_eval * nscale)


class ScaledCov(pm.gp.cov.Covariance):
    """Create a big kronecker product of scale and cov.

    Parameters
    ----------
    cov    : gp.cov.Covariance object
             The covariance that will be scaled
    scale  : RV
             A random variable that will scale cov differently
             in each block of the kronecker product
    powers : list
             The powers of scale that will multiply cov in each
             block of the kronecker product
    """

    def __init__(self, cov, scale, powers):
        super(ScaledCov, self).__init__(input_dim=cov.input_dim,
                                        active_dims=cov.active_dims)
        self.X = []
        self.Xs = []
        self.cov = cov
        self.powers = powers
        self.scales = [scale**(-2*n) for n in powers]
        self.scales_diag = tt.nlinalg.diag(self.scales)

    def diag(self, X):
        """[scales[0] * cov.diag(X), scales[1] * cov.diag(X), ... ].ravel()"""
        X_unique = self.unique_domain(X)
        return tt.outer(self.scales, self.cov.diag(X_unique)).ravel()

    def full(self, X, Xs=None):
        X_unique = self.unique_domain(X)
        Xs_unique = None
        if Xs is not None:
            Xs_unique = self.unique_domain(Xs)
        covfull = self.cov(X_unique, Xs_unique)
        return tt.slinalg.kron(self.scales_diag, covfull)
        # self.update_cov(X, Xs)
        # return self.kron_cov

    def update_cov(self, X, Xs=None):
        # if X != self.X or Xs != self.Xs:
        if not np.array_equal(X, self.X) or not np.array_equal(Xs, self.Xs):
            self.X = X
            self.Xs = Xs
            X_unique = self.unique_domain(X)
            Xs_unique = None
            if Xs is not None:
                Xs_unique = self.unique_domain(Xs)
            covfull = self.cov(X_unique, Xs_unique)
            self.kron_cov = tt.slinalg.kron(self.scales_diag, covfull)

    def unique_domain(self, X):
        unique_length = len(X)//len(self.powers)
        return X[:unique_length]
