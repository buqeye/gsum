import pymc3 as pm
import numpy as np
import theano.tensor as tt
import warnings


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

        scale = 1
        if self.expansion_parameter is not None:
            scale = self.expansion_parameter.scale

        for n, cn in zip(self.index_list, self.data):
            # Create a cov that handles an uncertain expansion parameter
            scaled_cov = scale**(-2*n) * cov

            # Treat the observed coefficients as draws from a GP
            # Constrain the model by the data:
            gp = pm.gp.Marginal(cov_func=scaled_cov)
            cnobs = gp.marginal_likelihood(
                'c{}obs'.format(n),
                X=self.X,
                y=cn,
                noise=self.noise
                )

            # Scale fixed cnobs due to possibly uncertain expansion parameter
            pm.Deterministic("c{}".format(n), scale**n * cnobs)


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

        if breakdown_dist is not None:
            self.Var('breakdown', breakdown_dist)
            self.setup_model(self.breakdown)

    def setup_hyperparameter(self, **kwargs):
        temp_kwargs = kwargs
        dist = temp_kwargs.pop("dist")
        return dist(**temp_kwargs)

    def setup_model(self, breakdown):
        # Sampling issues can occur if it does not start in reasonable location
        if breakdown.distribution.testval is None:
            raise AttributeError(
                    "breakdown must be given an appropriate testval. " +
                    "Possibly around breakdown_eval."
                    )
        # The scaling factor for coefficients: c_n ~ scale^n
        self.scale = breakdown/self.breakdown_eval
        return self.scale
