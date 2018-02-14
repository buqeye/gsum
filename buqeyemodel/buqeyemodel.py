from __future__ import division
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
import warnings
from pymc3.math import cartesian
import scipy.integrate as integrate
import scipy.stats as st


__all__ = ['ObservableModel', 'ExpansionParameterModel', 'GPCoeffs', 'Exponential']


class Exponential(pm.gp.mean.Mean):

    def __init__(self, b=1, active_dim=0):
        super(Exponential, self).__init__()
        self.b = b
        if not isinstance(active_dim, int):
            raise ValueError('active_dim must be an integer')
        self.active_dim = active_dim

    def __call__(self, X):
        p = X[:, self.active_dim]
        return self.b**p


class ObsCoeffs(object):
    """A base class that examines the convergence pattern of observables

    This simply extracts and sets up the relevant variables to be used
    in the coefficient models.
    """

    def __init__(self, X, obs, orders, rm_orders=None, ref=1, Q_est=1,
                 grid=False, **kwargs):
        self.grid = grid
        if grid:
            self.Xs = X
            self.X = cartesian(*X)
        else:
            self.Xs = None
            self.X = X
        self.N = len(self.X)

        try:
            self.Q_est_func = Q_est
            self.Q_est = Q_est(self.X)
        except TypeError:
            self.Q_est = Q_est * np.ones(len(self.X))
            self.Q_est_func = None

        assert len(obs) == len(orders), \
            "Orders must match the number of coefficients"
        if rm_orders is None:
            rm_orders = []

        self.orders = []
        self.cs = []
        for i, n in enumerate(orders):
            if i == 0:
                cn = obs[i] / (ref * self.Q_est**n)
            else:
                cn = (obs[i] - obs[i-1]) / (ref * self.Q_est**n)

            if n not in rm_orders:
                self.orders.append(n)
                self.cs.append(cn)
        self.orders = np.array(self.orders)
        self.cs = np.array(self.cs)
        self.k = self.orders[-1]
        self.obsk = np.array(obs[self.k])
        self.ref = ref


class GPCoeffs(pm.Model, ObsCoeffs):
    """Treats observables as sums of weighted iid Gaussian processes.

    Parameters for the Gaussian process are conditioned on observable data to
    permit the extimation of the truncation error.

    Parameters
    ----------
    name : str
        The name of the observable. This name will be
        placed before all RV names defined within this model,
        i.e. 'name_sd'.
    model : pymc3.Model object
        The parent model. If defined within a model context,
        it will use that one.
    """

    def __init__(self, X, obs, orders, rm_orders=None, ref=1, Q_est=1,
                 grid=False, name='', model=None, build=True,
                 **param_kwargs):
        # Can't use super since pm.Model doesn't accept kwargs
        # super(GPCoeffs, self).__init__(name=name, model=model)
        pm.Model.__init__(self, name=name, model=model)
        ObsCoeffs.__init__(
            self, X=X, obs=obs, orders=orders, rm_orders=rm_orders,
            ref=ref, Q_est=Q_est, grid=grid)
        if build:
            self.def_params(**param_kwargs)
            self.gp_model()

    def get_RV(self, rv):
        """Check both self and parent model for rv"""
        try:
            var = getattr(self, rv)
        except AttributeError:
            var = getattr(self.root, rv)
        return var

    def def_params(self, mu=0.0, sd=1.0, ls=1.0, sigma=1e-5, q=1.0):
        # Add a shape key to ls dict for anisotropic models
        if isinstance(ls, dict) and 'shape' not in ls:
            ls_size = 0
            try:
                ls_size = ls['mu'].shape[0]
            except (AttributeError, KeyError):
                pass
            try:
                sd_size = ls['sd'].shape[0]
            except (AttributeError, KeyError):
                pass
            else:
                if sd_size > ls_size:
                    ls_size = sd_size
            if ls_size > 1:
                ls['shape'] = ls_size

        # Convert non-dicts to dicts holding observed value
        args = [mu, sd, ls, sigma, q]
        for i, arg in enumerate(args):
            if not isinstance(arg, dict):
                args[i] = {'observed': arg}
        [mu, sd, ls, sigma, q] = args

        # Entering self ensures RVs are in the context of this instance
        # even if this method is called manually
        with self:
            pm.Normal('mu', **mu)
            pm.Lognormal('sd', **sd)
            pm.Lognormal('ls', **ls)
            pm.HalfNormal('sigma', **sigma)
            q = pm.Lognormal('q', **q)
            pm.Deterministic('Q', q * self.Q_est)
        return self

    def gp_model(self):
        with self:
            # Get predefined parameters
            mu = self.get_RV('mu')
            sd = self.get_RV('sd')
            ls = self.get_RV('ls')
            sigma = self.get_RV('sigma')
            try:
                q = self.get_RV('q')
            except AttributeError:  # If q hasn't been defined
                q = 1.0
                self.q = q

            # Define Q if it has been overlooked
            try:
                pm.Deterministic('Q', q * self.Q_est)
            except AttributeError:  # If q isn't an RV
                self.Q = tt.as_tensor_variable(q * self.Q_est)
            except ValueError:  # If Q is already defined
                pass

            # Setup mean and covariance functions
            orders = self.orders
            mean = pm.gp.mean.Constant(mu) * Exponential(b=q)
            B = tt.nlinalg.diag(q**(2*orders))
            coregion = pm.gp.cov.Coregion(1, B=B)
            covs = [coregion]
            if self.grid:  # Create cov for each grid entry
                Xgp = [orders[:, None], *self.Xs]
                num_Xs = len(self.Xs)
                for i, X in enumerate(self.Xs):
                    dim = X.shape[1]
                    try:
                        ls_i = ls[i]
                    except IndexError:  # Smaller length than Xs
                        print("Anisotropic models must have a distinct",
                              "length scale for each entry in Xs")
                        raise
                    except (TypeError, ValueError):  # Is scalar-like
                        ls_i = ls
                    cov = sd**(2.0/num_Xs) * pm.gp.cov.ExpQuad(dim, ls=ls_i)
                    covs.append(cov)
                ccov = pm.gp.cov.Kron(covs[1:])
            else:  # Create one cov
                Xgp = [orders[:, None], self.X]
                dim = self.X.shape[1]
                ccov = sd**2 * pm.gp.cov.ExpQuad(dim, ls=ls)
                covs.append(ccov)

            # Make Gaussian process and condition on data
            y = self.cs.ravel()
            gp = pm.gp.MarginalKron(mean_func=mean, cov_funcs=covs)
            gp.marginal_likelihood('cs_observed', Xs=Xgp, y=y, sigma=sigma)

        # For user access
        self.mean = mean
        self.covs = covs
        self.ccov = ccov
        self.gp = gp
        return self

    def setup_Deltak(self):
        # Get relevant variables
        k = self.orders[-1]
        mu = self.get_RV('mu')
        sd = self.get_RV('sd')
        ls = self.get_RV('ls')
        sigma = self.get_RV('sigma')
        Q = self.get_RV('Q')
        Q_est = self.Q_est

        # Set up variance matrix due to Q array
        rows, cols = tt.mgrid[0:self.N, 0:self.N]
        Qr, Qc = Q[rows], Q[cols]
        varQ = (Qr * Qc)**(k+1) / (1 - Qr * Qc)

        # Define variables for the Deltak process
        Dk_mean = pm.gp.mean.Constant(mu * Q**(k+1) / (1-Q))
        Dk_cov = varQ * self.ccov
        Dk_sigma = sigma * Q_est**(k+1) / tt.sqrt(1-Q_est**2)

        with self:  # Ensure Deltak belongs to this model context
            # In general, Q variance removes any potential Kronecker structure
            Dk_gp = pm.gp.Marginal(mean_func=Dk_mean, cov_func=Dk_cov)
            Dk = Dk_gp.marginal_likelihood(
                'Dk', X=self.X, y=None, noise=Dk_sigma, is_observed=False)
        return self


class InCoeffs(ObsCoeffs):
    """The observables are treated as sums of weighted Gaussian random variables.

    [description]
    """

    def __init__(self, X, obs, orders, rm_orders=None, ref=1, Q_est=1,
                 grid=False):
        super(InCoeffs, self).__init__(
            X=X, obs=obs, orders=orders, rm_orders=rm_orders, ref=ref,
            Q_est=Q_est, grid=grid)
        self.cksq = np.sum(self.cs**2, axis=0, dtype=float)
        self.qsq = self.Q_est**(2*self.k + 2) / (1.0 - self.Q_est**2)
        self.num_c = len(self.cs)

    def _a_n(self, a):
        return a + self.num_c/2.0

    def _b_n(self, b, data=None):
        if data is None:
            data = self.cksq
        return b + data/2.0

    def var_dist(self, a, b):
        return st.invgamma(a=self._a_n(a), scale=self._b_n(b))

    def error_dist(self, a, b, rescale=True):
        qsq = self.qsq
        a_n, b_n = self._a_n(a), self._b_n(b)
        loc = 0
        scale = np.sqrt(b_n * qsq / a_n)
        if rescale:
            loc = self.obsk
            scale *= self.ref
        return st.t(df=2*a_n, loc=loc, scale=scale)

    def error_pdf(self, Deltak, a, b, rescale=True):
        Dk = np.atleast_2d(Deltak).T
        dist = self.error_dist(a, b, rescale=rescale)
        return dist.pdf(Dk).T

    def error_interval(self, alpha, a, b, rescale=True):
        """Returns the centered degree of belief interval for Delta_k

        Parameters
        ----------
        alpha : float or ndarray
            The specifies the 100*alpha% interval
        a : float
            The hyperparameter of the variance prior `InvGam(a, b)`
        b : float
            The hyperparameter of the variance prior `InvGam(a, b)`
        rescale : bool, optional
            Whether or not the dimensionless error is scaled relative to the
            highest order observable obs_k, that is, obs_k + ref * error, the
            default is True

        Returns
        -------
        tuple
            The lower and upper bounds of the error interval
        """
        dist = self.error_dist(a, b, rescale=rescale)
        low, up = dist.interval(alpha=alpha)
        # if rescale:
        #     obsk, ref = self.obsk, self.ref
        #     low, up = obsk + ref*low, obsk + ref*up
        return low, up

    def fQ_prior_logpdf(self, fQ, a_fQ, b_fQ, scale=1.0, inverted=False):
        # if inverted:
        #     prior = st.invgamma(a=a_fQ, scale=b_fQ)
        # else:
        #     # Use rate param for consistency: b=1/scale
        #     scale = 1.0/b_fQ
        #     prior = st.gamma(a=a_fQ, scale=scale)
        # return prior
        if inverted:
            logpdf = st.beta.logpdf(1/fQ, a=a_fQ, b=b_fQ, scale=1/scale)
            logpdf -= 2*np.log(fQ)
        else:
            logpdf = st.beta.logpdf(fQ, a=a_fQ, b=b_fQ, scale=scale)
        return logpdf

    def fQ_ulogpdf(self, fQ, a, b, a_fQ, b_fQ, scale=1.0, inverted=False,
                   combine=True):
        """The unnormalized logpdf for the expansion parameter.

        See the docstring for expar_pdf.
        """
        # Handle scaling and inverting
        # x = x/scale
        # x = np.atleast_2d(x).T
        Q = fQ / scale
        if inverted:
            Q = Q**(-1)
        # if np.any(Q >= 1):
        #     raise ValueError('Q must be between 0 and 1')
        Q = np.atleast_2d(Q).T

        # Set up terms
        orders, cs, Q_est = self.orders, self.cs, self.Q_est
        cs_Q = np.array([cn*(Q_est/Q)**n for n, cn in zip(orders, cs)])
        cksq_Q = np.sum(cs_Q**2, axis=0)

        a_n = self._a_n(a)
        b_n = self._b_n(b, data=cksq_Q)
        logp = - a_n * np.log(b_n) - np.sum(orders) * np.log(Q)
        log_prior = self.fQ_prior_logpdf(fQ, a_fQ, b_fQ, scale, inverted)
        if combine:
            logp = np.sum(logp, axis=1)
            logp += log_prior
        else:
            logp += np.atleast_2d(log_prior).T
            logp = logp.T
        return logp

    def fQ_pdf(self, fQ, a, b, a_fQ, b_fQ, scale=1.0, inverted=False,
               combine=True):
        """The approximately normalized pdf for the expansion parameter.

        The pdf is normalized using the trapezoid rule based on the input
        points fQ. Hence fine grids that capture the probability mass will
        be accurately normalized.

        Parameters
        ----------
        fQ : ndarray
            A function f of the expansion parameter Q: f(Q)
        a : float
            [description]
        b : float
            [description]
        a_expar : float
            [description]
        b_expar : float
            [description]
        scale : float, optional
            [description] the default is 1.0
        inverted : bool, optional
            [description] (the default is False, which [default_description])
        combine : bool, optional
            Whether the obserables should use a common expansion parameter, the
            default is True

        Returns
        -------
        ndarray
            The pdf for f(Q)
        """
        logp = self.fQ_ulogpdf(fQ, a, b, a_fQ, b_fQ, scale=scale,
                               inverted=inverted, combine=combine)
        # Reduce underflow and overflow
        maxlogp = np.atleast_2d(np.max(logp, axis=-1)).T
        logp = logp - maxlogp
        pdf = np.exp(logp)

        # Integrate using trapezoid rule
        norm = np.atleast_2d(np.trapz(pdf, fQ)).T
        return np.squeeze(pdf/norm)

    def fQ_interval(self, alpha, fQ, a, b, a_fQ, b_fQ, scale=1.0,
                    inverted=False, combine=True):
        pdf = self.fQ_pdf(fQ, a, b, a_fQ, b_fQ, scale=scale,
                          inverted=inverted, combine=combine)
        cdf = integrate.cumtrapz(pdf, x=fQ, initial=0)

        # Invert cdf
        alpha = np.asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1, q2 = (1.0-alpha)/2, (1.0+alpha)/2
        low = (np.abs(cdf-q1)).argmin(axis=-1)
        up = (np.abs(cdf-q2)).argmin(axis=-1)
        return fQ[low], fQ[up]
