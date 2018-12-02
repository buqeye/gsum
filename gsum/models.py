from __future__ import division
from functools import reduce
from .helpers import coefficients, predictions, gaussian, stabilize, \
    cartesian, HPD, KL_Gauss, default_attributes, cholesky_errors, mahalanobis, lazy_property
from .cutils import pivoted_cholesky
import numpy as np
import pymc3 as pm
import scipy as sp
import scipy.integrate as integrate
import scipy.stats as st
from statsmodels.sandbox.distributions.mv_normal import MVT
import theano
import theano.tensor as tt
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import wraps
from cycler import cycler
from itertools import cycle


__all__ = [
    'SGP', 'PowerProcess', 'PowerSeries', 'ConjugateProcess',
    'ConjugateGaussianProcess', 'ConjugateStudentProcess', 'Diagnostic',
    'GraphicalDiagnostic']




class ConjugateProcess:
    
    def __init__(self, corr_kernel, m0=0, v0=1, a0=1, b0=1, sd=None):
        """A conjugate Gaussian Process model
        
        Parameters
        ----------
        corr_kernel : callable
            The kernel for the correlation matrix
        m0 : float
            The mean hyperparameter for the normal prior placed on the mean
        v0 : float
            The variance hyperparameter for the normal prior placed on the mean
        a0 : float > 0
            The shape hyperparameter for the inverse gamma prior placed on sd**2
        b0 : float > 0
            The scale hyperparameter for the inverse gamma prior placed on sd**2
        """
        self.m0 = m0
        self.v0 = v0
        self.a0 = a0
        self.b0 = b0
        self.corr_kernel = corr_kernel
        self.X = None
        self.y = None
        self._sd = sd
        self.corr_kwargs = {}
        self._corr_chol = None
        self.noise_sd = 1e-7
        
    def _recompute_corr(self, **corr_kwargs):
        # Must be non-empty and not equal to the defaults
        return corr_kwargs and corr_kwargs != self.corr_kwargs
    
    def cleanup(self):
        """Removes all attributes except those set at initialization"""
        def attr_name(name):
            return '_cache_' + name
        for attr in ['y', 'X', 'corr']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        for attr in ['m', 'v', 'a', 'b', 'sd']:
            try:
                delattr(self, attr_name(attr))
            except AttributeError:
                pass
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def m(self, y=None, corr_chol=None):
        """The posterior mean hyperparameter given y for the normal prior placed on the GP mean"""
        # Mean is not updated if its prior variance is zero (i.e. delta function prior)
        # Do by hand to prevent dividing by zero
        if self.v0 == 0:
            return self.m0

        y_avg = y
        if y.ndim == 2:
            y_avg = np.average(y, axis=0)
        ny = self.num_y(y)
        one = np.ones_like(y_avg)
        # print(y_avg.shape, corr_chol.shape)
        right_half = cholesky_errors(y_avg, 0, corr_chol)
        left_half = cholesky_errors(one, 0, corr_chol)
        # if left_half.ndim > 1:
        #     left_half = np.swapaxes(left_half, -1, -2)
        v = self.v(y=y, corr_chol=corr_chol)
        return v * (self.m0 / self.v0 + ny * np.sum(left_half * right_half, axis=-1))
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def v(self, y=None, corr_chol=None):
        """The posterior variance hyperparameter for the normal prior placed on the mean"""
        # If prior variance is zero, it stays zero
        # Do by hand to prevent dividing by zero
        if self.v0 == 0:
            return 0.

        ny = self.num_y(y)
        one = np.ones(corr_chol.shape[-1])
        quad = mahalanobis(one, 0, corr_chol) ** 2
        return (1. / self.v0 + ny * quad) ** (-1)
    
    @default_attributes(y='y')
    def a(self, y=None):
        """The posterior shape hyperparameter for the inverse gamma prior placed on sd**2"""
        return self.a0 + y.size / 2.
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def b(self, y=None, corr_chol=None):
        """The posterior scale hyperparameter for the inverse gamma prior placed on sd**2"""
        mean_terms = 0
        if self.v0 != 0:
            m = self.m(y=y, corr_chol=corr_chol)
            v = self.v(y=y, corr_chol=corr_chol)
            mean_terms = self.m0**2 / self.v0 - m**2 / v
        quad = mahalanobis(y, 0, corr_chol)**2
        # sum over y axes, but not extra corr axes
        if np.squeeze(y).ndim > 1:
            quad = np.sum(quad, axis=-1)
        return self.b0 + 0.5 * (mean_terms + quad)
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def sd(self, y=None, corr_chol=None, broadcast=False):
        """The mean value for the marginal standard deviation given y.
        
        It turns out that for both the GP and TP, `sd**2` is the conversion factor to go
        from the correlation matrix to the covariance matrix.
        
        Note: if the correlation matrix does equal 1 when `X == Xp`, `sd` **will not**
        be the standard deviation at `X`. Instead, one must look at `cov` directly.
        """
        sd = self._sd
        if sd is None:
            b = self.b(y=y, corr_chol=corr_chol)
            a = self.a(y=y)
            sd = np.sqrt(b / (a - 1))
        if broadcast:  # For when a set of corr_chols are given
            sd = np.atleast_1d(sd)[:, None, None]
        return sd

    @default_attributes(X='X', y='y', corr_chol='_corr_chol')
    def mean(self, X=None, y=None, corr_chol=None):
        """The MAP value for the mean given y"""
        m = np.atleast_1d(self.m(y=y, corr_chol=corr_chol))[:, None]
        return np.squeeze(m * np.ones(len(X)))

    @default_attributes(X='X', y='y', noise_sd='noise_sd', kwargs='corr_kwargs')
    def cov(self, X=None, Xp=None, y=None, noise_sd=None, **kwargs):
        if Xp is None:
            Xp = X
        corr = self.corr_kernel(X, Xp, **kwargs)
        corr_chol = self.corr_chol(noise_sd=noise_sd, **kwargs)  # use X from fit
        sd = self.sd(y=y, corr_chol=corr_chol, broadcast=True)
        return np.squeeze(sd**2 * corr)
    
    @default_attributes(X='X', noise_sd='noise_sd', kwargs='corr_kwargs')
    def corr_chol(self, X=None, noise_sd=None, **kwargs):
        attr = '_corr_chol'
        corr = self.corr_kernel(X, X, **kwargs)
        chol = np.linalg.cholesky(corr + noise_sd**2 * np.eye(len(X)))
        return chol
    
    @staticmethod
    def num_y(y):
        ny = 1
        if y.ndim == 2:
            ny = y.shape[0]
        return ny
    
    def fit(self, X, y, noise_sd=1e-7, **kwargs):
        """Fits the GP, i.e., sets/updates all hyperparameters, given y(X)"""
        # self.cleanup()
        self.X = X
        self.y = y
        self.corr_kwargs = kwargs
        self.noise_sd = noise_sd
        self.corr = self.corr_kernel(X, **kwargs)
        self._corr_chol = self.corr_chol(X, noise_sd=noise_sd, **kwargs)
    
    @default_attributes(y='y')
    def predict(self, Xnew, return_std=False, return_cov=False, y=None, pred_noise=True):
        """Returns the predictive GP at unevaluated points Xnew"""
        kwargs = self.corr_kwargs
        # corr_chol = self.corr_chol(**kwargs)
        corr_chol = self._corr_chol
        # Use y from fit for hyperparameters
        m_old = self.mean(y=self.y, corr_chol=corr_chol)
        m_new = self.mean(Xnew, y=self.y, corr_chol=corr_chol)
        R_on = self.corr_kernel(self.X, Xnew, **kwargs)
        R_no = R_on.T
        R_nn = self.corr_kernel(Xnew, Xnew, **kwargs)

        # Use given y for prediction
        mfilter = np.dot(R_no, sp.linalg.cho_solve((corr_chol, True), (y - m_old).T)).T
        m_pred = m_new + mfilter
        if return_std or return_cov:
            half_quad = sp.linalg.solve_triangular(corr_chol, R_on, lower=True)
            R_pred = R_nn - np.dot(half_quad.T, half_quad)
            if pred_noise:
                R_pred += self.noise_sd**2 * np.eye(len(Xnew))
            # Use y from fit for hyperparameters
            sd = self.sd(y=self.y, corr_chol=corr_chol, broadcast=True)
            K_pred = np.squeeze(sd**2 * R_pred)
            if return_std:
                return m_pred, np.sqrt(np.diag(K_pred))
            return m_pred, K_pred
        return m_pred
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def likelihood(self, log=True, y=None, corr_chol=None):
        raise NotImplementedError

    def ratio_likelihood(self, ratio, y, corr_chol, orders=None):
        if y.ndim < 2:
            raise ValueError('y must be at least 2d, not {}'.format(y.shape))
        if orders is None:
            orders = np.arange(y.shape[0])
        ys = y / ratio[:, None, ...] ** orders[:, None]
        loglikes = np.array([self.likelihood(log=True, y=yi, corr_chol=corr_chol) for yi in ys])
        loglikes -= np.sum(orders) * np.sum(np.log(ratio), axis=-1)[:, None]
        return loglikes

        
class ConjugateGaussianProcess(ConjugateProcess):
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def likelihood(self, log=True, y=None, corr_chol=None):
        # Multiple corr_chols can be passed to quickly get likelihoods for many correlation parameters
        if corr_chol.ndim == 2:
            corr_chol = corr_chol[None, :, :]
        n = corr_chol.shape[0]
        
        # Setup best guesses for mean and cov
        means = np.atleast_2d(self.mean(y=y, corr_chol=corr_chol))
        sd = self.sd(y=y, corr_chol=corr_chol, broadcast=True)
        corrs = corr_chol @ np.swapaxes(corr_chol, -2, -1)
        covs = sd**2 * corrs

        loglikes = np.zeros(n)
        for i in range(n):
            dist = st.multivariate_normal(mean=means[i], cov=covs[i])
            loglikes[i] = np.sum(dist.logpdf(y))
        loglikes = np.squeeze(loglikes)
        if log:
            return loglikes
        return np.exp(loglikes)

    
class ConjugateStudentProcess(ConjugateProcess):
    
    @default_attributes(y='y', corr_chol='_corr_chol')
    def likelihood(self, log=True, y=None, corr_chol=None):
        mean = self.mean(y=y, corr_chol=corr_chol)
        a0, a = self.a0, self.a(y=y)
        b0, b = self.b0, self.b(y=y, corr_chol=corr_chol)
        v0, v = self.v0, self.v(y=y, corr_chol=corr_chol)
        ny = self.num_y(y)
        N = chol.shape[-1]
        
        def log_nig_norm(aa, bb, vv):
            """Normalization of the normal inverse gamma distribution"""
            val = loggamma(aa) - aa * np.log(bb)
            if vv != 0:
                val += np.log(np.sqrt(2*np.pi*vv))
            return val
        
        tr_log_corr = 2 * np.sum(np.log(np.diagonal(corr_chol, axis1=-2, axis2=-1)), axis=-1)
        loglike = log_nig_norm(a, b, v) - log_nig_norm(a0, b0, v0) - ny / 2. * tr_log_corr
        loglike -= ny * N / 2. * np.log(2*np.pi)
        if log:
            return loglike
        return np.exp(loglike)

    
# def ratio_likelihood(ratio, process, y, corr_chol, orders=None):
#     if y.ndim < 2:
#         raise ValueError('y must be at least 2d, not {}'.format(y.shape))
#     if orders is None:
#         orders = np.arange(y.shape[0])
#     # ratios = ratio[:, None, ...]
#     ys = y / ratio[:, None, ...] ** orders[:, None]
#     loglikes = np.array([process.likelihood(log=True, y=yi, corr_chol=corr_chol) for yi in ys])
#     loglikes -= np.sum(orders) * np.sum(np.log(ratios), axis=-1)[:, None]
#     return loglikes

class Diagnostic:
    R"""A class for quickly testing model checking methods discussed in Bastos & O'Hagan.
    
    """
    
    def __init__(self, mean, cov, df=None):
        self.mean = mean
        self.cov = cov
        self.sd = sd = np.sqrt(np.diag(cov))
        if df is None:
            self.dist = st.multivariate_normal(mean=mean, cov=cov)
            self.udist = st.norm(loc=mean, scale=sd)
            self.std_udist = st.norm(loc=0., scale=1.)
        else:
            sigma = cov * (df - 2) / df
            self.dist = MVT(mean=mean, sigma=sigma, df=df)
            self.udist = st.t(loc=mean, scale=sd, df=df)
            self.std_udist = st.t(loc=0., scale=1., df=df)
        
    @lazy_property
    def chol(self):
        """Returns the lower cholesky matrix G where cov = G.G^T"""
        return np.linalg.cholesky(self.cov)
    
    @lazy_property
    def pivoted_chol(self):
        """Returns the pivoted cholesky matrix G where cov = G.G^T"""
        return pivoted_cholesky(self.cov)
    
    @lazy_property
    def eig(self):
        """Returns the eigen-docomposition matrix G where cov = G.G^T"""
        e, v = np.linalg.eigh(self.cov)
        ee = np.diag(np.sqrt(e))
        return np.dot(v, ee)

    def samples(self, n):
        return self.dist.rvs(n)
    
    def individual_errors(self, y):
        return (y - self.mean) / np.sqrt(np.diag(self.cov))
    
    def cholesky_errors(self, y):
        return cholesky_errors(y, self.mean, self.chol)
    
    def pivoted_cholesky_errors(self, y):
        return np.linalg.solve(self.pivoted_chol, (y - self.mean).T).T
    
    def eigen_errors(self, y):
        return np.linalg.solve(self.eig, (y - self.mean).T).T
    
    def chi2(self, y):
        return np.sum(self.indiv_errors(y), axis=-1)

    def md(self, y):
        R"""The Mahalanobis distance"""
        return mahalanobis(y, self.mean, self.chol)

    def kl(self, mean, cov):
        R"""The Kullbeck-Leibler divergence"""
        m1, c1, chol1 = self.mean, self.cov, self.chol
        m0, c0 = mean, cov
        tr = np.trace(sp.linalg.cho_solve((chol1, True), c0))
        dist = self.md(m0) ** 2
        k = c1.shape[-1]
        logs = 2*np.sum(np.log(np.diag(c1))) - np.linalg.slogdet(c0)[-1]
        return 0.5 * (tr + dist - k + logs)
    
    def credible_interval(self, y, intervals):
        """The credible interval diagnostic.
        
        Parameters
        ----------
        y : (n_c, d) shaped array
        intervals : 1d array
            The credible intervals at which to perform the test
        """
        lower, upper = self.udist.interval(np.atleast_2d(intervals).T)
        
        def diagnostic(data, lower, upper):
            indicator = (lower < data) & (data < upper)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)   # The diagnostic

        dci = np.apply_along_axis(
                diagnostic, axis=1, arr=np.atleast_2d(y), lower=lower,
                upper=upper)
        dci = np.squeeze(dci)
        return dci



class GraphicalDiagnostic:
    
    def __init__(self, diagnostic, data, nref=1000, colors=None):
        self.diagnostic = diagnostic
        self.data = data
        self.samples = self.diagnostic.samples(nref)
        if colors is None:
            # The standard Matplotlib 2.0 colors, or whatever they've been updated to be.
            clist = list(mpl.rcParams['axes.prop_cycle'])
            colors = [c['color'] for c in clist]
        self.colors = colors
        self.color_cycle = cycler('color', colors)
    
    def error_plot(self, err, title=None, ylabel=None, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axhline(0, 0, 1, linestyle='-', color='k', lw=1)
        # The standardized 2 sigma bands since the sd has been divided out.
        sd = self.diagnostic.std_udist.std()
        ax.axhline(-2*sd, 0, 1, linestyle='--', color='gray')
        ax.axhline(2*sd, 0, 1, linestyle='--', color='gray')
        ax.set_prop_cycle(self.color_cycle)
        ax.plot(np.arange(self.data.shape[-1]), err.T, ls='', marker='o')
        ax.set_xlabel('index')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        return ax
    
    def individual_errors(self, ax=None):
        err = self.diagnostic.individual_errors(self.data)
        return self.error_plot(err, title='Individual Errors', ax=ax)
    
    def individual_errors_qq(self, ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.individual_errors,
                       title='Individual QQ Plot', ax=ax)
    
    def cholesky_errors(self, ax=None):
        err = self.diagnostic.cholesky_errors(self.data)
        return self.error_plot(err, title='Cholesky Decomposed Errors', ax=ax)

    def cholesky_errors_qq(self, ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.cholesky_errors,
                       title='Cholesky QQ Plot', ax=ax)
    
    def pivoted_cholesky_errors(self, ax=None):
        err = self.diagnostic.pivoted_cholesky_errors(self.data)
        return self.error_plot(err, title='Pivoted Cholesky Decomposed Errors', ax=ax)
    
    def pivoted_cholesky_errors_qq(self, ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.pivoted_cholesky_errors,
                       title='Pivoted Cholesky QQ Plot', ax=ax)
    
    def eigen_errors(self, ax=None):
        err = self.diagnostic.eigen_errors(self.data)
        return self.error_plot(err, title='Eigen Decomposed Errors', ax=ax)
    
    def eigen_errors_qq(self, ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.eigen_errors,
                       title='Eigen QQ Plot', ax=ax)
    
    def hist(self, data, ref, title=None, xlabel=None, ylabel=None, vlines=True, ax=None):
        ref_stats = st.describe(ref)
        ref_sd = np.sqrt(ref_stats.variance)
        ref_mean = ref_stats.mean

        if ax is None:
            ax = plt.gca()
        ax.hist(ref, density=1, label='ref', histtype='step', color='k')
        # ax.vlines([ref_mean - ref_sd, ref_mean + ref_sd], 0, 1, color='gray',
        #           linestyle='--', transform=ax.get_xaxis_transform(), label='68%')
        ax.axvline(ref_mean - ref_sd, 0, 1, color='gray', linestyle='--', label='68%')
        ax.axvline(ref_mean + ref_sd, 0, 1, color='gray', linestyle='--')
        if vlines:
            for c, d in zip(cycle(self.color_cycle), np.atleast_1d(data)):
                ax.axvline(d, 0, 1, zorder=50, **c)
        else:
            ax.hist(data, density=1, label='data', histtype='step')
        ax.legend()
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return ax
    
    def qq(self, data, ref, band_perc, func, title=None, ax=None):
        data = np.sort(func(data.copy()), axis=-1)
        ref = np.sort(func(ref.copy()), axis=-1)
        bands = np.array([np.percentile(ref, [100*(1.-bi)/2, 100*(1.+bi)/2], axis=0)
                          for bi in band_perc])
        n = data.shape[-1]
        quants = (np.arange(1, n+1) - 0.5) / n
        q_theory = self.diagnostic.std_udist.ppf(quants)
        
        if ax is None:
            ax = plt.gca()
        ax.set_prop_cycle(self.color_cycle)
        
        for i in range(len(band_perc)-1, -1, -1):
            ax.fill_between(q_theory, bands[i, 0], bands[i, 1], alpha=0.5, color='gray')

        ax.plot(q_theory, data.T)
        yl, yu = ax.get_ylim()
        xl, xu = ax.get_xlim()
        ax.plot([xl, xu], [xl, xu], c='k')
        ax.set_ylim([yl, yu])
        ax.set_xlim([xl, xu])
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Quantiles')
        return ax
    
    def md(self, vlines=True, ax=None):
        if ax is None:
            ax = plt.gca()
        md_data = self.diagnostic.md(self.data)
        md_ref = self.diagnostic.md(self.samples)
        return self.hist(md_data, md_ref, title='Mahalanobis Distance',
                         xlabel='MD', ylabel='density', vlines=vlines, ax=ax)
    
    def kl(self, X, gp, predict=False, vlines=True, ax=None):
        if ax is None:
            ax = plt.gca()
        ref_means = []
        ref_covs = []
        for i, sample in enumerate(self.samples):
            gp.fit(X, sample)
            if predict:
                mean, cov = gp.predict(X, return_cov=True)
            else:
                mean, cov = gp.mean(X), gp.cov(X)
            ref_means.append(mean)
            ref_covs.append(cov)
            
        data_means = []
        data_covs = []
        for i, data in enumerate(np.atleast_2d(self.data)):
            gp.fit(X, data)
            if predict:
                mean, cov = gp.predict(X, return_cov=True)
            else:
                mean, cov = gp.mean(X), gp.cov(X)
            data_means.append(mean)
            data_covs.append(cov)
        
        kl_ref = [self.diagnostic.kl(mean, cov) for mean, cov in zip(ref_means, ref_covs)]
        kl_data = [self.diagnostic.kl(mean, cov) for mean, cov in zip(data_means, data_covs)]
        return self.hist(kl_data, kl_ref, title='KL Divergence',
                         xlabel='KL', ylabel='density', vlines=vlines, ax=ax)
    
    def credible_interval(self, intervals, band_perc, ax=None):
        dci_data = self.diagnostic.credible_interval(self.data, intervals)
        dci_ref = self.diagnostic.credible_interval(self.samples, intervals)
        bands = np.array([np.percentile(dci_ref, [100*(1.-bi)/2, 100*(1.+bi)/2], axis=0)
                          for bi in band_perc])
        if ax is None:
            ax = plt.gca()
        for i in range(len(band_perc)-1, -1, -1):
            ax.fill_between(intervals, bands[i, 0], bands[i, 1], alpha=0.5, color='gray')
        
        ax.plot([0, 1], [0, 1], c='k')
        ax.set_prop_cycle(self.color_cycle)
        ax.plot(intervals, dci_data.T)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Empirical Coverage')
        ax.set_xlabel('Credible Interval')
        return ax
    
    def plotzilla(self, X, gp=None, predict=False, vlines=True):
        if gp is None:
            pass
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        self.md(vlines=vlines, ax=axes[0, 0])
        self.kl(X, gp, predict, vlines=vlines, ax=axes[0, 1])
        self.credible_interval(np.linspace(0, 1, 101), [0.68, 0.95], axes[0, 2])
        self.individual_errors(axes[1, 0])
        self.individual_errors_qq(axes[2, 0])
        self.cholesky_errors(axes[1, 1])
        self.cholesky_errors_qq(axes[2, 1])
        self.eigen_errors(axes[1, 2])
        self.eigen_errors_qq(axes[2, 2])
        self.pivoted_cholesky_errors(axes[3, 0])
        self.pivoted_cholesky_errors_qq(axes[3, 1])
        fig.tight_layout()
        return fig, axes



















































class SGP(object):
    R"""A semiparametric Gaussian process class

    Treats a function :math:`y` as a Gaussian process

    .. math::

        y(x) | \beta, \sigma^2, \psi \sim N[m(x), \sigma^2 R(x,x; \psi)]

    with a parameterized mean function :math:`m` and correlation function
    :math:`R`. The mean at any input point is given by

    .. math::

        m(x) = h(x)^T \beta

    where :math:`h: \mathbb{R}^d \to \mathbb{R}^q` is a given basis function
    and :math:`\beta` is a :math:`q\times 1` vector of random variables.
    The variance is split into a marginal part :math:`\sigma^2` and a
    correlation function :math:`R` that may further depend on parameters
    :math:`\psi`, e.g., length scales.
    A normal-inverse-gamma prior is placed on :math:`\beta` and
    :math:`\sigma^2`

    .. math::

        \beta, \sigma^2 \sim NIG(\mu, V, a, b)

    The means and covariance of the normal prior placed on :math:`\beta` are
    :math:`\mu` and :math:`\sigma^2 V`, while the shape and scale parameter of
    the inverse gamma prior placed on :math:`\sigma^2` are :math:`a` and
    :math:`b`.

    Parameters
    ----------
    dim : scalar, optional
        The dimension of the ``means`` vector and the ``cov`` matrix, which
        determines how many mean variables are undetermined in the linear
        model. Must be greater than 0. By default, ``dim`` is inferred by
        the columns of the ``basis`` callable.
    basis : callable, optional
        The basis function for the mean vector.
    means : (dim,) array, optional
    cov : (dim,dim) array, optional
    shape : scalar, optional
    scale : scalar, optional
    corr : callable, optional
    """

    def __init__(self, dim=None, basis=None, means=None, cov=None, shape=None,
                 scale=None, corr=None):
        if corr is None:
            corr = gaussian
        if not callable(corr):
            raise ValueError('corr must be callable')
        self.corr = corr

        if basis == 0:
            raise ValueError('basis must be non-zero scalar or callable')
        elif basis is None:
            basis = 1
        self.basis = self._domain_function(basis, cols=1)
        # if not callable(basis):
        #     raise ValueError('basis must be callable')
        # self.basis = basis

        if dim is None:
            dim = self.basis(np.zeros((1, 1))).shape[1]

        if dim < 1:
            raise ValueError('dim must be greater than or equal to one')

        if means is None:
            means = np.zeros(dim)
        else:
            means = np.atleast_1d(means)

        if means.shape[0] != dim:
            raise ValueError('means length does not match dim')

        if means.ndim > 1:
            raise ValueError('means must be 1d array or scalar')

        self.means_0 = means
        # self.mean_dim = len(self.means_0)

        if cov is None:
            self.inv_cov_0 = np.zeros((dim, dim))
        else:
            self.inv_cov_0 = np.linalg.inv(cov)

        if self.inv_cov_0.shape != (dim, dim):
            raise ValueError('Shape of cov must be (dim, dim)')

        if shape is None:
            self.shape_0 = 0
        else:
            self.shape_0 = shape

        if scale is None:
            self.scale_0 = 0
        else:
            self.scale_0 = scale

        self._shape = self.shape_0
        self._scale = self.scale_0
        self._means = self.means_0
        self._cov = cov
        self._inv_cov = self.inv_cov_0

    def _domain_function(self, obj, cols=None):
        try:
            isNumber = 0 == 0*obj
        except:
            isNumber = False

        if callable(obj):
            return obj
        elif isNumber:
            def dom_func(X, **kwargs):
                if cols is None:
                    vec = np.ones(len(X))
                else:
                    vec = np.ones((len(X), cols))
                return obj * vec
            return dom_func
        else:
            raise ValueError('{} must be a number or function'.format(obj))

    def observe(self, X, y, **corr_kwargs):
        R"""Observe GP outputs and update parameters.

        Conditions on :math:`n` iid processes at :math:`N` locations
        :math:`X`.

        Parameters
        ----------
        X : (N,d) array
            The input locations where the GPs are observed. :math:`N` is the
            number of points observed along each process, and :math:`d` is the
            dimensionality of each input point. If the process is a 1d curve,
            then ``X`` must be an ``(N, 1)`` shaped vector.
        y : (n, N) array
            The :math:`N` observed values of each of the :math:`n` iid
            processes. If only one process has been observed, this must have
            shape ``(1, N)``.
        **corr_kwargs : optional
            The keyword arguments passed to the correlation function. These
            values will be saved and used as defaults for all other methods.
        """

        if X.ndim != 2 and y != 2:
            raise ValueError('X and y must be 2d arrays')

        if y.shape[-1] != X.shape[0]:
            raise ValueError('X row length must match y column length')

        self._X = X
        self._y = y

        self._corr_kwargs = corr_kwargs

        self._chol = self.setup_chol(**corr_kwargs)
        self._shape = self.shape(y=y)
        self._scale = self.scale(y=y, **corr_kwargs)
        self._means = self.means(y=y, **corr_kwargs)
        self._inv_cov = self.inv_cov(y=y, **corr_kwargs)
        self._cov = self.cov(y=y, **corr_kwargs)

    def _recompute_corr(self, **corr_kwargs):
        # Must be non-empty and not equal to the defaults
        return corr_kwargs and corr_kwargs != self._corr_kwargs

    def shape(self, y=None):
        R"""The shape parameter :math:`a` of the inverse gamma distribution

        If ``y`` is given or ``observe`` has been used, then the posterior
        value is returned, else, the prior value is returned. The prior
        value can also be accessed with the ``shape_0`` attribute.

        Parameters
        ----------
        y : (n, N) array, optional
            The data on which to condition. Defaults to ``None``, which uses
            the data supplied by ``observe`` and returns a cached value.
            If observe has not been called, this returns the prior.

        Returns
        -------
        scalar
            The shape parameter
        """
        if y is None:
            return self._shape
        num_y, N = y.shape
        return self.shape_0 + N * num_y / 2.0

    def scale(self, y=None, **corr_kwargs):
        R"""The scale parameter :math:`b` of the inverse gamma distribution

        If ``y`` is given or ``observe`` has been used,
        then the posterior value is returned, else, the prior value is
        returned. The prior value can also be accessed with the ``scale_0``
        attribute.

        Parameters
        ----------
        y : (n, N) array, optional
            The data on which to condition. Defaults to ``None``, which uses
            the data supplied by ``observe``.
            If observe has not been called, this returns the prior.
        **corr_kwargs : optional
            The keyword arguments passed to the correlation function. Defaults
            to the valued supplied to ``observe``.

        Returns
        -------
        scalar
            The scale parameter. If ``y is None`` and ``corr_kwargs`` are
            omitted or the same as those passed to ``observe``, a cached
            value is returned, else it is recomputed.
        """
        # print(y, corr_kwargs)
        if y is None and not self._recompute_corr(**corr_kwargs):
            return self._scale

        # Set up variables
        means_0, inv_cov_0 = self.means_0, self.inv_cov_0
        means = self.means(y=y, **corr_kwargs)
        inv_cov = self.inv_cov(y=y, **corr_kwargs)
        y = y if y is not None else self.y
        R_chol = self.chol(**corr_kwargs)

        # Compute quadratics
        # val = np.dot(means_0.T, np.dot(inv_cov_0, means_0)) + \
        #     np.trace(np.dot(y, sp.linalg.cho_solve((R_chol, True), y.T))) - \
        #     np.dot(means.T, np.dot(inv_cov, means))

        right_quad = sp.linalg.solve_triangular(R_chol, y.T, lower=True)
        quad = np.trace(np.dot(right_quad.T, right_quad))
        # print('scalequad', quad)
        val = np.dot(means_0.T, np.dot(inv_cov_0, means_0)) + quad - \
            np.dot(means.T, np.dot(inv_cov, means))
        # print('scaleval', self._corr_kwargs, corr_kwargs, y, val)
        return self.scale_0 + val / 2.0

    def means(self, y=None, **corr_kwargs):
        R"""The mean parameters :math:`\mu` of the normal distribution on :math:`\beta`

        If ``y`` is given or ``observe`` has been used,
        then the posterior value is returned, else, the prior value is
        returned. The prior value can also be accessed with the ``means_0``
        attribute.

        Parameters
        ----------
        y : (n, N) array, optional
            The data on which to condition. Defaults to ``None``, which uses
            the data supplied by ``observe``.
            If observe has not been called, this returns the prior.
        **corr_kwargs : optional
            The keyword arguments passed to the correlation function. Defaults
            to the valued supplied to ``observe``.

        Returns
        -------
        (dim,) array
            The means of :math:`\beta`. If ``y is None`` and ``corr_kwargs``
            are omitted or the same as those passed to ``observe``, a cached
            value is returned, else it is recomputed.
        """
        if y is None and not self._recompute_corr(**corr_kwargs):
            return self._means
        y = y if y is not None else self.y
        num_y = y.shape[0]
        avg_y = np.average(y, axis=0)
        R_chol = self.chol(**corr_kwargs)
        H = self.basis(self.X)
        cov = self.cov(y=y, **corr_kwargs)

        Rinv_y = sp.linalg.cho_solve((R_chol, True), avg_y)
        val = np.dot(self.inv_cov_0, self.means_0) + \
            num_y * np.dot(H.T, Rinv_y)
        return np.dot(cov, val)

    def inv_cov(self, y=None, **corr_kwargs):
        R"""The inverse covariance :math:`V^{-1}` of the normal distribution on
        :math:`\beta`

        If ``y`` is given or ``observe`` has been used,
        then the posterior value is returned, else, the prior value is
        returned. The prior value can also be accessed with the ``inv_cov_0``
        attribute.

        Parameters
        ----------
        y : (n, N) array, optional
            The data on which to condition. Defaults to ``None``, which uses
            the data supplied by ``observe``.
            If observe has not been called, this returns the prior.
        **corr_kwargs : optional
            The keyword arguments passed to the correlation function. Defaults
            to the valued supplied to ``observe``.

        Returns
        -------
        (dim,) array
            The inverse covariance of :math:`\beta`. If ``y is None`` and
            ``corr_kwargs`` are omitted or the same as those passed to
            ``observe``, a cached value is returned, else it is recomputed.
        """
        if y is None and not self._recompute_corr(**corr_kwargs):
            return self._inv_cov
        y = y if y is not None else self.y
        num_y = y.shape[0]
        # num_y = num_y if num_y is not None else self.num_y
        R_chol = self.chol(**corr_kwargs)
        H = self.basis(self.X)

        right = sp.linalg.solve_triangular(R_chol, H, lower=True)
        quad = np.dot(right.T, right)
        # print('inv_cov_quad', quad, H)
        return self.inv_cov_0 + num_y * quad

    def cov(self, y=None, **corr_kwargs):
        R"""The covariance :math:`V` of the normal distribution on :math:`\beta`

        If ``y`` is given or ``observe`` has been used,
        then the posterior value is returned, else, the prior value is
        returned. The prior value can also be accessed with the ``cov_0``
        attribute.

        Parameters
        ----------
        y : (n, N) array, optional
            The data on which to condition. Defaults to ``None``, which uses
            the data supplied by ``observe``.
            If observe has not been called, this returns the prior.
        **corr_kwargs : optional
            The keyword arguments passed to the correlation function. Defaults
            to the valued supplied to ``observe``.

        Returns
        -------
        (dim,) array
            The covariance of :math:`\beta`. If ``y is None`` and
            ``corr_kwargs`` are omitted or the same as those passed to
            ``observe``, a cached value is returned, else it is recomputed.
        """
        if y is None and not self._recompute_corr(**corr_kwargs):
            return self._cov
        return np.linalg.inv(self.inv_cov(y=y, **corr_kwargs))

    def setup_chol(self, **corr_kwargs):
        corr = stabilize(self.corr(self.X, **corr_kwargs))
        cond = np.linalg.cond(corr)
        print('Warning: stabilized corr has condition number of {:.2e}'.format(cond))
        return np.linalg.cholesky(corr)

    def chol(self, **corr_kwargs):
        if self._recompute_corr(**corr_kwargs):
            return self.setup_chol(**corr_kwargs)
        else:
            return self._chol

    @property
    def y(self):
        return self._y

    @property
    def X(self):
        return self._X

    def ESS(self, **corr_kwargs):
        R_chol = self.chol(**corr_kwargs)
        N = len(self.X)
        H = self.basis(self.X)
        right = sp.linalg.solve_triangular(R_chol, H, lower=True)
        return N * np.trace(np.dot(right.T, right)) / np.trace(np.dot(H.T, H))

    def MAP_params(self, y=None, **corr_kwargs):
        """The MAP parameters mu, cov for the GP based on data
        
        [description]
        
        Parameters
        ----------
        **corr_kwargs : {[type]}
            [description]
        y : {[type]}, optional
            [description] (the default is None, which [default_description])
        """
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs
        R = self.corr(self.X, **corr_kwargs)
        corr_chol = self.chol(**corr_kwargs)

        H = self.basis(self.X)
        shape = self.shape(y=y)
        scale = self.scale(y=y, **corr_kwargs)
        means = self.means(y=y, **corr_kwargs)
        MAP_mean = np.dot(H, means)
        MAP_var = scale / (1.0 + shape)

        MAP_cov = MAP_var * R
        MAP_chol = np.sqrt(MAP_var) * corr_chol
        return MAP_mean, MAP_cov, MAP_chol

    def student_params(self, X=None, H=None, R=None, y=None, **corr_kwargs):
        R"""Returns the parameters of the student :math:`t` distribution.

        Given a function

        .. math::

            y | \beta, \sigma^2, \psi \sim N[H\beta, \sigma^2 R]

        with a normal inverse gamma prior on :math:`\beta, \sigma^2`,

        .. math::

            \beta, \sigma^2 \sim NIG(\mu, V, a, b)

        the integrated process is given by

        .. math::

            y | \psi \sim MVT\left[2a, H\mu, \frac{b}{a} (R + HVH^T)\right]

        If data has been observed, then posterior values for :math:`\mu,V,a,b`
        are used.

        Parameters
        ----------
        X : (N, d) array, optional
            The input points at which to compute the basis ``H`` and ``R``.
            If ``None``, then the defaults from the ``observe`` method are
            used.
        H : (N, q) array, optional
            The basis function evaluated at :math:`N` points. If None, then
            the basis function is computed at ``X``.
        R : (N, N) array, optional
            The correlation matrix. If None, then the correlation function is
            computed using ``X`` and **corr_kwargs.
        y : (n, N) array, optional
            Observed GP values used to compute the updated normal-inverse-gamma
            hyperparameters. If ``None``, then the data passed to the
            ``observe`` method are used. If ``observe`` has not been called,
            prior values are used.
        **corr_kwargs : optional
            Optional keyword arguments for the correlation function. If none
            are provided, defaults from the ``observe`` method are used
            instead.

        Returns
        -------
        tuple
            The degrees of freedom, mean, and sigma matrix of a multivariate
            :math:`t` distribution.
        """
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs
        if X is None:
            X = self.X
        if H is None:
            H = self.basis(X)
        if R is None:
            R = self.corr(X, **corr_kwargs)
        shape = self.shape(y=y)
        scale = self.scale(y=y, **corr_kwargs)
        means = self.means(y=y, **corr_kwargs)
        cov = self.cov(y=y, **corr_kwargs)

        mean = np.dot(H, means)
        sigma = scale / shape * (R + np.dot(H, np.dot(cov, H.T)))
        df = 2 * shape
        return df, mean, sigma

    def _build_conditional(self, Xnew, index=None, y=None, corr=None,
                           basis=None, **corr_kwargs):
        # Set up variables
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs

        if corr is None:
            corr = self.corr
            R_chol = self.chol(**corr_kwargs)
        else:
            R = corr(X=self.X, Xp=self.X, **corr_kwargs)
            R_chol = np.linalg.cholesky(stabilize(R))
        R_12 = corr(X=self.X, Xp=Xnew, **corr_kwargs)
        R_21 = R_12.T
        R_22 = corr(X=Xnew, Xp=None, **corr_kwargs)

        if basis is None:
            basis = self.basis
        H = basis(self.X)
        H_2 = basis(Xnew)

        # Compute conditional covariance
        Rinv_R12 = sp.linalg.cho_solve((R_chol, True), R_12)
        quad = np.dot(R_21, Rinv_R12)
        R_new = R_22 - quad

        # Conditional basis
        Rinv_H = sp.linalg.cho_solve((R_chol, True), H)
        H_new = H_2 - np.dot(R_21, Rinv_H)

        if y is None:
            y = self.y
        if index is None:
            Rinv_yn = sp.linalg.cho_solve((R_chol, True), y.T)
        else:
            Rinv_yn = sp.linalg.cho_solve((R_chol, True), y.T[:, index])
        shift = np.dot(R_21, Rinv_yn)

        return H_new, shift, R_new

    def conditional(self, index, Xnew, corr=False, **corr_kwargs):
        R"""Returns a conditional distribution object anchored to observed points.

        The conditional Gaussian process given the observed ``y[index]`` is
        marginalized over :math:`\beta` and :math:`\sigma^2`. The resulting
        distribution is a multivariate student :math:`t` distribution.

        Parameters
        ----------
        index : int
            The index of the observed ``y`` to condition upon. Despite only
            one process being interpolated at a time, the hyperparameters
            are still updated by all curves at once.
        Xnew : (M, d) array
            The :math:`M` new input points at which to predict the value of the
            process.
        corr : bool, optional
            Whether or not the distribution object is correlated. For
            visualizing the mean and marginal variance, an uncorrelated
            conditional often suffices. Defaults to ``False``.
        **corr_kwargs : optional
            Optional keyword arguments for the correlation function. If none
            are provided, defaults from the ``observe`` method are used
            instead.

        Returns
        -------
        distribution object
            If ``corr is False`` then a ``scipy.stats.t`` distribution is
            returned, else a
            ``statsmodels.sandbox.distributions.mv_normal.MVT`` is returned
        """
        H_new, shift, R_new = self._build_conditional(
            Xnew=Xnew, index=index, y=None, **corr_kwargs)
        df, mean, sigma = self.student_params(
            H=H_new, R=R_new, y=None, **corr_kwargs)
        mean += shift
        if corr:
            return MVT(mean=mean, sigma=stabilize(sigma), df=df)
        else:
            scale = np.sqrt(np.diag(sigma))
            return st.t(df=df, loc=mean, scale=scale)

    def condition(self, index, Xnew, dob=None, **corr_kwargs):
        R"""Conditions on observed data and returns interpolant and error bands.

        Extracts the mean and degree of belief intervals from the corresponding
        conditional object.

        Parameters
        ----------
        index : int
            The index of the observed ``y`` to condition upon. Despite only
            one process being interpolated at a time, the hyperparameters
            are still updated by all curves at once.
        Xnew : (M, d) array
            The :math:`M` new input points at which to predict the value of the
            process.
        dob : scalar or 1d array, optional
            The degree of belief intervals to compute, between 0 and 1.
        **corr_kwargs : optional
            Optional keyword arguments for the correlation function. If none
            are provided, defaults from the ``observe`` method are used
            instead.

        Returns
        -------
        array or tuple
            If ``dob is None``, then only the 1D array of predictions is
            returned. Otherwise, the predictions along with a
            :math:`2 \times N` (or :math:`len(dob) \times 2 \times N`) of
            degree of belief intervals is returned.
        """
        dist = self.conditional(index, Xnew, corr=False, **corr_kwargs)
        return predictions(dist, dob=dob)

    def predictive(self, Xnew, corr=False, **corr_kwargs):
        """Returns a posterior predictive distribution object.

        Predicts new curves given the observed curves ``y``

        Parameters
        ----------
        Xnew : (M, d) array
            The :math:`M` new input points at which to predict the value of the
            process.
        corr : bool, optional
            Whether or not the distribution object is correlated. For
            visualizing the mean and marginal variance, an uncorrelated
            conditional often suffices. Defaults to ``False``.
        **corr_kwargs : optional
            Optional keyword arguments for the correlation function. If none
            are provided, defaults from the ``observe`` method are used
            instead.

        Returns
        -------
        distribution object
            If ``corr is False`` then a ``scipy.stats.t`` distribution is
            returned, else a
            ``statsmodels.sandbox.distributions.mv_normal.MVT`` is returned
        """
        df, mean, sigma = self.student_params(X=Xnew, y=None, **corr_kwargs)
        if corr:
            return MVT(mean=mean, sigma=stabilize(sigma), df=df)
        else:
            scale = np.sqrt(np.diag(sigma))
            return st.t(df=df, loc=mean, scale=scale)

    def predict(self, Xnew, dob=None, **corr_kwargs):
        dist = self.predictive(Xnew=Xnew, corr=False, **corr_kwargs)
        return predictions(dist, dob=dob)

    def evidence(self, log=True, y=None, **corr_kwargs):
        R"""Computes the evidence, or marginal likelihood, of the observed data

        Specifically, the evidence integrates out :math:`\beta` and
        :math:`\sigma^2` such that

        .. math::

            pr(y | \psi) = \frac{\Gamma(a)}{\Gamma(a_0)} \frac{b_0^{a_0}}{b^a}
                \sqrt{\frac{|V|}{|V_0|}} [(2\pi)^N |R|]^{-n/2}

        where subscript 0's denote prior values. If the priors on :math:`a_0`
        or :math:`V_0` are uninformative, then the evidence is undefined, but
        in this case the evidence is approximated by

        .. math::

            pr(y | \psi) = \frac{\Gamma(a) \sqrt{|V|}}{b^a}
                [(2\pi)^N |R|]^{-n/2}

        This is appropriate for model comparison since the factor due to priors
        is only a constant and hence cancels.

        Parameters
        ----------
        log : bool, optional
            Whether to return the log of the evidence, which can be useful
            for numerical reasons.
        y : (n, N) array, optional
            Data for which to compute the evidence. Defaults to the data
            passed to the ``observe`` method.
        **corr_kwargs : optional
            Keyword arguments passed to the correlation function. Defaults
            to those passed to the ``observe`` method.

        Returns
        -------
        scalar
            The (log) evidence
        """
        shape = self.shape(y=y)
        scale = self.scale(y=y, **corr_kwargs)
        means = self.means(y=y, **corr_kwargs)
        inv_cov = self.inv_cov(y=y, **corr_kwargs)
        cov = self.cov(y=y, **corr_kwargs)
        R_chol = self.chol(**corr_kwargs)
        y = y if y is not None else self.y
        num_y, N = y.shape

        tr_log_R = 2 * np.sum(np.log(np.diag(R_chol)))
        _, logdet_cov = np.linalg.slogdet(cov)

        # print(num_y, N, shape, scale, means, inv_cov, tr_log_R, logdet_cov)

        ev = - 0.5 * num_y * (N * np.log(2*np.pi) + tr_log_R)
        ev += sp.special.gammaln(shape) + 0.5 * logdet_cov - \
            shape * np.log(scale)
        # print('parent1ev', ev)
        if self.inv_cov_0.any() and self.scale_0 != 0:  # If non-zero
            _, logdet_inv_cov_0 = np.linalg.slogdet(self.inv_cov_0)
            ev += - sp.special.gammaln(self.shape_0) + \
                0.5 * logdet_inv_cov_0 + self.shape_0 * np.log(self.scale_0)
        # print('parent2ev', ev)
        if not log:
            ev = np.exp(ev)
        return ev

    def posterior(self, name, logprior=None, log=False, y=None, **corr_kwargs):
        R"""Returns the posterior pdf for arbitrary correlation variables

        Uses Bayes' Theorem to compute

        .. math::

            pr(\ell | y, ...) \propto pr(y | \ell, ...) pr(\ell)

        for any correlation parameter :math:`\ell`. The evidence given
        :math:`\ell` and the other correlation parameters (...) is then
        used to compute the posterior.

        Parameters
        ----------
        name : str
            The name of the variable passed to the correlation function
            for which to calculate the posterior
        logprior : callable, optional
            The log prior to place on ``name``. Must accept ``**corr_kwargs``
            as arguments. Defaults to ``None``, which sets ``logprior`` to zero
        log : bool, optional
            Whether to return the log posterior. If ``False``, then the pdf
            will be approximately normalized using the trapezoid rule. Defaults
            to ``False``
        **corr_kwargs :
            Keyword arguments passed to the correlation function. One of the
            arguments must match ``name`` and must be an array. Nothing will
            be inferred from the ``observe`` call here.

        Returns
        -------
        array
            The (log) posterior pdf for the ``name`` variable.
        """
        def ev(val):
            kw = {name: np.squeeze(val)}
            # print(kw, corr_kwargs)
            return self.evidence(log=True, y=y, **kw, **corr_kwargs)

        log_pdf = 0
        if logprior is not None:
            log_pdf += logprior(**corr_kwargs)

        vals = corr_kwargs.pop(name)
        log_pdf += np.apply_along_axis(ev, 1, np.atleast_2d(vals).T)

        if not log:
            log_pdf -= np.max(log_pdf)
            pdf = np.exp(log_pdf)
            # Integrate using trapezoid rule
            norm = np.trapz(pdf, vals)
            return pdf/norm
        return log_pdf

    def credible_diagnostic(self, data, dobs, band_intervals=None,
                            band_dobs=None, samples=1e4, **kwargs):
        dist = self.predictive(corr=False, **kwargs)
        lower, upper = dist.interval(np.atleast_2d(dobs).T)

        def diagnostic(data, lower, upper):
            indicator = (lower < data) & (data < upper)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)   # The diagnostic

        D_CI = np.apply_along_axis(
                diagnostic, axis=1, arr=np.atleast_2d(data), lower=lower,
                upper=upper)
        D_CI = np.squeeze(D_CI)

        # Calculate uncertainty in result using a reference distribution
        if band_intervals is not None:
            band_intervals = np.atleast_1d(band_intervals)
            if band_dobs is None:
                band_dobs = dobs
            band_dobs = np.atleast_2d(band_dobs)

            corr_dist = self.predictive(corr=True, **kwargs)
            random_data = corr_dist.rvs(size=int(samples))
            band_lower, band_upper = dist.interval(band_dobs.T)
            band_D_CI = np.apply_along_axis(
                diagnostic, axis=1, arr=random_data, lower=band_lower,
                upper=band_upper)
            # bands = np.array(
            #     [pm.hpd(band_D_CI, 1-bi) for bi in band_intervals])
            # Band shape: (len(dobs), 2, len(X))
            bands = np.array(
                [np.percentile(band_D_CI, [100*(1-bi)/2, 100*(1+bi)/2], axis=0)
                 for bi in band_intervals])
            # bands = np.transpose(bands, [0, 2, 1])
            return D_CI, bands
        return D_CI

    # def corr_post(self, logprior=None, **corr_kwargs):
    #     """Evaluates the posterior for the correlation parameters in corr_kwargs

    #     Parameters
    #     ----------
    #     logprior : callable
    #     corr_kwargs : dict
    #         The values of the correlation parameters at which to evaluate the
    #         posterior. Because the evidence is vectorized, standard
    #         array broadcasting rules apply
    #     """
    #     # if not callable(logprior):
    #     #     raise ValueError('logprior must be callable')

    #     vec_evidence = np.vectorize(self.evidence)
    #     log_post = vec_evidence(log=True, **corr_kwargs)
    #     if logprior is not None:
    #         log_post = log_post + logprior(**corr_kwargs)
    #     log_post -= np.max(log_post)
    #     return np.exp(log_post)


class PowerProcess(SGP):
    R"""A power series with iid random processes as coefficients.

    Implements the following model

    .. math::

        S_k(x) = S_{\mathrm{ref}}(x) \sum_{n=0}^k c_n(x) r(x)^n

    where the :math:`c_n` are Gaussian processes with parameters using
    conjugate priors

    .. math::

        c_n(x) | \beta, \sigma^2, \psi & \sim N(m(x), \sigma^2 R(x,x;\psi)) \\
        \beta, \sigma^2 & \sim NIG(\mu, V, a, b)

    Conditioning on partial sums :math:`S_{0}`, :math:`\dots,` :math:`S_k`,
    allow one to estimate the full summation and obtain posteriors for the
    parameters.
    """

    def _recompute_coeffs(self, **ratio_kwargs):
        if ratio_kwargs and ratio_kwargs != self._ratio_kwargs:
            # print('recomputing...')
            coeffs = coefficients(
                partials=self.partials, ratio=self.ratio, X=self.X,
                ref=self.ref(self.X), orders=self._full_orders,
                # rm_orders=self.rm_orders,
                **ratio_kwargs)
            return coeffs[self._mask]
        return None

    def observe(self, X, partials, ratio, ref=1, orders=None,
                leading_kwargs=None, ratio_kwargs=None, **corr_kwargs):
        R"""Observe the partial sums of the series and update parameters.

        The partial sums are observed at the input locations ``X``.
        Using the given ``ratio`` and ``ref``, the coefficients of the series
        are extracted and are used to create posterior distributions for the
        mean and variance parameters. One can then interpolate partial sums or
        predict the value of the full summation at new points. Additional
        functionality, including evidence calculations and posteriors for the
        ratio and correlation functions are possible.

        Parameters
        ----------
        X : (N,d) array
            The :math`N` input locations where the partial sums are observed.
            Columns correspond to the dimensionality of the input space. If 1D,
            ``X`` must be an :math:`N \times 1` column vector.
        partials : (n,N) array
            The :math:`n` lowest known partial sums, each with :math:`N` points
            observed along each curve.
        ratio : callable, scalar, or (N,) array
            The value of the ratio that scales each order in the power
            series with increasing powers. If callable, it must accept ``X``
            as its first argument and can optionally accept **ratio_kwargs.
        ref : callable, scalar, or length (N,) array
            The overall scale of the power series. If callable, it must
            accept ``X`` as its first argument. The default value is 1.
        orders : (n,) array
            The orders of the given partial sums. If ``None``, it is assumed
            that all orders from 0 to math:`n` are given: ``[0, 1, ..., n]``.
        leading_kwargs : dict
            Keyword arguments passed to an ``SGP`` initializer for the leading
            order coefficient. This allows the leading order to be treated
            differently than the other orders of the series. ``corr_kwargs``
            can also be in this dict and will be passed to the ``observe``
            method of the leading order ``SGP``. Defaults to ``None``, in which
            case the leading order is an iid coefficient like the others.
        ratio_kwargs : dict
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``.
        **corr_kwargs : optional
            Additional keyword arguments passed to the correlation function.
            These values will be saved and used as defaults for all other
            methods.
        """
        # if corr_kwargs is None:
        #     corr_kwargs = {}
        if ratio_kwargs is None:
            ratio_kwargs = {}

        self._ratio_kwargs = ratio_kwargs
        self._full_orders = np.asarray(orders)

        self.partials = partials
        self.ratio = self._domain_function(ratio)
        self.ref = self._domain_function(ref)

        coeffs = coefficients(
            partials=partials, ratio=ratio, X=X, ref=ref(X), orders=orders,
            **ratio_kwargs)

        self._mask = np.ones(len(orders), dtype=bool)
        if leading_kwargs is not None:
            self._mask[0] = False
            # Separate pieces without editing leading_kwargs in place
            leading_corr_kwargs = leading_kwargs.get('corr_kwargs', {})
            self._leading_corr_kwargs = leading_corr_kwargs
            leading_kwargs = {k: v for k, v in leading_kwargs.items()
                              if k != 'corr_kwargs'}
            self.leading_process = SGP(**leading_kwargs)
            self.leading_process.observe(X=X, y=np.atleast_2d(coeffs[0]),
                                         **leading_corr_kwargs)

        self.orders = np.asarray(orders)[self._mask]
        self.ordersvec = np.atleast_2d(self.orders).T

        # Get max order
        max_order_arg = np.argmax(self.orders)
        self.max_order = orders[max_order_arg]
        self.max_partial = partials[max_order_arg]
        super(PowerProcess, self).observe(
            X=X, y=coeffs[self._mask], **corr_kwargs)

    def _build_conditional(self, Xnew, index=None, rescale=False,
                           ratio_kwargs=None, **corr_kwargs):
        # Set up variables
        if ratio_kwargs is None:
            ratio_kwargs = self._ratio_kwargs
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs
        all_ords = self._full_orders

        if index == 0 and hasattr(self, 'leading_process'):
            H_new, shift, R_new = self.leading_process._build_conditional(
                Xnew=Xnew, index=0)
            if rescale:
                # print(self.ref(Xnew).shape)
                H_new *= self.ref(Xnew)[:, None]
                shift *= self.ref(Xnew)
                R_new *= self.ref(Xnew)[:, None] * self.ref(Xnew)
            return H_new, shift, R_new
        elif rescale:
            low_ords = [i for i in self.orders if i <= all_ords[index]]

            def basis(X):
                ratio = self.ratio(X, **ratio_kwargs)[:, None]
                ratio_sum = 1
                if low_ords:
                    ratio_sum = np.sum([ratio**n for n in low_ords], axis=0)
                ref = self.ref(X)[:, None]
                return ref * ratio_sum * self.basis(X)

            def corr(X, Xp=None, **kw):
                if Xp is None:
                    Xp = X
                ratioX = self.ratio(X, **ratio_kwargs)
                ratioXp = self.ratio(Xp, **ratio_kwargs)
                # ratio_mat = 1
                if low_ords:
                    ratio_mat = ratioX[:, None] * ratioXp
                    ratio_mat = np.sum([ratio_mat**n for n in low_ords], axis=0)

                ref_mat = self.ref(X)[:, None] * self.ref(Xp)
                return ref_mat * ratio_mat * self.corr(X, Xp, **kw)

            y = self.partials
            if hasattr(self, 'leading_process'):
                y = y - self.partials[0]
            H_new, shift, R_new = super(PowerProcess, self)._build_conditional(
                Xnew=Xnew, index=index, y=y, basis=basis,
                corr=corr, **corr_kwargs)
            # if hasattr(self, 'leading_process'):
            #     shift += self.ref(Xnew) * shift_0
            # H_new, shift, R_new = super(PowerProcess, self)._build_conditional(
            #     Xnew=Xnew, index=index, y=self.partials, basis=basis,
            #     corr=corr, **corr_kwargs)
            # shift += self.ref(Xnew)
        else:
            coeffs = coefficients(
                partials=self.partials, ratio=self.ratio, X=self.X,
                ref=self.ref(self.X), orders=all_ords,
                **ratio_kwargs)
            H_new, shift, R_new = super(PowerProcess, self)._build_conditional(
                Xnew=Xnew, index=index, y=coeffs, **corr_kwargs)

        return H_new, shift, R_new

    def _integrated_conditional(self, Xnew, index, rescale=True,
                                max_order=None, ratio_kwargs=None,
                                **corr_kwargs):
        if ratio_kwargs is None:
            ratio_kwargs = self._ratio_kwargs
        coeffs = self._recompute_coeffs(**ratio_kwargs)
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs

        H_new, shift, R_new = self._build_conditional(
            Xnew=Xnew, index=index, rescale=rescale,
            ratio_kwargs=ratio_kwargs, **corr_kwargs)

        if max_order is not None:  # For predictive
            H_pred, R_pred = self._build_predictive(
                Xnew=Xnew, max_order=max_order, rescale=rescale,
                ratio_kwargs=ratio_kwargs, **corr_kwargs)
            H_new = H_new + H_pred
            R_new = R_new + R_pred

        df, mean, sigma = self.student_params(
            H=H_new, R=R_new, y=coeffs, **corr_kwargs)
        mean = mean + shift

        if rescale and index > 0 and hasattr(self, 'leading_process'):
            # Must include leading term here
            H_0, shift_0, R_0 = self._build_conditional(
                Xnew=Xnew, index=0, rescale=rescale,
                ratio_kwargs=ratio_kwargs, **corr_kwargs)
            # H_0_scaled = self.ref(Xnew) * H_0
            df_0, mean_0, sigma_0 = self.student_params(
                X=Xnew, H=H_0, R=R_0)
            mean = mean + mean_0 + shift_0
            sigma = sigma + sigma_0
            df = df + df_0

        return df, mean, sigma

    def conditional(self, Xnew, index, corr=False, rescale=True,
                    ratio_kwargs=None, **corr_kwargs):
        R"""Returns a conditional distribution object anchored to observed points.

        The conditional distribution of the coefficients, or partial sums, is
        marginalized over the means and variance parameter.

        Parameters
        ----------
        Xnew : (M,d) array
            The new input points at which to predict the values of the
            coefficients or partial sums.
        index : int
            The index of the partial sum to interpolate.
        corr : bool, optional
            Whether or not the conditional distribution is correlated in the
            input space. For visualizing the mean and marginal variance, an
            uncorrelated conditional often suffices. Defaults to ``False``.
        rescale : bool, optional
            Whether or not to rescale the coefficient back to a partial sum.
            Defaults to ``True``.
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``, which uses the values passed to ``observe``.
        **corr_kwargs : optional
            Additional keyword arguments passed to the correlation function.
            Defaults to the keywords passed to ``observe``.

        Returns
        -------
        distribution object
            If ``corr is False`` then a ``scipy.stats.t`` distribution is
            returned, else a
            ``statsmodels.sandbox.distributions.mv_normal.MVT`` is returned
        """
        df, mean, sigma = self._integrated_conditional(
            Xnew=Xnew, index=index, rescale=rescale, max_order=None,
            ratio_kwargs=ratio_kwargs, **corr_kwargs)

        if corr:
            return MVT(mean=mean, sigma=sigma, df=df)
        else:
            scale = np.sqrt(np.diag(sigma))
            return st.t(df=df, loc=mean, scale=scale)

    def condition(self, Xnew, index, dob=None, rescale=True,
                  ratio_kwargs=None, **corr_kwargs):
        R"""Conditions on observed data and returns interpolant and error bands.

        Extracts the mean and degree of belief intervals from the corresponding
        conditional object.

        Parameters
        ----------
        Xnew : 2D array
            The new input points at which to predict the values of the
            coefficients or partial sums.
        index : int
            The index of the partial sum to interpolate.
        dob : scalar or 1D array, optional
            The degree of belief intervals to compute, between 0 and 1.
        rescale : bool, optional
            Whether or not to rescale the coefficient back to a partial sum.
            Defaults to ``True``
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``, which uses the values passed to ``observe``.
        **corr_kwargs : optional
            Additional keyword arguments passed to the correlation function.
            Defaults to the keywords passed to ``observe``.

        Returns
        -------
        1D array or tuple
            If ``dob is None``, then only the 1D array of predictions is
            returned. Otherwise, the predictions along with a
            :math:`2 \times N` (or :math:`len(dob) \times 2 \times N`) of
            degree of belief intervals is returned.
        """
        dist = self.conditional(Xnew=Xnew, index=index, rescale=rescale,
                                ratio_kwargs=ratio_kwargs, **corr_kwargs)
        return predictions(dist, dob=dob)

    def _build_predictive(self, Xnew, max_order=None, rescale=True,
                          ratio_kwargs=None, **corr_kwargs):
        if ratio_kwargs is None:
            ratio_kwargs = self._ratio_kwargs
        if not self._recompute_corr(**corr_kwargs):
            corr_kwargs = self._corr_kwargs
        if max_order is None:
            max_order = np.inf
        # Largest observed order
        k = self._full_orders[-1]

        # Find geometric sum of ratio from k to max_order
        r = self.ratio(Xnew, **ratio_kwargs)[:, None]
        r_mat = r * r.ravel()
        mu_sum = r**(k+1) * (1 - r**(max_order - k)) / (1 - r)
        corr_sum = r_mat**(k+1) * (1 - r_mat**(max_order - k)) / (1 - r_mat)

        # Truncation uncertainty
        H = self.basis(Xnew)
        R = self.corr(Xnew, **corr_kwargs)
        H = H * mu_sum
        R = R * corr_sum
        if rescale:
            ref = self.ref(Xnew)[:, None]
            ref_mat = ref * ref.ravel()
            H = H * ref
            R = R * ref_mat
        return H, R

    def predictive(self, Xnew, corr=False, max_order=None, rescale=True,
                   ratio_kwargs=None, **corr_kwargs):
        R"""Returns a posterior predictive distribution object.

        Predicts the value of the power series up to ``max_order`` at
        the input locations ``X``.

        Parameters
        ----------
        X : 2D array
            The new input points at which to predict the values of the
            coefficients or partial sums.
        corr : bool, optional
            Whether or not the distribution is correlated in the
            input space. For visualizing the mean and marginal variance, an
            uncorrelated conditional often suffices. Defaults to ``False``.
        max_order : int, optional
            The order at which to truncate the power series.
            Defaults to ``None``, which corresponds to an infinite sum.
        rescale : bool, optional
            Whether or not to rescale the truncated orders back to error bands
            on a partial sum. Defaults to ``True``.
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``, which uses the values passed to ``observe``.
        **corr_kwargs : optional
            Additional keyword arguments passed to the correlation function.
            Defaults to the keywords passed to ``observe``.

        Returns
        -------
        distribution object
            If ``corr is False`` then a ``scipy.stats.t`` distribution is
            returned, else a
            ``statsmodels.sandbox.distributions.mv_normal.MVT`` is returned
        """
        if ratio_kwargs is None:
            ratio_kwargs = self._ratio_kwargs
        if max_order is None:
            max_order = np.inf
        # Always find error from best prediction
        index = len(self.partials) - 1

        if rescale:
            df, mean, sigma = self._integrated_conditional(
                Xnew=Xnew, index=index, rescale=rescale, max_order=max_order,
                ratio_kwargs=ratio_kwargs, **corr_kwargs)
        else:
            coeffs = self._recompute_coeffs(**ratio_kwargs)
            H, R = self._build_predictive(
                Xnew=Xnew, max_order=max_order, rescale=False,
                ratio_kwargs=ratio_kwargs, **corr_kwargs)
            df, mean, sigma = self.student_params(
                H=H, R=R, y=coeffs, **corr_kwargs)
        if corr:
            return MVT(mean=mean, sigma=stabilize(sigma), df=df)
        else:
            scale = np.sqrt(np.diag(sigma))
            return st.t(df=df, loc=mean, scale=scale)

    def predict(self, Xnew, dob=None, max_order=None, rescale=True,
                ratio_kwargs=None, **corr_kwargs):
        R"""Predicts the power series value and provides error bands.

        Gets the mean value of the predictive distribution and computes
        the degree of belief interval.

        Parameters
        ----------
        Xnew : 2D array
            The new input points at which to predict the values of the
            coefficients or partial sums.
        dob : scalar or 1D array, optional
            The degree of belief intervals to compute, between 0 and 1.
        max_order : int, optional
            The order at which to truncate the power series.
            Defaults to ``None``, which corresponds to an infinite sum.
        rescale : bool, optional
            Whether or not to rescale the truncated orders back to error bands
            on a partial sum. Defaults to ``True``.
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``, which uses the values passed to ``observe``.
        **corr_kwargs : optional
            Additional keyword arguments passed to the correlation function.

        Returns
        -------
        1D array or tuple
            If ``dob is None``, then only the 1D array of predictions is
            returned. Otherwise, the predictions along with a
            :math:`2 \times N` (or :math:`len(dob) \times 2 \times N`) of
            degree of belief intervals is returned.
        """
        dist = self.predictive(
            Xnew=Xnew, corr=False, max_order=max_order, rescale=rescale,
            ratio_kwargs=ratio_kwargs, **corr_kwargs)
        return predictions(dist, dob=dob)

    def evidence(self, log=True, ratio_kwargs=None, y=None, **corr_kwargs):
        R"""Computes the evidence, or marginal likelihood, of the partial sums

        Specifically, the evidence integrates out :math:`\beta` and
        :math:`\sigma^2` such that

        .. math::

            pr({S_i} | \psi, r, S_{\mathrm{ref}})
            & = \frac{pr(y | \psi)}{\prod_i |S_{\mathrm{ref}} r^i|} \\
            & = \frac{1}{\prod_n |S_{\mathrm{ref}} r^n|}
                \frac{\Gamma(a)}{\Gamma(a_0)} \frac{b_0^{a_0}}{b^a}
                \sqrt{\frac{|V|}{|V_0|}} [(2\pi)^N |R|]^{-n/2}

        where subscript 0's denote prior values. If the priors on :math:`a_0`
        or :math:`V_0` are uninformative, then the evidence is undefined, but
        in this case the evidence is approximated by

        .. math::

            pr({S_i} | \psi, r, S_{\mathrm{ref}})
            = \frac{1}{\prod_i |S_{\mathrm{ref}} r^i|}
              \frac{\Gamma(a) \sqrt{|V|}}{b^a} [(2\pi)^N |R|]^{-n/2}

        This is appropriate for model comparison since the factor due to priors
        is only a constant and hence cancels.

        Parameters
        ----------
        log : bool, optional
            Whether to return the log of the evidence, which can be useful
            for numerical reasons.
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``, which uses the values passed to ``observe``.
        **corr_kwargs : optional
            Keyword arguments passed to the correlation function. Defaults
            to those passed to the ``observe`` method.

        Returns
        -------
        scalar
            The (log) evidence
        """
        if ratio_kwargs is None:
            ratio_kwargs = self._ratio_kwargs

        # Compute evidence of coefficients
        if y is not None:
            coeffs = y
        else:
            coeffs = self._recompute_coeffs(**ratio_kwargs)
        # print('1257', corr_kwargs)
        ev = super(PowerProcess, self).evidence(
            log=True, y=coeffs, **corr_kwargs)
        # print('ev1', ev)

        if hasattr(self, 'leading_process'):
            ev += self.leading_process.evidence(log=True)
            # print('ev2', ev)

        # Consider ratio and ref too
        ev -= len(self.orders) * np.sum(np.log(self.ref(self.X)))
        ev -= np.sum(self.orders) * \
            np.sum(np.log(self.ratio(self.X, **ratio_kwargs)))

        # print('ev3', ev)
        if not log:
            ev = np.exp(ev)
        return ev

    def posterior(self, name, logprior=None, log=False, ratio_kwargs=None,
                  **corr_kwargs):
        R"""Returns the posterior pdf for arbitrary correlation or ratio variables

        Uses Bayes' Theorem to compute

        .. math::

            pr(x | y, ...) \propto pr(y | x, ...) pr(x)

        for any correlation or ratio parameter :math:`x`. The evidence given
        :math:`x` and the other correlation and ratio parameters (...) is then
        used to compute the posterior.

        Parameters
        ----------
        name : str
            The name of the variable passed to the correlation or ratio
            function for which to calculate the posterior. First checks
            ``corr_kwargs`` and if it is not found, then checks
            ``ratio_kwargs``.
        logprior : callable, optional
            The log prior to place on ``name``. Must accept ``**corr_kwargs``
            as arguments. Defaults to ``None``, which sets ``logprior`` to zero
        log : bool, optional
            Whether to return the log posterior. If ``False``, then the pdf
            will be approximately normalized using the trapezoid rule. Defaults
            to ``False``
        ratio_kwargs : dict, optional
            Additional keyword arguments passed to the ratio function.
        **corr_kwargs :
            Keyword arguments passed to the correlation function. One of the
            arguments must match ``name`` and must be an array. Nothing will
            be inferred from the ``observe`` call here.

        Returns
        -------
        array
            The (log) posterior pdf for the ``name`` variable.
        """
        if name in corr_kwargs:
            if ratio_kwargs is None:
                ratio_kwargs = self._ratio_kwargs
            y = self._recompute_coeffs(**ratio_kwargs)
            # print('1314', corr_kwargs)
            return super(PowerProcess, self).posterior(
                name=name, logprior=logprior, log=log, y=y, **corr_kwargs)

        log_pdf = 0
        if logprior is not None:
            log_pdf += logprior(**ratio_kwargs)

        # vals = ratio_kwargs.pop(name)
        vals = ratio_kwargs[name]

        def ev(val):
            # print(val)
            # ratio_kwargs[name] = np.squeeze(val)
            rkw = ratio_kwargs.copy()
            rkw[name] = np.squeeze(val)
            return self.evidence(log=True, ratio_kwargs=rkw,
                                 **corr_kwargs)

        log_pdf += np.apply_along_axis(ev, 1, np.atleast_2d(vals).T)
        # print('logpdf', log_pdf)

        if not log:
            log_pdf -= np.max(log_pdf)
            pdf = np.exp(log_pdf)
            # Integrate using trapezoid rule
            norm = np.trapz(pdf, vals)
            return pdf/norm
        return log_pdf

    # def ratio_post(self, ratio_kwargs, logprior=None, **corr_kwargs):
    #     if not callable(logprior):
    #         raise ValueError('logprior must be callable')

    #     def ratio_ev(**kwargs):
    #         coeffs = self._recompute_coeffs(**kwargs)
    #         return self.evidence(log=True, y=coeffs, **corr_kwargs)

    #     vec_evidence = np.vectorize(self.ratio_ev)
    #     log_post = vec_evidence(**ratio_kwargs)
    #     if logprior is not None:
    #         log_post = log_post + logprior(**ratio_kwargs)
    #     log_post -= np.max(log_post)
    #     return np.exp(log_post)


class PowerSeries(object):
    R"""A power series with iid random variables as coefficients.

    Implements the following model

    .. math::

        S_k = S_{\mathrm{ref}} \sum_{n=0}^k c_n r^n

    where the :math:`c_n` are iid Gaussians and :math:`\sigma^2` has a
    conjugate prior

    .. math::

        c_n | \sigma^2 & \sim N(0, \sigma^2) \\
        \sigma^2 & \sim IG(a, b)

    Conditioning on partial sums :math:`S_0`, :math:`\dots,` :math:`S_k`, allow
    one to estimate the full summation and obtain posteriors for the
    parameters.
    """

    def __init__(self, shape=None, scale=None):
        if shape is None:
            self.shape_0 = 0
        else:
            self.shape_0 = shape

        if scale is None:
            self.scale_0 = 0
        else:
            self.scale_0 = scale

        self._shape = self.shape_0
        self._scale = self.scale_0

    def observe(self, partials, ratio, ref=1, orders=None,
                rm_orders=None, X=None, **ratio_kwargs):
        R"""Observe the partial sums of the series and update parameters.

        The partial sums are observed and converted to coefficients
        using the given ``ratio`` and ``ref``. Posterior distributions for the
        mean and variance parameters can then be calculated.

        Parameters
        ----------
        partials : (n,N) array
            The :math:`n` lowest known partial sums, each with :math:`N` points
            observed along each partial sum.
        ratio : callable, scalar, or length (N,) 1D array
            The value of the ratio that scales each order in the power
            series with increasing powers. If callable, it must accept ``X``
            as its first argument and can optionally accept ``**ratio_kwargs``.
        ref : callable, scalar, or length (N,) 1D array
            The overall scale of the power series. If callable, it must
            accept ``X`` as its first argument. The default value is 1.
        orders : (n,) array
            The orders of the given partial sums. If ``None``, it is assumed
            that all orders from 0 to math:`n` are given: ``[0, 1, ..., n]``.
        rm_orders : 1D array
            The orders of partial sums, if any, to ignore during conditioning.
            This could be useful if it is known that some coefficients will
            not behave as iid Gaussian.
        X : (N,d) array, optional
            The :math`N` input locations where the partial sums are observed.
            Columns correspond to the dimensionality of the input space. If 1D,
            ``X`` must be an :math:`N \times 1` column vector.
        **ratio_kwargs : optional
            Additional keyword arguments passed to the ratio function. Defaults
            to ``None``.
        """

        self._ratio_kwargs = ratio_kwargs
        self._full_orders = orders

        self.partials = partials
        self.ratio = self._domain_function(ratio)
        # self.ref = self._domain_function(ref)
        # if callable(ratio):
        #     self.ratio = ratio(X, **ratio_kwargs)
        # else:
        #     self.ratio = ratio
        self.ratio = self._domain_function(ratio)

        if callable(ref):
            self.ref = ref(X)
        else:
            self.ref = ref

        if rm_orders is None:
            rm_orders = []

        if orders is None:
            orders = np.arange(0, len(partials), dtype=int)

        coeffs = coefficients(
            partials=partials, ratio=ratio, X=X, ref=ref, orders=orders,
            **ratio_kwargs)

        # Get max order
        max_order_arg = np.argmax(orders)
        self.max_order = orders[max_order_arg]
        self.max_partial = partials[max_order_arg]

        self.X = X
        self._mask = np.logical_not(np.isin(orders, rm_orders))
        self.orders = orders[self._mask]
        self.ordersvec = np.atleast_2d(self.orders).T
        self.rm_orders = rm_orders
        self.coeffs = coeffs[self._mask]
        self._shape = self.shape()

    def _domain_function(self, obj, cols=None):
        try:
            isNumber = 0 == 0*obj
        except:
            isNumber = False

        if callable(obj):
            return obj
        elif isNumber:
            def dom_func(X, **kwargs):
                if cols is None:
                    vec = np.ones(self.partials.shape[1])
                else:
                    vec = np.ones((self.partials.shape[1], cols))
                return obj * vec
            return dom_func
        else:
            raise ValueError('{} must be a number or function'.format(obj))

    def _recompute_coeffs(self, **ratio_kwargs):
        if ratio_kwargs and ratio_kwargs != self._ratio_kwargs:
            # print('recomputing...')
            coeffs = coefficients(
                partials=self.partials, ratio=self.ratio,
                ref=self.ref, orders=self._full_orders, X=self.X,
                **ratio_kwargs)
            return coeffs[self._mask]
        return self.coeffs

    def shape(self, **ratio_kwargs):
        R"""The shape parameter :math:`a` of the inverse gamma distribution.

        """
        coeffs = self._recompute_coeffs(**ratio_kwargs)
        num_c = coeffs.shape[0]
        return self.shape_0 + num_c / 2.0

    def scale(self, **ratio_kwargs):
        R"""The scale parameter :math:`b` of the inverse gamma distribution.

        [description]

        Parameters
        ----------
        **ratio_kwargs : {[type]}
            [description]
        combine : {bool}, optional
            [description] (the default is False, which [default_description])

        Returns
        -------
        [type]
            [description]
        """
        coeffs = self._recompute_coeffs(**ratio_kwargs)
        csq = np.sum(coeffs**2, axis=0, dtype=float)
        return self.scale_0 + csq/2.0

    def predictive(self, order=None, rescale=True, **ratio_kwargs):
        if not ratio_kwargs:
            ratio_kwargs = self._ratio_kwargs
        shape = self.shape(**ratio_kwargs)
        scale = self.scale(**ratio_kwargs)
        df = 2 * shape
        mu = 0
        sd = np.sqrt(scale / shape)

        # Geometric sum of ratio orders
        if order is None:
            order = np.inf
        k = self.max_order
        r2 = self.ratio(self.X, **ratio_kwargs)**2
        sd *= np.sqrt(r2**(k+1) * (1 - r2**(order-k)) / (1 - r2))

        # Create error bands around best prediction
        if rescale:
            sd *= np.abs(self.ref)
            mu = self.max_partial
        return st.t(df=df, loc=mu, scale=sd)

    def predict(self, dob=None, order=None, rescale=True, **ratio_kwargs):
        dist = self.predictive(order=order, rescale=rescale, **ratio_kwargs)
        return predictions(dist, dob=dob)

    def evidence(self, log=True, combine=False, **ratio_kwargs):
        if not ratio_kwargs:
            ratio_kwargs = self._ratio_kwargs
        shape = self.shape(**ratio_kwargs)
        scale = self.scale(**ratio_kwargs)
        coeffs = self._recompute_coeffs(**ratio_kwargs)
        num_c = coeffs.shape[0]

        # Compute evidence of coefficients elementwise
        ev = - 0.5 * num_c * np.log(2*np.pi)
        ev += sp.special.gammaln(shape) - shape * np.log(scale)
        if self.scale_0 != 0:
            shape_0 = self.shape_0
            scale_0 = self.scale_0
            ev += - sp.special.gammaln(shape_0) + shape_0 * np.log(scale_0)

        # Consider ratio and ref too
        ev -= num_c * np.log(np.abs(self.ref))
        ev -= np.sum(self.orders) * np.log(self.ratio(self.X, **ratio_kwargs))

        if combine:
            ev = np.sum(ev, axis=0, dtype=float)
        if not log:
            ev = np.exp(ev)
        return ev

    def posterior(self, name, logprior=None, log=False, **ratio_kwargs):
        def ev(val):
            kw = {name: np.squeeze(val)}
            return self.evidence(log=True, combine=True, **kw, **ratio_kwargs)

        log_pdf = 0
        if logprior is not None:
            log_pdf += logprior(**ratio_kwargs)

        vals = ratio_kwargs.pop(name)
        log_pdf += np.apply_along_axis(ev, 1, np.atleast_2d(vals).T)

        if not log:
            log_pdf -= np.max(log_pdf)
            pdf = np.exp(log_pdf)
            # Integrate using trapezoid rule
            norm = np.trapz(pdf, vals)
            return pdf/norm
        return log_pdf

    def credible_diagnostic(self, data, dobs, band_intervals=None,
                            band_dobs=None, samples=1e4, beta=True, **kwargs):
        dist = self.predictive(**kwargs)
        lower, upper = dist.interval(np.atleast_2d(dobs).T)
        # indicator = (lower < data) & (data < upper)  # 1 if within, 0 if out
        # return np.average(indicator, axis=1)   # The diagnostic

        def diagnostic(data, lower, upper):
            indicator = (lower < data) & (data < upper)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)   # The diagnostic

        D_CI = np.apply_along_axis(
                diagnostic, axis=1, arr=np.atleast_2d(data), lower=lower,
                upper=upper)
        D_CI = np.squeeze(D_CI)

        if band_intervals is not None:
            if band_dobs is None:
                band_dobs = dobs
            band_dobs = np.atleast_1d(band_dobs)

            N = self.partials.shape[1]
            # bands = np.zeros((len(band_intervals), len(band_dobs), 2))
            # # diag_line = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
            # for i, interval in enumerate(band_intervals):
            #     hpds = np.asarray([HPD(st.beta, interval, N*s+1, N-N*s+1) for s in band_dobs])
            #     # hpds = np.asarray([HPD(st.beta, s, N*interval+1, N-N*interval+1) for s in band_dobs])
            #     bands[i] = hpds
            # return D_CI, bands
            if beta:
                # band_dist = sp.stats.beta(a=N*band_dobs+1, b=N-N*band_dobs+1)
                # bands = np.apply_along_axis(
                #     HPD, axis=1, arr=band_intervals, dist=band_dist)
                band_intervals = np.atleast_1d(band_intervals)
                # bands = np.array([HPD(band_dist, p) for p in band_intervals])
                # Band shape: (len(dobs), 2, len(X))
                bands = np.zeros((len(band_intervals), 2, len(band_dobs)))
                for i, p in enumerate(band_intervals):
                    bands[i] = np.array(
                        [HPD(sp.stats.beta, p, N*s+1, N-N*s+1)
                         for s in band_dobs]).T
                # bands = np.transpose(bands, [0, 1, 2])
            else:
                band_dist = st.binom(n=N, p=band_dobs)
                band_intervals = np.atleast_2d(band_intervals)
                bands = np.asarray(band_dist.interval(band_intervals.T)) / N
                bands = np.transpose(bands, [1, 0, 2])
            return D_CI, bands
            # random_data = band_dist.rvs(size=(int(samples), len(band_dobs)))
            # band_lower, band_upper = band_dist.interval(band_dobs.T)
            # band_D_CI = np.apply_along_axis(
            #     diagnostic, axis=1, arr=random_data, lower=band_lower,
            #     upper=band_upper)
            # # bands = np.array(
            # #     [pm.hpd(band_D_CI, 1-bi) for bi in band_intervals])
            # bands = np.array(
            #     [np.percentile(band_D_CI, [100*(1-bi)/2, 100*(1+bi)/2], axis=0)
            #      for bi in band_intervals])
            # bands = np.transpose(bands, [0, 2, 1])
        return D_CI
