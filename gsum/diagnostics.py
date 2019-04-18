from __future__ import division
from .helpers import cholesky_errors, mahalanobis, lazy_property, \
    VariogramFourthRoot
from .cutils import pivoted_cholesky
import numpy as np
from numpy.linalg import solve
from scipy.linalg import cho_solve
import scipy.stats as st
from statsmodels.sandbox.distributions.mv_normal import MVT

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle

__all__ = ['Diagnostic', 'GraphicalDiagnostic']


class Diagnostic:
    R"""A class for quickly testing model checking methods discussed in Bastos & O'Hagan.

    """

    def __init__(self, mean, cov, df=None, random_state=1):
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
        self.dist.random_state = random_state
        self.udist.random_state = random_state
        self.std_udist.random_state = random_state

    @lazy_property
    def chol(self):
        """Returns the lower Cholesky matrix G where cov = G.G^T"""
        return np.linalg.cholesky(self.cov)

    @lazy_property
    def pivoted_chol(self):
        """Returns the pivoted Cholesky matrix G where cov = G.G^T"""
        return pivoted_cholesky(self.cov)

    @lazy_property
    def eig(self):
        """Returns the eigendecomposition matrix G where cov = G.G^T"""
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
        return solve(self.pivoted_chol, (y - self.mean).T).T

    def eigen_errors(self, y):
        return solve(self.eig, (y - self.mean).T).T

    def chi2(self, y):
        return np.sum(self.individual_errors(y), axis=-1)

    def md_squared(self, y):
        R"""The squared Mahalanobis distance"""
        return mahalanobis(y, self.mean, self.chol) ** 2

    def kl(self, mean, cov):
        R"""The Kullbeck-Leibler divergence"""
        m1, c1, chol1 = self.mean, self.cov, self.chol
        m0, c0 = mean, cov
        tr = np.trace(cho_solve((chol1, True), c0))
        dist = self.md_squared(m0)
        k = c1.shape[-1]
        logs = 2* np.sum(np.log(np.diag(c1))) - np.linalg.slogdet(c0)[-1]
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

        def diagnostic(data_, lower_, upper_):
            indicator = (lower_ < data_) & (data_ < upper_)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)  # The diagnostic

        dci = np.apply_along_axis(
            diagnostic, axis=1, arr=np.atleast_2d(y), lower=lower,
            upper=upper)
        dci = np.squeeze(dci)
        return dci

    def variogram(self, X, y, bin_bounds):
        v = VariogramFourthRoot(X, y, bin_bounds)
        bin_locations = v.bin_locations
        gamma, lower, upper = v.compute(rt_scale=False)
        return v, bin_locations, gamma, lower, upper


class GraphicalDiagnostic:

    def __init__(self, diagnostic, data, nref=1000, colors=None, markers=None):
        self.diagnostic = diagnostic
        self.data = data
        self.samples = self.diagnostic.samples(nref)
        prop_list = list(mpl.rcParams['axes.prop_cycle'])
        if colors is None:
            # The standard Matplotlib 2.0 colors, or whatever they've been updated to be.
            colors = [c['color'] for c in prop_list]
        if markers is None:
            markers = ['o' for c in prop_list]
        self.markers = markers
        self.marker_cycle = cycler('marker', colors)
        self.colors = colors
        self.color_cycle = cycler('color', colors)

    def error_plot(self, err, title=None, ylabel=None, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axhline(0, 0, 1, linestyle='-', color='k', lw=1, zorder=0)
        # The standardized 2 sigma bands since the sd has been divided out.
        sd = self.diagnostic.std_udist.std()
        ax.axhline(-2 * sd, 0, 1, linestyle='--', color='gray', label=r'$2\sigma$', zorder=0)
        ax.axhline(2 * sd, 0, 1, linestyle='--', color='gray', zorder=0)
        ax.set_prop_cycle(color=self.colors, marker=self.markers)
        index = np.arange(self.data.shape[-1])
        # print(np.arange(self.data.shape[-1]).shape, err.T.shape)
        for error in err:
            ax.plot(index, error, ls='')
        ax.set_xlabel('Index')
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
        ax.axvline(ref_mean - 2 * ref_sd, 0, 1, color='gray', linestyle='--', label=r'$2\sigma$')
        ax.axvline(ref_mean + 2 * ref_sd, 0, 1, color='gray', linestyle='--')
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

    def violin(self, data, ref, title=None, xlabel=None, ylabel=None, vlines=True, ax=None):
        import seaborn as sns
        import pandas as pd
        if ax is None:
            ax = plt.gca()
        n = len(data)
        nref = len(ref)
        orders = np.arange(n)
        zero = np.zeros(len(data), dtype=int)
        nans = np.nan * np.ones(nref)
        fake = np.hstack((np.ones(nref, dtype=bool), np.zeros(nref, dtype=bool)))
        fake_ref = np.hstack((fake[:, None], np.hstack((ref, nans))[:, None]))
        ref_df = pd.DataFrame(fake_ref, columns=['fake', title])
        tidy_data = np.hstack((orders[:, None], data[:, None]))
        print(tidy_data.shape, tidy_data)
        data_df = pd.DataFrame(tidy_data, columns=['orders', title])
        sns.violinplot(x=np.zeros(2 * nref, dtype=int), y=title, data=ref_df,
                       color='lightgrey', hue='fake', split=True, inner='box', ax=ax)
        sns.set_palette(self.colors)
        sns.swarmplot(x=zero, y=title, data=data_df, hue='orders', ax=ax)
        ax.set_xlabel('Density')
        ax.set_xlim(-0.05, 0.5)
        return ax

    def box(self, data, ref, title=None, xlabel=None, ylabel=None, vlines=True, trim=True, size=5, ax=None):
        import seaborn as sns
        import pandas as pd
        if ax is None:
            ax = plt.gca()
        n = len(data)
        nref = len(ref)
        orders = np.array([r'$c_{{{}}}$'.format(i) for i in range(n)])
        zero = np.zeros(len(data), dtype=int)
        # nans = np.nan * np.ones(nref)
        # fake = np.hstack((np.ones(nref, dtype=bool), np.zeros(nref, dtype=bool)))
        # fake_ref = np.hstack((fake[:, None], np.hstack((ref, nans))[:, None]))
        ref_df = pd.DataFrame(ref, columns=[title])
        tidy_data = np.array([orders, data], dtype=np.object).T
        # print(tidy_data)
        data_df = pd.DataFrame(tidy_data, columns=['orders', title])
        sns.boxplot(x=np.zeros(nref, dtype=int), y=title, data=ref_df,
                    color='lightgrey', ax=ax,
                    fliersize=0,
                    sym='',
                    whis=[2.5, 97.5],
                    bootstrap=None,
                    )
        sns.set_palette(self.colors)
        sns.swarmplot(x=zero, y=title, data=data_df, hue='orders', ax=ax, size=size)
        ax.set_xticks([])
        ax.legend(title=None)
        sns.despine(offset=0, trim=trim, bottom=True, ax=ax)
        return ax

    def qq(self, data, ref, band_perc, func, title=None, ax=None):
        data = np.sort(func(data.copy()), axis=-1)
        ref = np.sort(func(ref.copy()), axis=-1)
        bands = np.array([np.percentile(ref, [100 * (1. - bi) / 2, 100 * (1. + bi) / 2], axis=0)
                          for bi in band_perc])
        n = data.shape[-1]
        quants = (np.arange(1, n + 1) - 0.5) / n
        q_theory = self.diagnostic.std_udist.ppf(quants)

        if ax is None:
            ax = plt.gca()
        ax.set_prop_cycle(self.color_cycle)

        for i in range(len(band_perc) - 1, -1, -1):
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

    def md(self, ax=None, type='hist', **kwargs):
        if ax is None:
            ax = plt.gca()
        md_data = self.diagnostic.md(self.data)
        md_ref = self.diagnostic.md(self.samples)
        if type == 'violin':
            return self.violin(
                md_data, md_ref, title='Mahalanobis Distance',
                xlabel='MD', ylabel='density', ax=ax)
        elif type == 'hist':
            return self.hist(md_data, md_ref, title='Mahalanobis Distance',
                             xlabel='MD', ylabel='density', ax=ax, **kwargs)
        elif type == 'box':
            return self.box(
                md_data, md_ref, title='Mahalanobis Distance',
                xlabel='MD', ylabel='density', ax=ax, **kwargs)

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
        bands = np.array([np.percentile(dci_ref, [100 * (1. - bi) / 2, 100 * (1. + bi) / 2], axis=0)
                          for bi in band_perc])
        greys = mpl.cm.get_cmap('Greys')
        if ax is None:
            ax = plt.gca()
        # for i in range(len(band_perc)-1, -1, -1):
        #     ax.fill_between(intervals, bands[i, 0], bands[i, 1], alpha=1., color=greys((i+1)/(len(band_perc)+1)))
        band_perc = np.sort(band_perc)
        for i, perc in enumerate(band_perc):
            ax.fill_between(intervals, bands[i, 0], bands[i, 1], alpha=1.,
                            color=greys((len(band_perc) - i) / (len(band_perc) + 2.5)),
                            zorder=-perc)

        ax.plot([0, 1], [0, 1], c='k')
        ax.set_prop_cycle(self.color_cycle)
        ax.plot(intervals, dci_data.T)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Empirical Coverage')
        ax.set_xlabel('Credible Interval')
        ax.set_title('Credible Interval Diagnostic')
        return ax

    def variogram(self, X, ax=None):
        y = self.data
        N = len(X)
        nbins = np.ceil((N * (N - 1) / 2.) ** (1. / 3))
        bin_bounds = np.linspace(0, np.max(np.linalg.norm(X, axis=-1)), nbins)
        v, loc, gamma, lower, upper = self.diagnostic.variogram(X, y, bin_bounds)

        if ax is None:
            ax = plt.gca()

        ax.set_title('Variogram')
        ax.set_xlabel(r"$|x-x'|$")
        for i in range(y.shape[0]):
            ax.plot(loc, gamma[:, i], ls='', marker='o', c=self.colors[i])
            ax.plot(loc, lower[:, i], lw=0.5, c=self.colors[i])
            ax.plot(loc, upper[:, i], lw=0.5, c=self.colors[i])
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
        # self.variogram(X, axes[3, 2])
        fig.tight_layout()
        return fig, axes

    def essentials(self, vlines=True, bare=False):
        if bare:
            fig, axes = plt.subplots(1, 3, figsize=(7, 3))
            self.md(vlines=vlines, ax=axes[0])
            self.pivoted_cholesky_errors(axes[1])
            self.credible_interval(np.linspace(0, 1, 101), [0.68, 0.95], axes[2])
            axes[0].set_title('')
            axes[0].legend(title=r'$\mathrm{D}_{\mathrm{MD}}$')
            axes[0].set_ylabel('')
            axes[0].set_yticks([])
            axes[1].set_yticks([])
            axes[1].legend(title=r'$\mathrm{D}_{\mathrm{PC}}$')
            axes[1].set_title('')
            axes[1].set_ylabel('')
            axes[2].set_title('')
            axes[2].set_ylabel('')
            # axes[2].set_yticks([])
            axes[2].set_xticks([0, 0.5, 1])
            axes[2].set_xticklabels(['0', '0.5', '1'])
            axes[2].yaxis.tick_right()
            axes[2].text(0.05, 0.94, r'$\mathrm{D}_{\mathrm{CI}}$', transform=axes[2].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, ec='grey'))
            # axes[2].legend(title=r'$\mathrm{D}_{\mathrm{CI}}$')
            # plt.tight
            fig.tight_layout(h_pad=0.01, w_pad=0.1)
        else:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            self.md(vlines=vlines, ax=axes[0, 0])
            self.credible_interval(np.linspace(0, 1, 101), [0.68, 0.95], axes[1, 0])
            self.eigen_errors(axes[0, 1])
            self.eigen_errors_qq(axes[1, 1])
            self.pivoted_cholesky_errors(axes[0, 2])
            self.pivoted_cholesky_errors_qq(axes[1, 2])
            fig.tight_layout()
        return fig, axes
