from __future__ import division
from . import cholesky_errors, mahalanobis, VariogramFourthRoot
from . import pivoted_cholesky
import numpy as np
from numpy.linalg import solve, cholesky
from scipy.linalg import cho_solve
import scipy.stats as stats
from statsmodels.sandbox.distributions.mv_normal import MVT

import seaborn as sns
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle

__all__ = ['Diagnostic', 'GraphicalDiagnostic']


class Diagnostic:
    R"""A class for quickly testing model checking methods discussed in Bastos & O'Hagan.

    This class is under construction and the implementation may change in the future.

    Parameters
    ----------
    mean : array, shape = (n_samples,)
        The mean
    cov : array, shape = (n_samples, n_samples)
        The covariance
    df : int, optional
        The degrees of freedom. Defaults to `None`, which treats the distribution as Gaussian
    random_state : int, optional
        The random state for the random number generator
    """

    def __init__(self, mean, cov, df=None, random_state=1):
        self.mean = mean
        self.cov = cov
        self.sd = sd = np.sqrt(np.diag(cov))
        if df is None:
            # TODO: Handle when cov is ill-conditioned so multivariate_normal fails.
            self.dist = stats.multivariate_normal(mean=mean, cov=cov)
            # try:
            #     self.dist = stats.multivariate_normal(mean=mean, cov=cov)
            # except np.linalg.LinAlgError:
            #     self.dist = None
            self.udist = stats.norm(loc=mean, scale=sd)
            self.std_udist = stats.norm(loc=0., scale=1.)
        else:
            sigma = cov * (df - 2) / df
            self.dist = MVT(mean=mean, sigma=sigma, df=df)
            self.udist = stats.t(loc=mean, scale=sd, df=df)
            self.std_udist = stats.t(loc=0., scale=1., df=df)
        self.dist.random_state = random_state
        self.udist.random_state = random_state
        self.std_udist.random_state = random_state

        self._chol = cholesky(self.cov)
        self._pchol = pivoted_cholesky(self.cov)

        e, v = np.linalg.eigh(self.cov)
        # To match Bastos and O'Hagan definition
        # i.e., eigenvalues ordered from largest to smallest
        e, v = e[::-1], v[:, ::-1]
        ee = np.diag(np.sqrt(e))
        self._eig = v @ ee

    def samples(self, n):
        R"""Sample random variables

        Parameters
        ----------
        n : int
            The number of curves to sample

        Returns
        -------
        array, shape = (n_samples, n_curves)
        """
        return self.dist.rvs(n).T

    def individual_errors(self, y):
        R"""Computes the scaled individual errors diagnostic

        .. math::
            D_I(y) = \frac{y-m}{\sigma}

        Parameters
        ----------
        y : array, shape = (n_samples, [n_curves])

        Returns
        -------
        array : shape = (n_samples, [n_curves])
        """
        return ((y.T - self.mean) / np.sqrt(np.diag(self.cov))).T

    def cholesky_errors(self, y):
        return cholesky_errors(y.T, self.mean, self._chol).T

    def pivoted_cholesky_errors(self, y):
        return solve(self._pchol, (y.T - self.mean).T)

    def eigen_errors(self, y):
        return solve(self._eig, (y.T - self.mean).T)

    def chi2(self, y):
        return np.sum(self.individual_errors(y), axis=0)

    def md_squared(self, y):
        R"""Computes the squared Mahalanobis distance"""
        return mahalanobis(y.T, self.mean, self._chol) ** 2

    def kl(self, mean, cov):
        R"""The Kullback-Leibler divergence between two multivariate normal distributions

        .. math::
            D_{KL}(N_0 | N_1) = \frac{1}{2} \left [
                \mathrm{Tr}(\Sigma_1^{-1}\Sigma_0)
              + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1 - \mu_0)
              - k + \log\left(\frac{\det \Sigma_1}{\det \Sigma_0}\right)
            \right]

        where :math:`k` is the dimension of Normal distributions. The :math:`\mu_1` and :math:`\Sigma_1` are those
        fed during the initialization of the Diagnostic object, and :math:`\mu_0` and :math:`\Sigma_0` are the
        arguments of this function.

        Parameters
        ----------
        mean : array, shape = (n_samples,)
        cov : array, shape = (n_samples, n_samples)

        Returns
        -------
        float
            The KL divergence
        """
        m1, c1, chol1 = self.mean, self.cov, self._chol
        m0, c0 = mean, cov
        tr = np.trace(cho_solve((chol1, True), c0))
        dist = self.md_squared(m0)
        k = c1.shape[-1]
        logs = 2 * np.sum(np.log(np.diag(c1))) - np.linalg.slogdet(c0)[-1]
        return 0.5 * (tr + dist - k + logs)

    def credible_interval(self, y, intervals):
        """The credible interval diagnostic.

        Parameters
        ----------
        y : (n_samples, [n_curves]) shaped array
        intervals : 1d array
            The credible intervals at which to perform the test

        Returns
        -------
        array, shape = ([n_curves], n_intervals)
        """
        lower, upper = self.udist.interval(np.atleast_2d(intervals).T)

        def diagnostic(data_, lower_, upper_):
            indicator = (lower_ < data_) & (data_ < upper_)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)  # The diagnostic

        dci = np.apply_along_axis(
            diagnostic, axis=1, arr=np.atleast_2d(y).T, lower_=lower, upper_=upper)
        if y.ndim == 1:
            dci = np.squeeze(dci)  # If y didn't have n_curves dim, then remove it now.
        return dci

    @staticmethod
    def variogram(X, y, bin_bounds):
        R"""Computes the variogram for the data y at input points X.

        Parameters
        ----------
        X
        y
        bin_bounds

        Returns
        -------
        v : array
        bin_locations :
        gamma :
        lower :
        upper :
        """
        v = VariogramFourthRoot(X, y, bin_bounds)
        bin_locations = v.bin_locations
        gamma, lower, upper = v.compute(rt_scale=False)
        return v, bin_locations, gamma, lower, upper


class GraphicalDiagnostic:
    R"""A class for plotting diagnostics and their reference distributions.

    This class is under construction and the implementation may change in the future.

    Parameters
    ----------
    data : array, shape = (n_samples, n_curves)
        The data to compute diagnostics against
    mean : array
        The mean for the diagnostic object
    cov : array
        The covariance of the diagnostic object
    df : int, optional
        If a Student-t distribution, then this is the degrees of freedom. If `None`, it is
        treated as Gaussian
    random_state : int, optional
    nref : int
        The number of samples to use in computing a reference distribution by simulation
    colors : list
        The colors to use for each curve
    markers : list
        The markers to use for each curve, where applicable.

    Examples
    --------

    """

    # See: https://ianstormtaylor.com/design-tip-never-use-black/
    # soft_black = '#262626'

    def __init__(self, data, mean, cov, df=None, random_state=1, nref=1000, colors=None, markers=None, labels=None,
                 gray='lightgray', black='#262626', markeredgecolors=None, markerfillstyles=None):
        self.diagnostic = Diagnostic(mean=mean, cov=cov, df=df, random_state=random_state)
        if data.ndim == 1:
            data = np.atleast_2d(data).T  # Add n_curves dim if it doesn't exist
        self.data = data
        self.samples = self.diagnostic.samples(nref)
        prop_list = list(mpl.rcParams['axes.prop_cycle'])
        if colors is None:
            # The standard Matplotlib 2.0 colors, or whatever they've been updated to be.
            colors = [c['color'] for c in prop_list]
        if markers is None:
            markers = ['o' for c in prop_list]
        if markeredgecolors is None:
            markeredgecolors = [None for c in prop_list]
        if markerfillstyles is None:
            markerfillstyles = ['full' for c in prop_list]
        if labels is None:
            labels = np.array([r'$c_{{{}}}$'.format(i) for i in range(data.shape[-1])])
        self.labels = labels
        self.markers = markers
        self.markeredgecolors = markeredgecolors
        self.markerfillstyles = markerfillstyles
        self.marker_cycle = cycler('marker', colors)
        self.colors = colors
        self.color_cycle = cycler('color', colors)
        self.gray = gray
        self.black = black

        n = len(cov)
        if df is None:
            self.md_ref_dist = stats.chi2(df=n)
        else:
            self.md_ref_dist = stats.f(dfn=n, dfd=df, scale=(df-2)*n/df)

    def error_plot(self, err, title=None, xlabel='Index', ylabel=None, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axhline(0, 0, 1, linestyle='-', color=self.black, lw=1, zorder=0)
        # The standardized 2 sigma bands since the sd has been divided out.
        sd = self.diagnostic.std_udist.std()
        ax.axhline(-2 * sd, 0, 1, color=self.gray, zorder=0, lw=1)
        ax.axhline(2 * sd, 0, 1, color=self.gray, zorder=0, lw=1)
        index = np.arange(1, self.data.shape[0]+1)
        size = 8

        if err.ndim == 1:
            err = err[:, None]
        for i, error in enumerate(err.T):
            ax.plot(
                index, error, ls='', color=self.colors[i],
                marker=self.markers[i], markeredgecolor=self.markeredgecolors[i],
                fillstyle=self.markerfillstyles[i], markersize=size, markeredgewidth=0.5
            )
            # ax.scatter(
            #     index, error, color=self.colors[i], marker=self.markers[i],
            #     edgecolor=self.markeredgecolors[i], linestyle=self.markerlinestyles[i]
            # )
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.margins(x=0.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax

    def individual_errors(self, title='Individual Errors', ax=None):
        err = self.diagnostic.individual_errors(self.data)
        return self.error_plot(err, title=title, ax=ax)

    def individual_errors_qq(self, title='Individual QQ Plot', ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.individual_errors,
                       title=title, ax=ax)

    def cholesky_errors(self, title='Cholesky Errors', ax=None):
        err = self.diagnostic.cholesky_errors(self.data)
        return self.error_plot(err, title=title, ax=ax)

    def cholesky_errors_qq(self, title='Cholesky QQ Plot', ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.cholesky_errors,
                       title=title, ax=ax)

    def pivoted_cholesky_errors(self, title='Pivoted Cholesky Errors', ax=None):
        err = self.diagnostic.pivoted_cholesky_errors(self.data)
        return self.error_plot(err, title=title, ax=ax)

    def pivoted_cholesky_errors_qq(self, title='Pivoted Cholesky QQ Plot', ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.pivoted_cholesky_errors,
                       title=title, ax=ax)

    def eigen_errors(self, title='Eigen Errors', ax=None):
        err = self.diagnostic.eigen_errors(self.data)
        return self.error_plot(err, title=title, ax=ax)

    def eigen_errors_qq(self, title='Eigen QQ Plot', ax=None):
        return self.qq(self.data, self.samples, [0.68, 0.95], self.diagnostic.eigen_errors,
                       title=title, ax=ax)

    def hist(self, data, ref, title=None, xlabel=None, ylabel=None, vlines=True, ax=None):

        if hasattr(ref, 'ppf'):
            lower_95 = ref.ppf(0.975)
            upper_95 = ref.ppf(0.025)
            x = np.linspace(lower_95, upper_95, 100)
            ax.plot(x, ref.pdf(x), label='ref', color=self.black)
        else:
            ref_stats = stats.describe(ref)
            ref_sd = np.sqrt(ref_stats.variance)
            ref_mean = ref_stats.mean
            # This doesn't exactly match 95% intervals from distribution
            lower_95 = ref_mean - 2 * ref_sd
            upper_95 = ref_mean + 2 * ref_sd
            ax.hist(ref, density=1, label='ref', histtype='step', color=self.black)

        if ax is None:
            ax = plt.gca()

        ax.axvline(lower_95, 0, 1, color='gray', linestyle='--', label=r'$2\sigma$')
        ax.axvline(upper_95, 0, 1, color='gray', linestyle='--')
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

    def violin(self, data, ref, title=None, xlabel=None, ylabel=None, ax=None):
        if ax is None:
            ax = plt.gca()
        n = len(data)
        nref = len(ref)
        orders = np.arange(n)
        zero = np.zeros(len(data), dtype=int)
        nans = np.nan * np.ones(nref)
        fake = np.hstack((np.ones(nref, dtype=bool), np.zeros(nref, dtype=bool)))
        fake_ref = np.hstack((fake[:, None], np.hstack((ref, nans))[:, None]))

        label = 'label_'  # Placeholder
        ref_df = pd.DataFrame(fake_ref, columns=['fake', label])
        tidy_data = np.hstack((orders[:, None], data[:, None]))
        data_df = pd.DataFrame(tidy_data, columns=['orders', label])
        sns.violinplot(x=np.zeros(2 * nref, dtype=int), y=label, data=ref_df,
                       color=self.gray, hue='fake', split=True, inner='box', ax=ax)
        with sns.color_palette(self.colors):
            sns.swarmplot(x=zero, y=label, data=data_df, hue='orders', ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(-0.05, 0.5)
        return ax

    def box(self, data, ref, title=None, xlabel=None, ylabel=None, trim=True, size=8, legend=False, ax=None):
        if ax is None:
            ax = plt.gca()

        label = 'labelll'  # Placeholder

        # Plot reference dist
        if hasattr(ref, 'ppf'):
            gray = 'gray'
            boxartist = self._dist_boxplot(
                ref, ax=ax, positions=[0],
                patch_artist=True,
                widths=0.8)
            for box in boxartist['boxes']:
                box.update(dict(facecolor='lightgrey', edgecolor=gray))
            for whisk in boxartist["whiskers"]:
                whisk.update(dict(color=gray))
            for cap in boxartist["caps"]:
                cap.update(dict(color=gray))
            for med in boxartist["medians"]:
                med.update(dict(color=gray))
        else:
            nref = len(ref)
            ref_df = pd.DataFrame(ref, columns=[label])
            sns.boxplot(
                x=np.zeros(nref, dtype=int), y=label, data=ref_df, color='lightgrey', ax=ax, fliersize=0, sym='',
                whis=[2.5, 97.5], bootstrap=None,
            )

        # Plot data
        n = len(data)
        orders = np.array([r'$c_{{{}}}$'.format(i) for i in range(n)])
        zero = np.zeros(len(data), dtype=int)
        tidy_data = np.array([orders, data], dtype=np.object).T
        data_df = pd.DataFrame(tidy_data, columns=['orders', label])
        data_df[label] = data_df[label].astype(float)
        # print(data_df)
        from matplotlib.markers import MarkerStyle
        with sns.color_palette(self.colors):
            # Only use this to get the right positions
            ss = sns.swarmplot(
                x=zero, y=label, data=data_df,
                hue='orders',
                ax=ax, size=size,
                linewidth=0.5,
                # marker=[MarkerStyle('o', fillstyle=style) for style in self.markerfillstyles]
                # marker='left'
                # marker=MarkerStyle('o', fillstyle='left')
                # marker=MarkerStyle('o', fillstyle='top'),
                # facecolor='none',
                # facecoloralt='w',
                # color='none',
                # alpha=0
            )
            # Swarmplot plots markers in an order from smallest to largest
            # This rearranges the marker line styles to be in that order
            positions = ss.collections[0].get_offsets()  # These are ordered by swarmplot
            ss.collections[0].remove()  # Don't show swarmplot, we will plot below
            _, idx, inv = np.unique(data_df[label].values, return_index=True, return_inverse=True)
            # positions = positions[idx]
            positions = positions[inv]
            assert np.allclose(positions[:, -1], data_df[label].values)
        for i, (x, y) in enumerate(positions):
            ax.plot(
                [x], [y],
                marker=self.markers[i], ls='', markersize=size,
                zorder=5+i,
                c=self.colors[i], fillstyle=self.markerfillstyles[i],
                markeredgecolor=self.markeredgecolors[i], markeredgewidth=0.5, clip_on=False
            )
            # linestyles_new = np.array(self.markerlinestyles)[inv]
            # print(ss.collections[0].get_fill())
            # # collections[0] *should* be the markers created by swarmplot... but might not always be?
            # ss.collections[0].set_dashes(linestyles_new)

        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if legend:
            ax.legend(title=None)
        else:
            ax.get_legend().remove()
        sns.despine(offset=0, trim=trim, bottom=True, ax=ax)
        return ax

    @staticmethod
    def _dist_boxplot(dist, q1=0.25, q3=0.75, whislo=0.025, whishi=0.975, label=None, ax=None, other_stats=None,
                     **kwargs):
        """Creates a boxplot computed from a Scipy.stats-like distribution."""
        if ax is None:
            ax = plt.gca()
        stat_dict = [{'med': dist.median(), 'q1': dist.ppf(q1), 'q3': dist.ppf(q3),
                      'whislo': dist.ppf(whislo), 'whishi': dist.ppf(whishi)}]
        if label is not None:
            stat_dict[0]['label'] = label
        if other_stats is not None:
            stat_dict = [*stat_dict, *other_stats]
        return ax.bxp(stat_dict, showfliers=False, **kwargs)

    def qq(self, data, ref, band_perc, func, title=None, ax=None):
        data = np.sort(func(data.copy()), axis=0)
        ref = np.sort(func(ref.copy()), axis=0)
        bands = np.array([np.percentile(ref, [100 * (1. - bi) / 2, 100 * (1. + bi) / 2], axis=1)
                          for bi in band_perc])
        n = data.shape[0]
        quants = (np.arange(1, n + 1) - 0.5) / n
        q_theory = self.diagnostic.std_udist.ppf(quants)

        if ax is None:
            ax = plt.gca()

        for i in range(len(band_perc) - 1, -1, -1):
            ax.fill_between(q_theory, bands[i, 0], bands[i, 1], alpha=0.5, color='gray')

        for i, dat in enumerate(data.T):
            ax.plot(q_theory, dat, c=self.colors[i], label=self.labels[i])
        yl, yu = ax.get_ylim()
        xl, xu = ax.get_xlim()
        ax.plot([xl, xu], [xl, xu], c=self.black)
        ax.set_ylim([yl, yu])
        ax.set_xlim([xl, xu])
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Quantiles')
        return ax

    def md_squared(self, ax=None, type='hist', title='Mahalanobis Distance', xlabel='MD', **kwargs):
        if ax is None:
            ax = plt.gca()
        md_data = self.diagnostic.md_squared(self.data)
        if type == 'hist':
            return self.hist(md_data, self.md_ref_dist, title=title,
                             xlabel=xlabel, ax=ax, **kwargs)
        elif type == 'box':
            return self.box(
                md_data, self.md_ref_dist, title=title,
                xlabel=xlabel, ax=ax, **kwargs)

    def kl(self, X, gp, predict=False, vlines=True, title='KL Divergence', xlabel='KL', ax=None):
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
        return self.hist(kl_data, kl_ref, title=title,
                         xlabel=xlabel, vlines=vlines, ax=ax)

    def credible_interval(self, intervals, band_perc, title='Credible Interval Diagnostic',
                          xlabel='Credible Interval', ylabel='Empirical Coverage', ax=None, linestyles=None):
        dci_data = self.diagnostic.credible_interval(self.data, intervals)
        dci_ref = self.diagnostic.credible_interval(self.samples, intervals)
        bands = np.array([np.percentile(dci_ref, [100 * (1. - bi) / 2, 100 * (1. + bi) / 2], axis=0)
                          for bi in band_perc])
        greys = mpl.cm.get_cmap('Greys')
        if ax is None:
            ax = plt.gca()
        band_perc = np.sort(band_perc)
        for i, perc in enumerate(band_perc):
            ax.fill_between(intervals, bands[i, 0], bands[i, 1], alpha=1.,
                            color=greys((len(band_perc) - i) / (len(band_perc) + 2.5)),
                            zorder=-perc)

        ax.plot([0, 1], [0, 1], c=self.black)
        for i, data in enumerate(dci_data):
            if linestyles is None:
                ls = None
            else:
                ls = linestyles[i]
            ax.plot(intervals, data, color=self.colors[i], ls=ls, label=self.labels[i])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        return ax

    def variogram(self, X, title='Variogram', xlabel='Lag', ax=None):
        y = self.data
        N = len(X)
        nbins = np.ceil((N * (N - 1) / 2.) ** (1. / 3))
        bin_bounds = np.linspace(0, np.max(np.linalg.norm(X, axis=-1)), nbins)
        v, loc, gamma, lower, upper = self.diagnostic.variogram(X, y, bin_bounds)

        if ax is None:
            ax = plt.gca()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        for i in range(y.shape[0]):
            ax.plot(loc, gamma[:, i], ls='', marker='o', c=self.colors[i])
            ax.plot(loc, lower[:, i], lw=0.5, c=self.colors[i])
            ax.plot(loc, upper[:, i], lw=0.5, c=self.colors[i])
        return ax

    def plotzilla(self, X, gp=None, predict=False, vlines=True):
        R"""A convenience method for plotting a lot of diagnostics at once.

        """
        if gp is None:
            pass
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        self.md_squared(vlines=vlines, ax=axes[0, 0])
        if gp is not None:
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

    def essentials(self, vlines=True, bare=False):
        R"""A convenience method for plotting the essential diagnostics quickly.

        Parameters
        ----------
        vlines
        bare

        Returns
        -------

        """
        if bare:
            fig, axes = plt.subplots(1, 3, figsize=(7, 3))
            self.md_squared(vlines=vlines, ax=axes[0])
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
            axes[2].set_xticks([0, 0.5, 1])
            axes[2].set_xticklabels(['0', '0.5', '1'])
            axes[2].yaxis.tick_right()
            axes[2].text(0.05, 0.94, r'$\mathrm{D}_{\mathrm{CI}}$', transform=axes[2].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, ec='grey'))
            fig.tight_layout(h_pad=0.01, w_pad=0.1)
        else:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            self.md_squared(vlines=vlines, ax=axes[0, 0])
            self.credible_interval(np.linspace(0, 1, 101), [0.68, 0.95], axes[1, 0])
            self.eigen_errors(axes[0, 1])
            self.eigen_errors_qq(axes[1, 1])
            self.pivoted_cholesky_errors(axes[0, 2])
            self.pivoted_cholesky_errors_qq(axes[1, 2])
            fig.tight_layout()
        return fig, axes
