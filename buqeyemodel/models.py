from __future__ import division
from functools import reduce
from .helpers import coefficients, predictions, gaussian, stabilize, \
    cartesian, HPD
import numpy as np
import pymc3 as pm
import scipy as sp
import scipy.integrate as integrate
import scipy.stats as st
from statsmodels.sandbox.distributions.mv_normal import MVT
import theano
import theano.tensor as tt
import warnings


__all__ = ['SGP', 'PowerProcess', 'PowerSeries']


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
        self._chol = np.linalg.cholesky(stabilize(self.corr(X, **corr_kwargs)))
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

    def chol(self, **corr_kwargs):
        if self._recompute_corr(**corr_kwargs):
            return np.linalg.cholesky(stabilize(self.corr(self.X, **corr_kwargs)))
        else:
            return self._chol
        # return np.linalg.cholesky(stabilize(self.corr(self.X, **corr_kwargs)))

    @property
    def y(self):
        return self._y

    @property
    def X(self):
        return self._X

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
        """Returns the posterior pdf for arbitrary correlation variables

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
        """Returns the posterior pdf for arbitrary correlation or ratio variables

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
