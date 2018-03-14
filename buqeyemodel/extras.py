from __future__ import division
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np


__all__ = [
    'toy_data', 'coefficients', 'partials', 'stabilize', 'predictions',
    'gaussian'
]


def toy_data(orders, mu=0, sd=1, Q=0.5, ref=1, ls=None, noise=1e-6, X=None, size=None):
    # Set up covariance matrix
    if ls is not None:
        if size is not None:
            print('Warning: size parameter will be overriden')
        size = X.shape[0]
        with pm.Model():
            cov = sd**2 * pm.gp.cov.ExpQuad(input_dim=X.shape[1], ls=ls)
            K = cov(X) + noise**2 * tt.eye(size)
        # evaluate the covariance with the given hyperparameters
        K = theano.function([], K)()
    else:
        K = (sd**2 + noise**2) * np.eye(size)
    mean = mu*np.ones(size)
    coeffs = np.random.multivariate_normal(mean, K, size=len(orders))

    # Convert coefficients to observables
    ordervec = np.atleast_2d(orders).T
    obs_diffs = ref * coeffs * Q**(ordervec)
    obs = np.cumsum(obs_diffs, axis=0)
    return obs


def coefficients(partials, ratio, X=None, ref=1, orders=None, **ratio_kwargs):
    """Returns the coefficients of a power series

    Parameters
    ----------
    partials : 2d array
    ratio : 1d array, scalar, or callable
    X : 2d array (optional)
    ref : 1d array or scalar (optional)
    orders : 1d array (optional)
    rm_orders : 1d array (optional)

    Returns
    -------
    tuple
    An (n, N) array of the extracted coefficients and a (n,) array of their
    corresponding orders
    """
    if not callable(ratio):
        ratio_vals = ratio
    else:
        ratio_vals = ratio(X, **ratio_kwargs)

    if orders is None:
        orders = np.asarray(list(range(len(partials))))
    # if rm_orders is None:
    #     rm_orders = []

    if len(orders) != len(partials):
        raise ValueError('partials and orders must have the same length')

    # Make coefficients
    # Find differences but keep leading term
    coeffs = np.diff(partials, axis=0)
    coeffs = np.insert(coeffs, 0, partials[0], axis=0)
    # Scale each order appropriately
    ordervec = np.atleast_2d(orders).T
    coeffs = coeffs / (ref * ratio_vals**ordervec)

    # Remove unwanted orders
    # keepers = np.logical_not(np.isin(orders, rm_orders))
    # coeffs = coeffs[keepers]
    # orders = orders[keepers]
    return coeffs


def partials(coeffs, ratio, X=None, ref=1, orders=None, **ratio_kwargs):
    """Returns the partial sums of a power series given the coefficients

    The ``k``th partial sum is given by

    .. math::

        S_k = S_{\mathrm{ref}} \sum_{n=0}^k c_n r^n

    Parameters
    ----------
    coeffs : (n, N) array
        The n lowest order coefficients in a power series
    ratio : callable, scalar, or (N,) array
        The ratio variable that is raised to the nth power in the nth term of
        the sum
    X : (N, d) array, optional
        Input points passed to the ratio callable
    ref : (N,) array, optional
        The overall multiplicative scale of the series, default is 1
    orders : (n,) array, optional
        The orders corresponding to the given coefficients. All ungiven
        orders are assumed to have coefficients equal to zero. The default
        assumes that the n lowest order coefficients are given:
        ``[0, 1, ..., n-1]``.
    **ratio_kwargs : optional
        Keywords passed to the ratio callable

    Returns
    -------
    (n, N) array
        The partial sums
    """
    if callable(ratio):
        ratio_vals = ratio(X, **ratio_kwargs)
    else:
        ratio_vals = ratio

    if orders is None:
        orders = np.asarray(list(range(len(coeffs))))

    # Convert coefficients to partial sums
    ordervec = np.atleast_2d(orders).T
    terms = ref * coeffs * ratio_vals**(ordervec)
    partials = np.cumsum(terms, axis=0)
    return partials


def stabilize(M):
    return M + 1e-6 * np.eye(*M.shape)


def predictions(dist, dob=None):
    """Return the mean and set of degree of belief intervals for a distribution

    Parameters
    ----------
    dist : distribution object
    dob : scalar or 1D array

    Returns
    -------
    array or tuple
        If dob is None, just the mean is returned, else a tuple of the mean and
        degree of belief intervals is returned. The interval array is shaped
        (len(dob), 2, len(mean)) and is then squeezed to remove all axes of
        length 1.
    """
    mean = dist.mean()
    if dob is not None:
        dob = np.atleast_2d(dob).T
        interval = np.asarray(dist.interval(dob))
        # Make shape: (len(dobs), 2, len(X))
        interval = interval.transpose((1, 0, 2))
        # Remove unnecessary axes
        return mean, np.squeeze(interval)
    return mean


def gaussian(X, Xp=None, ls=1):
    """A gaussian correlation function

    Parameters
    ----------
    X : (N, d) array
    Xp : (M, d) array, optional
    ls : scalar
    """
    cov = pm.gp.cov.ExpQuad(input_dim=X.shape[-1], ls=ls)
    return cov(X, Xp).eval()
