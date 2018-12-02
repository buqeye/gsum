from __future__ import division
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import scipy as sp
from scipy.optimize import fmin
from functools import wraps
import inspect


__all__ = [
    'cartesian', 'toy_data', 'coefficients', 'partials', 'stabilize',
    'predictions', 'gaussian', 'HPD', 'KL_Gauss', 'rbf', 'default_attributes',
    'cholesky_errors', 'mahalanobis'
]


def cartesian(*arrays):
    """Makes the Cartesian product of arrays.

    Parameters
    ----------
    arrays: 1D array-like
            1D arrays where earlier arrays loop more slowly than later ones
    """
    N = len(arrays)
    return np.stack(np.meshgrid(*arrays, indexing='ij'), -1).reshape(-1, N)


# def toy_data(orders, mu=0, sd=1, Q=0.5, ref=1, ls=None, noise=1e-6, X=None, size=None):
#     # Set up covariance matrix
#     if ls is not None:
#         if size is not None:
#             print('Warning: size parameter will be overriden')
#         size = X.shape[0]
#         with pm.Model():
#             cov = sd**2 * pm.gp.cov.ExpQuad(input_dim=X.shape[1], ls=ls)
#             K = cov(X) + noise**2 * tt.eye(size)
#         # evaluate the covariance with the given hyperparameters
#         K = theano.function([], K)()
#     else:
#         K = (sd**2 + noise**2) * np.eye(size)
#     mean = mu*np.ones(size)
#     coeffs = np.random.multivariate_normal(mean, K, size=len(orders))

#     # Convert coefficients to observables
#     ordervec = np.atleast_2d(orders).T
#     obs_diffs = ref * coeffs * Q**(ordervec)
#     obs = np.cumsum(obs_diffs, axis=0)
#     return obs

def toy_data(X, orders, basis=None, corr=None, beta=0, sd=1, ratio=0.5,
             ref=1, noise=1e-5, ratio_kwargs=None, **corr_kwargs):
    # Set up covariance matrix
    if corr is None:
        corr = gaussian
    K = sd**2 * corr(X, **corr_kwargs)
    K += noise**2 * np.eye(K.shape[0])

    if basis is None:
        basis = np.ones((len(X), 1))
    else:
        basis = basis(X)
    mean = np.dot(basis, np.atleast_1d(beta))

    if ratio_kwargs is None:
        ratio_kwargs = {}
    coeffs = np.random.multivariate_normal(mean, K, size=len(orders))
    return partials(coeffs=coeffs, ratio=ratio, X=X, ref=ref,
                    orders=orders, **ratio_kwargs)


def generate_coefficients(X, size=1, basis=None, corr=None, beta=0, sd=1,
                          noise=1e-5, **corr_kwargs):
    # Set up covariance matrix
    if corr is None:
        corr = gaussian
    K = sd**2 * corr(X, **corr_kwargs)
    K += noise**2 * np.eye(K.shape[0])

    if basis is None:
        basis = np.ones((len(X), 1))
    else:
        basis = basis(X)
    mean = np.dot(basis, np.atleast_1d(beta))
    return np.random.multivariate_normal(mean, K, size=size)


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
    R"""Returns the partial sums of a power series given the coefficients

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
    return M + 1e-5 * np.eye(*M.shape)


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
    X = X * 1.0/ls
    X2 = np.sum(X**2, axis=1)
    if Xp is None:
        Xp = X
    Xp2 = np.sum(Xp**2, axis=1)
    sqd = -2.0 * np.dot(X, Xp.T) + (np.reshape(X2, (-1, 1)) + np.reshape(Xp2, (1, -1)))
    sqd = np.clip(sqd, 0.0, np.inf)
    return np.exp(-0.5 * sqd)
    # cov = pm.gp.cov.ExpQuad(input_dim=X.shape[-1], ls=ls)
    # return cov(X, Xp).eval()


def rbf(X, Xp=None, ls=1):
    if Xp is None:
        Xp = X
    diff = X[:, None, ...] - Xp[None, ...]
    dist = np.linalg.norm(diff, axis=-1)
    if ls == 0:
        return np.where(dist == 0, 1., 0.)
    return np.exp(- dist**2 / ls**2)


def HPD(dist, alpha, *args):
    R"""Returns the highest probability density interval of scipy dist.

    Inspired by this answer https://stackoverflow.com/a/25777507
    """
    # Freeze dist if args provided
    if args:
        dist = dist(*args)

    def interval_length(start):
        return dist.ppf(start + alpha) - dist.ppf(start)
    # find start of cdf interval that minimizes the pdf interval length
    start = fmin(interval_length, 1-alpha, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return dist.ppf([start, alpha + start])


def HPD_pdf(pdf, alpha, x, opt_kwargs=None, *args):
    R"""Returns the highest probability density interval given the pdf.

    Inspired by this answer https://stackoverflow.com/a/22290087
    """
    # if not callable(pdf):
    #     pdf = sp.interpolate.interp1d(x, pdf)
    #     args = []
    if opt_kwargs is None:
        opt_kwargs = {}

    lb, ub = np.min(x), np.max(x)

    def errfn(p, alpha, *args):
        # def fn(xx):
        #     f = pdf(xx, *args)
        #     return f if f > p else 0
        # prob = integrate.quad(fn, lb, ub)[0]
        if callable(pdf):
            pdf_array = pdf(x, *args)
        else:
            pdf_array = pdf
        f = np.zeros(len(pdf_array))
        mask = pdf_array > p
        f[mask] = pdf_array[mask]
        prob = np.trapz(f, x)
        return (prob - alpha)**2

    hline = fmin(errfn, x0=0, args=(alpha, *args), **opt_kwargs)[0]
    if callable(pdf):
        mask = pdf(x, *args) > hline
    else:
        mask = pdf > hline
    interval = np.asarray(x)[mask]
    return np.min(interval), np.max(interval)


def KL_Gauss(mu0, cov0, mu1, cov1=None, chol1=None):
    R"""The Kullbeck-Liebler divergence between two mv Gaussians
    
    The divergence from :math:`\mathcal{N}_1` to :math:`\mathcal{N}_0` is given by

    .. math::

        D_\text{KL}(\mathcal{N}_0 \| \mathcal{N}_1) = \frac{1}{2} \left[ \mathrm{Tr} \left( \Sigma_1^{-1} \Sigma_0 \right) + \left( \mu_1 - \mu_0\right)^\text{T} \Sigma_1^{-1} ( \mu_1 - \mu_0 ) - k + \ln \left( \frac{\det \Sigma_1}{\det \Sigma_0} \right)  \right],
    
    which can be thought of as the amount of information lost when :math:`\mathcal{N}_1`
    is used to approximate :math:`\mathcal{N}_0`.

    Parameters
    ----------
    mu0 : Scalar or 1d array
        The mean of the posterior
    cov0 : Scalar or 2d array
        The covariance of the posterior
    mu1 : Scalar or 1d array
        The mean of the prior
    cov1 : Scalar or 2d array
        The covariance of the prior
    chol1 : Scalar or 2d array
        The cholesky docomposition of the prior
    
    Returns
    -------
    number
        The KL divergence
    
    Raises
    ------
    ValueError
        Exactly one of cov1 or chol1 must be given
    """
    mu0, mu1 = np.atleast_1d(mu0), np.atleast_1d(mu1)
    cov0 = np.atleast_2d(cov0)
    if chol1 is not None and cov1 is None:
        chol1 = np.atleast_2d(chol1)
    elif cov1 is not None and chol1 is None:
        cov1 = np.atleast_2d(cov1)
        chol1 = np.linalg.cholesky(stabilize(cov1))
    else:
        raise ValueError('Exactly one of cov1 or chol1 must be given.')
    
    k = cov0.shape[0]
    _, logdet0 = np.linalg.slogdet(cov0)
    logdet1 = 2*np.sum(np.log(np.diag(chol1)))

    right_quad = np.linalg.solve(chol1, mu1-mu0)
    quad = np.dot(right_quad.T, right_quad)

    tr_mat = np.trace(sp.linalg.cho_solve((chol1, True), cov0))

    return 0.5*(tr_mat + quad - k + logdet1 - logdet0)


def lazy_property(function):
    R"""Stores as a hidden attribute if it doesn't exist yet, else it returns the hidden attribute.
    
    This means that any method decorated with this function will only be computed if necessary, and will be
    stored if needed to be called again.
    """
    attribute = '_cache_' + function.__name__

    @property
    @wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator            


def lazy(function):
    attribute = '_cache_' + function.__name__

    @wraps(function)
    def decorator(self, *args, **kwargs):
        lazy = True

        # If y are passed and are not the defaults
        y = kwargs.pop('y', None)
        if y is not None and not np.allclose(y, self.y):
            lazy = False
        else:
            y = self.y

        # If cholesky is passed and are not the defaults
        corr_chol = kwargs.pop('corr_chol', None)
        if corr_chol is not None and not np.allclose(corr_chol, self._corr_chol):
            lazy = False
        else:
            corr_chol = self._corr_chol

        if not lazy or not hasattr(self, attribute):
            setattr(self, attribute, function(self, *args, y=y, corr_chol=corr_chol, **kwargs))
        return getattr(self, attribute)
    return decorator


def default_attributes(**kws):
    """Sets `None` or empty `*args`/`**kwargs` arguments to attributes already stored in a class.
    
    This is a handy decorator to avoid `if` statements at the beginning of method
    calls:
    
    def func(self, x=None):
        if x is None:
            x = self.x
        ...
    
    but is particularly useful when the function uses a cache to avoid
    unnecessary computations. Caches don't recognize when the attributes change,
    so could result in incorrect returned values. This decorator **must** be put
    outside of the cache decorator though.
    
    Parameters
    ----------
    kws : dict
        The key must match the parameter name in the decorated function, and the value
        corresponds to the name of the attribute to use as the default
        
    Example
    -------
    from fastcache import lru_cache
    
    class TestClass:
    
        def __init__(self, x, y):
            self.x = x
            self._y = y

        @lru_cache()
        def add(self, x=None, y=None):
            if x is None:
                x = self.x
            if y is None:
                y = self._y
            return x + y

        @lru_cache()
        @default_attributes(x='x', y='_y')
        def add2(self, x=None, y=None):
            return x + y

        @default_attributes(x='x', y='_y')
        @lru_cache()
        def add_correctly(self, x=None, y=None):
            return x + y

    tc = TestClass(2, 3)
    print(tc.add(), tc.add2(), tc.add_correctly())  # Prints 5 5 5
    tc.x = 20
    print(tc.add(), tc.add2(), tc.add_correctly())  # Prints 5 5 23
    tc._y = 5
    print(tc.add(), tc.add2(), tc.add_correctly())  # Prints 5 5 25
    """
    def decorator(function):
        sig = inspect.signature(function)

        @wraps(function)
        def new_func(self, *args, **kwargs):
            # Puts all arguments---positional, keyword, and default---explicitly in bound_args
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for key, value in bound_args.arguments.items():
                param = sig.parameters[key]
                if isinstance(value, np.ndarray):
                    continue
                
                # Update standard arguments if they are `None`, but also allow for
                # *args and **kwargs to be set to defaults if they are empty.
                # Standard arguments:
                default_poskey = value is None and param.kind == param.POSITIONAL_OR_KEYWORD
                # Keyword only arguments (comes after *args):
                default_key = value is None and param.kind == param.KEYWORD_ONLY
                # *args argument:
                default_varpos = value == () and param.kind == param.VAR_POSITIONAL
                # **kwargs argument:
                default_varkey = value == {} and param.kind == param.VAR_KEYWORD

                if (default_poskey or default_key or default_varpos or default_varkey) and key in kws:
                    bound_args.arguments[key] = getattr(self, kws[key])
            return function(*bound_args.args, **bound_args.kwargs)
        return new_func
    return decorator

vec_solve_triangular = np.vectorize(sp.linalg.solve_triangular, excluded=['lower'], signature='(m,m),(m,n)->(m,n)')

def cholesky_errors(y, mean, chol):
    y = np.atleast_2d(y)
    return np.squeeze(np.swapaxes(vec_solve_triangular(chol, (y - mean).T, lower=True), -1, -2))

# def chol_errors(y, mean, chol):
#     return sp.linalg.solve_triangular(chol, (y - mean).T, lower=True).T

def mahalanobis(y, mean, chol):
    err = cholesky_errors(y, mean, chol)
    return np.linalg.norm(err, axis=-1)
