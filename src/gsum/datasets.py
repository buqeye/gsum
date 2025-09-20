import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state
from . import cartesian, partials


def make_gaussian_partial_sums(
        X, orders=5, kernel=None, mean=None, ratio=0.3,
        ref=1., nugget=0, random_state=0, allow_singular=True
):
    R"""
    Generates a dataset of Gaussian partial sums at the input points X.

    Parameters
    ----------
    X : array, shape = (n_samples, n_features)
        The input locations at which to sample the Gaussian process coefficients
    orders : int or array, optional (default = 5)
        The orders included in the partial sum. If an int is provided, then the partial sums from [0, 1, ..., orders-1]
        are generated. If orders is an array, then only the partial sums in `orders` are returned, assuming that any
        order not in `orders` does not contribute to the sum (i.e. its coefficient is zero).
    kernel : callable
        The kernel specifying the covariance function of the GP.
        If None is passed, the kernel `RBF(0.5)` is used as default.
    mean : callable
        The mean function of the series coefficients
    ratio : float or callable
        The ratio in the geometric sum.
    ref : float or callable
        The overall scale factor of the geometric sum
    nugget : float, optional (default = 0)
        Value added to the diagonal of the covariance matrix.
        Larger values correspond to increased noise level in the observations.
        This can also prevent potential numerical issues, by
        ensuring that the calculated values form a positive definite matrix.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    allow_singular : bool, optional (default = True)
        Whether to allow a singular covariance matrix.

    Returns
    -------
    X : array, shape = (n_samples, n_features)
        The input points.
    y : array, shape = (n_samples,)
        The response values.
    """
    if kernel is None:
        kernel = RBF(0.5)
    if mean is None:
        def mean(a):
            return np.zeros(a.shape[0])

    if isinstance(orders, int):
        orders = np.arange(orders)
    if callable(ratio):
        ratio = ratio(X)
    if callable(ref):
        ref = ref(X)

    m = mean(X)
    K = kernel(X)
    K += nugget * np.eye(K.shape[0])

    dist = stats.multivariate_normal(mean=m, cov=K, allow_singular=allow_singular)
    coeffs = dist.rvs(len(orders), random_state=random_state).T
    y = partials(coeffs=coeffs, ratio=ratio, ref=ref, orders=orders)
    return y


def make_gaussian_partial_sums_uniform(
        n_samples=100, n_features=1, orders=5, kernel=None, mean=None, ratio=0.3, ref=1.,
        nugget=0, random_state=0, allow_singular=True
):
    R"""
    Generates a dataset of Gaussian partial sums at random input locations.

    The input X randomly sampled from [0, 1] in n_features dimensions.

    Parameters
    ----------
    n_samples : int, optional (default = 100)
        The number of samples from each feature dimension.
    n_features : int, optional (default = 1)
        The number of features.
    orders : int or array, optional (default = 5)
        The orders included in the partial sum. If an int is provided, then the partial sums from [0, 1, ..., orders-1]
        are generated. If orders is an array, then only the partial sums in `orders` are returned, assuming that any
        order not in `orders` does not contribute to the sum (i.e. its coefficient is zero).
    kernel : callable
        The kernel specifying the covariance function of the GP.
        If None is passed, the kernel `RBF(0.5)` is used as default.
    mean : callable
        The mean function of the series coefficients
    ratio : float or callable
        The ratio in the geometric sum.
    ref : float or callable
        The overall scale factor of the geometric sum
    nugget : float, optional (default = 0)
        Value added to the diagonal of the covariance matrix.
        Larger values correspond to increased noise level in the observations.
        This can also prevent potential numerical issues, by
        ensuring that the calculated values form a positive definite matrix.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    allow_singular : bool, optional (default = True)
        Whether to allow a singular covariance matrix.

    Returns
    -------
    X : array, shape = (n_samples, n_features)
        The input points.
    y : array, shape = (n_samples,)
        The response values.
    """
    generator = check_random_state(random_state)
    X = generator.rand(n_samples, n_features)
    y = make_gaussian_partial_sums(
        X=X, orders=orders, kernel=kernel, mean=mean, ratio=ratio, ref=ref,
        nugget=nugget, random_state=random_state, allow_singular=allow_singular
    )
    return X, y


def make_gaussian_partial_sums_on_grid(
        n_samples=100, n_features=1, orders=5, kernel=None, mean=None, ratio=0.3, ref=1.,
        nugget=0, random_state=0, allow_singular=True
):
    R"""
    Generates a dataset of Gaussian partial sums on a full grid.

    The input X is n_samples from [0, 1], which is then put on a full grid in n_features dimensions.

    Parameters
    ----------
    n_samples : int, optional (default = 100)
        The number of samples from each feature dimension.
    n_features : int, optional (default = 1)
        The number of features.
    orders : int or array, optional (default = 5)
        The orders included in the partial sum. If an int is provided, then the partial sums from [0, 1, ..., orders-1]
        are generated. If orders is an array, then only the partial sums in `orders` are returned, assuming that any
        order not in `orders` does not contribute to the sum (i.e. its coefficient is zero).
    kernel : callable
        The kernel specifying the covariance function of the GP.
        If None is passed, the kernel `RBF(0.5)` is used as default.
    mean : callable
        The mean function of the series coefficients
    ratio : float or callable
        The ratio in the geometric sum.
    ref : float or callable
        The overall scale factor of the geometric sum
    nugget : float, optional (default = 0)
        Value added to the diagonal of the covariance matrix.
        Larger values correspond to increased noise level in the observations.
        This can also prevent potential numerical issues, by
        ensuring that the calculated values form a positive definite matrix.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    allow_singular : bool, optional (default = True)
        Whether to allow a singular covariance matrix.

    Returns
    -------
    X : array, shape = (n_samples ** n_features, n_features)
        The input points.
    y : array, shape = (n_samples ** n_features,)
        The response values.
    """
    x = np.linspace(0, 1, n_samples)
    if n_features > 1:
        X = cartesian(*[x for x in range(n_features)])
    else:
        X = x[:, None]

    y = make_gaussian_partial_sums(
        X=X, orders=orders, kernel=kernel, mean=mean, ratio=ratio, ref=ref,
        nugget=nugget, random_state=random_state, allow_singular=allow_singular
    )
    return X, y
