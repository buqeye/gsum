from __future__ import division
import docrep
from .helpers import coefficients, hpd, mahalanobis, geometric_sum
import numpy as np
from numpy.linalg import solve, cholesky
import scipy as sp
from scipy.linalg import cho_solve, solve_triangular, inv, eigh
from scipy.special import loggamma
import scipy.stats as st
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning

import warnings
from operator import itemgetter


__all__ = [
    'ConjugateGaussianProcess', 'ConjugateStudentProcess',
    'TruncationGP', 'TruncationTP', 'TruncationPointwise'
]

docstrings = docrep.DocstringProcessor()


@docstrings.get_sectionsf('BaseConjugateProcess')
@docstrings.dedent
class BaseConjugateProcess:
    """
    The base class for the stochastic process estimator.

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the correlation function of the GP.
        The covariance matrix is the kernel multiplied by the squared scale.
        If None is passed, the kernel "RBF(1.0)" is used as default.
        Note that the kernel’s hyperparameters are optimized during fitting.
    center : float
        The prior central values for the parameters of the mean function.
    disp : float >= 0
        The dispersion parameter for the normal prior placed on the mean. This, multiplied by the squared scale
        parameter from the inverse chi squared prior, determines the variance of the mean.
        The smaller the dispersion, the better determined is the mean.
        Set this to zero for a mean that is known to be `mean`.
    df : float > 0
        The degrees of freedom parameter for the inverse chi squared prior placed on the marginal variance.
        This is a measure of how well the marginal standard deviation (or variance) is known, with
        larger degrees of freedom implying a better known standard deviation. Set this to infinity for a
        standard deviation that is known to be `scale`, or use the `sd` keyword argument.
    scale : float > 0
        The scale parameter of the scaled inverse chi squared prior placed on the marginal variance
        of the Gaussian process. Approximately the prior standard deviation for the Gaussian process.
    sd : float > 0, optional
        A convenience argument that sets the marginal standard deviation for the Gaussian process.
        This is equivalent to setting df0 to infinity and scale0 to sd
        (i.e., a delta function prior on the standard deviation).
    nugget : float, optional (default: 1e-10)
        Value added to the diagonal of the correlation matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    """
    
    def __init__(self, kernel=None, center=0, disp=0, df=1, scale=1, sd=None, basis=None, nugget=1e-10,
                 optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, copy_X_train=True, random_state=None,
                 decomposition='cholesky'):
        self.kernel = kernel

        # Setup hyperparameters
        self._center_0 = np.atleast_1d(center)
        self._disp_0 = np.atleast_2d(disp)
        if sd is not None:
            self._df_0 = np.inf
            self._scale_0 = sd
        else:
            self._df_0 = df
            self._scale_0 = scale

        # Break with scikit learn convention here. Define all attributes in __init__.
        # Use value of self._fit to determine whether the `fit` has been called.
        self._fit = False
        self.X_train_ = None
        self.y_train_ = None
        self.corr_L_ = self.corr_sqrt_ = None
        self.corr_ = None
        self.center_ = None
        self.disp_ = None
        self.df_ = None
        self.scale_ = None
        self.cov_factor_ = None
        self.cbar_sq_mean_ = None
        self.kernel_ = None
        self._rng = None
        self._eigh_tuple_ = None

        self.nugget = nugget
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.optimizer = optimizer
        self.decomposition = decomposition

        self._default_kernel = ConstantKernel(1.0, constant_value_bounds='fixed') * \
            RBF(1.0, length_scale_bounds='fixed')

        if basis is None:
            self.basis = lambda X: np.ones((X.shape[0], 1))
        self.basis_train_ = None

    @property
    def center0(self):
        return self._center_0

    @property
    def disp0(self):
        return self._disp_0

    @property
    def df0(self):
        return self._df_0

    @property
    def scale0(self):
        return self._scale_0

    @classmethod
    def compute_center(cls, y, sqrt_R, basis, center0, disp0, decomposition, eval_gradient=False, dR=None):
        R"""Computes the regression coefficients' center hyperparameter :math:`\eta` updated based on data

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
            The data to condition upon
        sqrt_R : array, shape = (n_samples, n_samples)
            The decomposition of the correlation matrix. Its value depends on `decomposition`
        basis : array, shape = (n_samples, n_param)
            The basis matrix that multiplies the regression coefficients beta to create the GP mean.
        center0 : scalar or array, shape = (n_param)
            The prior regression coefficients for the mean
        disp0 : scalar or array, shape = (n_param, n_param)
            The prior dispersion for the regression coefficients
        decomposition : str
            The way that R has been decomposed into sqrt_R: either 'cholesky' or 'eig'.
        eval_gradient : bool, optional
            Whether to return the gradient with respect to the kernel hyperparameters. Defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params), optional
            The gradient of the correlation matrix. This is required if eval_gradient is True.

        Returns
        -------
        center : scalar or array, shape = (n_param)
            The posterior regression coefficients for the mean
        grad_center : array, shape = (n_param, n_kernel_params), optional
            The gradient of the posterior regression coefficients for the mean with respect to kernel hyperparameters
        """
        # Mean is not updated if its prior variance is zero (i.e. delta function prior)
        # Do by hand to prevent dividing by zero
        if np.all(disp0 == 0):
            if eval_gradient:
                if dR is None:
                    raise ValueError('dR must be given if eval_gradient is True')
                return np.copy(center0), np.zeros((*center0.shape, dR.shape[-1]))
            return np.copy(center0)

        y_avg = cls.avg_y(y)
        ny = cls.num_y(y)

        # if decomposition == 'cholesky':
        #     invR_y_avg = cho_solve((sqrt_R, True), y_avg)
        # elif decomposition == 'eig':
        #     invR_y_avg = solve(sqrt_R, y_avg)
        # else:
        #     raise ValueError('decomposition must be either "cholesky" or "eig"')
        invR_y_avg = cls.solve_sqrt(sqrt_R, y=y_avg, decomposition=decomposition)
        disp = cls.compute_disp(y=y, sqrt_R=sqrt_R, basis=basis, disp0=disp0, decomposition=decomposition)
        factor = solve(disp0, center0) + ny * basis.T @ invR_y_avg
        center = disp @ factor

        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            # invR_basis = cho_solve((chol, True), basis)
            # invR_diff = cho_solve((chol, True), basis @ center - y_avg)
            invR_basis = cls.solve_sqrt(sqrt_R, y=basis, decomposition=decomposition)
            invR_diff = cls.solve_sqrt(sqrt_R, y=basis @ center - y_avg, decomposition=decomposition)
            d_center = ny * disp @ np.einsum('ji,jkp,k->ip', invR_basis, dR, invR_diff)
            return center, d_center
        return center

    @classmethod
    def compute_disp(cls, y, sqrt_R, basis, disp0, decomposition, eval_gradient=False, dR=None):
        R"""The dispersion hyperparameter :math:`V` updated based on data.

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
            The data to condition upon
        sqrt_R : (n_samples, n_samples)-shaped array
            The lower Cholesky decomposition of the correlation matrix
        basis : (n_samples, n_param)-shaped array
            The basis for the `p` regression coefficients `beta`
        disp0 : (n_param, n_param)-shaped array
            The prior dispersion
        eval_gradient : bool, optional
            Whether to return the gradient with respect to the kernel hyperparameters. Defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params), optional
            The gradient of the correlation matrix. This is required if eval_gradient is True.

        Returns
        -------
        disp : (p, p)-shaped array
            The updated dispersion hyperparameter
        grad_disp : array, shape = (p,p), optional
        """
        # If prior variance is zero, it stays zero
        # Do by hand to prevent dividing by zero
        if np.all(disp0 == 0):
            if eval_gradient:
                if dR is None:
                    raise ValueError('dR must be given if eval_gradient is True')
                return np.zeros_like(disp0), np.zeros((*disp0.shape, dR.shape[-1]))
            return np.zeros_like(disp0)

        ny = cls.num_y(y)
        # quad = mahalanobis(basis.T, 0, chol) ** 2
        quad = basis.T @ cls.solve_sqrt(sqrt_R, y=basis, decomposition=decomposition)
        disp = inv(inv(disp0) + ny * quad)
        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            # invRBV = cho_solve((chol, True), basis) @ disp
            invRBV = cls.solve_sqrt(sqrt_R, y=basis, decomposition=decomposition) @ disp
            dV = ny * np.einsum('ji,jkp,kl->ilp', invRBV, dR, invRBV)
            return disp, dV
        return disp

    @classmethod
    def compute_df(cls, y, df0, eval_gradient=False, dR=None):
        R"""Computes the degrees of freedom hyperparameter :math:`\nu` based on data

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
            The data to condition upon
        df0 : scalar
            The prior degrees of freedom
        eval_gradient : bool, optional
            Whether to return the gradient with respect to the kernel hyperparameters. Defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params), optional
            The gradient of the correlation matrix. This is required if eval_gradient is True.

        Returns
        -------
        df : scalar
            The updated degrees of freedom
        grad_df : array, size = (n_kernel_params,), optional
            The gradient of the updated degrees of freedom with respect to the kernel parameters
        """
        df = df0 + y.size
        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            return df, np.zeros(dR.shape[-1])
        return df

    @classmethod
    def compute_scale_sq_v2(cls, y, sqrt_R, basis, center0, disp0, df0, scale0, decomposition,
                            eval_gradient=False, dR=None):
        R"""The squared scale hyperparameter :math:`\tau^2` updated based on data.

        Parameters
        ----------
        y : array, shape = (n_samples, [n_curves])
            The data to condition upon
        chol : array, shape = (n_samples, n_samples)
            The lower Cholesky decomposition of the correlation matrix
        basis : array, shape = (n_samples, n_param)
            The basis for the `p` regression coefficients `beta`
        center0 : scalar or array, shape = (n_param)
            The prior regression coefficients for the mean
        disp0 : array, shape = (n_param, n_param)
            The prior dispersion
        df0 : scalar
            The prior degrees of freedom hyperparameter
        scale0 : scalar
            The prior scale hyperparameter
        decomposition
        eval_gradient : bool, optional
            Whether to return to the gradient with respect to kernel hyperparameters. Defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params), optional
            The gradient of the correlation matrix. This is required if eval_gradient is True.

        Returns
        -------
        scale_sq : scalar
            The updated scale hyperparameter squared
        grad_scale_sq : array, shape = (n_kernel_params,), optional
            The gradient of scale^2 with respect to the kernel hyperparameters. Only returned if eval_gradient is True.
        """
        if df0 == np.inf:
            if eval_gradient:
                return scale0**2, np.zeros(dR.shape[-1])
            return scale0**2

        avg_y, ny = cls.avg_y(y), cls.num_y(y)

        # Compute contributions from a non-zero mean
        if np.all(disp0 == 0):
            # The disp -> 0 limit must be taken carefully to find these terms
            center = center0
            # invR_diff0 = cho_solve((chol, True), 2 * avg_y - basis @ center)
            invR_diff0 = cls.solve_sqrt(sqrt_R, 2 * avg_y - basis @ center, decomposition=decomposition)
            mean_terms = - ny * center0 @ basis.T @ invR_diff0
        else:
            center = cls.compute_center(
                y=y, sqrt_R=sqrt_R, basis=basis, center0=center0, disp0=disp0, decomposition=decomposition)
            disp = cls.compute_disp(y=y, sqrt_R=sqrt_R, basis=basis, disp0=disp0, decomposition=decomposition)
            mean_terms = center0 @ inv(disp0) @ center0 - center @ inv(disp) @ center

        # Combine the prior info, quadratic form, and mean contributions to find scale**2
        if y.ndim == 1:
            y = y[:, None]
        # invR_y = cho_solve((chol, True), y)
        invR_y = cls.solve_sqrt(sqrt_R, y=y, decomposition=decomposition)
        quad = np.trace(y.T @ invR_y)
        df = cls.compute_df(y=y, df0=df0)
        scale_sq = (df0 * scale0**2 + mean_terms + quad) / df

        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is true')
            # Both the disp -> 0 and non-zero forms have the same gradient formula
            d_scale_sq = - np.einsum('ij,jkp,ki->p', invR_y.T, dR, invR_y)  # From the quadratic form
            # invR_diff = cho_solve((chol, True), 2 * avg_y - basis @ center)
            # invR_basis_center = cho_solve((chol, True), basis) @ center
            invR_diff = cls.solve_sqrt(sqrt_R, 2 * avg_y - basis @ center, decomposition=decomposition)
            invR_basis_center = cls.solve_sqrt(sqrt_R, basis, decomposition=decomposition) @ center
            d_scale_sq += ny * np.einsum('i,ijp,j->p', invR_basis_center, dR, invR_diff)
            d_scale_sq /= df
            return scale_sq, d_scale_sq
        return scale_sq

    @classmethod
    def compute_scale_sq(cls, y, sqrt_R, basis, center0, disp0, df0, scale0, decomposition,
                         eval_gradient=False, dR=None):
        R"""The squared scale hyperparameter :math:`\tau^2` updated based on data.

        Parameters
        ----------
        y : ndarray, shape = (n_samples, [n_curves])
            The data to condition upon
        sqrt_R : ndarray, shape = (n_samples, n_samples)
            The lower Cholesky decomposition of the correlation matrix
        basis : ndarray, shape = (n_samples, n_param)
            The basis for the `p` regression coefficients `beta`
        center0 : int or float or array, shape = (n_param)
            The prior regression coefficients for the mean
        disp0 : ndarray, shape = (n_param, n_param)
            The prior dispersion
        df0 : int or float
            The prior degrees of freedom hyperparameter
        scale0 : int or float
            The prior scale hyperparameter
        eval_gradient : bool, optional
            Whether to return to the gradient with respect to kernel hyperparameters. Defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params)
            The gradient of the correlation matrix. This is required if eval_gradient is True.

        Returns
        -------
        scale_sq : scalar
            The updated scale hyperparameter squared
        grad_scale_sq : array, shape = (n_kernel_params,), optional
            The gradient of scale^2 with respect to the kernel hyperparameters. Only returned if eval_gradient is True.
        """
        if df0 == np.inf:
            if eval_gradient:
                return scale0**2, np.zeros(dR.shape[-1])
            return scale0**2

        if y.ndim == 1:
            y = y[:, None]
        avg_y = cls.avg_y(y)
        N = len(avg_y)
        ny = cls.num_y(y)

        y_centered = y - avg_y[:, None]
        # invR_yc = cho_solve((chol, True), y_centered)
        invR_yc = cls.solve_sqrt(sqrt_R, y_centered, decomposition=decomposition)
        quad = np.trace(y_centered.T @ invR_yc)

        avg_y_centered = avg_y - basis @ center0
        disp = cls.compute_disp(
            y=y, sqrt_R=sqrt_R, basis=basis, disp0=disp0, decomposition=decomposition, eval_gradient=False)
        invR_basis = cls.solve_sqrt(sqrt_R, basis, decomposition=decomposition)
        invR_avg_yc = cls.solve_sqrt(sqrt_R, avg_y_centered, decomposition=decomposition)
        # Use the Woodbury matrix identity on Melendez et al Eq. (A31):
        mat = np.eye(N) - ny * invR_basis @ disp @ basis.T
        mat_invR_avg_yc = ny * mat @ invR_avg_yc
        # mat = np.eye(N) - ny * cho_solve((chol, True), basis) @ disp @ basis.T
        # mat_invR_avg_yc = ny * mat @ cho_solve((chol, True), avg_y_centered)
        quad2 = avg_y_centered @ mat_invR_avg_yc

        df = cls.compute_df(y=y, df0=df0)
        scale_sq = (df0 * scale0 ** 2 + quad + quad2) / df

        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is true')
            d_scale_sq = - np.einsum('ji,jkp,ki->p', invR_yc, dR, invR_yc)
            d_scale_sq -= np.einsum('i,ijp,j->p', mat_invR_avg_yc, dR, mat_invR_avg_yc) / ny
            d_scale_sq /= df
            return scale_sq, d_scale_sq
        return scale_sq

    @staticmethod
    def solve_sqrt(sqrt_mat, y, decomposition):
        R"""Solves a system Mx = y given sqrt_M and y.

        Parameters
        ----------
        sqrt_mat : array
            The square root of a matrix. If decomposition is 'eig', then this can be a tuple (eig, Q) such that
            M = Q @ np.diag(eig) @ Q.T. This can speed up the inversion due to the simple property that
            M^-1 = Q @ np.diag(1/eig) @ Q.T.
        y : array
        decomposition : str
            The way that the square root has been performed. Either 'cholesky' or 'eig'. If cholesky,
            then it is assumed that sqrt_mat is the lower triangular matrix `L` such that `M = L L.T`.

        Returns
        -------
        x
        """
        if decomposition == 'cholesky':
            return cho_solve((sqrt_mat, True), y)
        elif decomposition == 'eig':
            if isinstance(sqrt_mat, tuple):
                eig, Q = sqrt_mat
                inv_mat = Q @ np.diag(1. / eig) @ Q.T
                return inv_mat @ y
            return solve(sqrt_mat.T, solve(sqrt_mat, y))
        else:
            raise ValueError('decomposition must be either "cholesky" or "eig"')

    @staticmethod
    def compute_cov_factor(scale_sq, df):
        R"""Converts the squared scale hyperparameter :math:`\tau^2` to the correlation -> covariance conversion factor

        The conversion is given by :math:`\sigma^2 = \nu \tau^2 / (\nu - 2)` for :math:`\nu > 2`

        Warnings
        --------
        If the correlation matrix does equal 1 on the diagonal, :math:`\sigma^2` **will not**
        be the marginal variance. Instead, one must look at the diagonal of the covariance directly.
        """
        var = scale_sq
        if df != np.inf:
            var = df * scale_sq / (df - 2)
        return var

    def center(self):
        """The regression coefficient hyperparameters for the mean updated by the call to `fit`.
        """
        if self.decomposition == 'cholesky':
            sqrt_R = self.corr_sqrt_
        elif self.decomposition == 'eig':
            sqrt_R = self._eigh_tuple_
        else:
            raise ValueError('decomposition must be either "cholesky" or "eig"')
        return self.compute_center(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_,
            center0=self.center0, disp0=self.disp0, decomposition=self.decomposition)

    def disp(self):
        """The dispersion hyperparameter updated by the call to `fit`.
        """
        if self.decomposition == 'cholesky':
            sqrt_R = self.corr_sqrt_
        elif self.decomposition == 'eig':
            sqrt_R = self._eigh_tuple_
        else:
            raise ValueError('decomposition must be either "cholesky" or "eig"')
        return self.compute_disp(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_, disp0=self.disp0,
            decomposition=self.decomposition)

    def df(self):
        """The degrees of freedom hyperparameter for the standard deviation updated by the call to `fit`
        """
        return self.compute_df(y=self.y_train_, df0=self.df0)

    def scale(self):
        """The scale hyperparameter for the standard deviation updated by the call to `fit`
        """
        if self.decomposition == 'cholesky':
            sqrt_R = self.corr_sqrt_
        elif self.decomposition == 'eig':
            sqrt_R = self._eigh_tuple_
        else:
            raise ValueError('decomposition must be either "cholesky" or "eig"')
        scale_sq = self.compute_scale_sq(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_,
            center0=self.center0, disp0=self.disp0, df0=self.df0, scale0=self.scale0,
            decomposition=self.decomposition)
        return np.sqrt(scale_sq)

    def mean(self, X):
        """The MAP value for the mean of the process at inputs X with hyperparameters updated by y.

        This does not interpolate the y values. For that functionality, use `predict`.
        """
        if not self._fit:  # Unfitted; predict based on GP prior
            center = self.center0
        else:
            center = self.center_
        return self.basis(X) @ center

    def cov(self, X, Xp=None):
        R"""Computes the covariance matrix.

        If `fit` has not been called, then this uses the prior values of `df` and `scale` and the default
        unoptimized kernel. Otherwise it uses the posterior values of `df` and `scale`, and the optimized kernel.
        This does not return the conditional covariance matrix. For that, use `predict`.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
        Xp : array, optional, shape = (n_samples2, n_features)

        Returns
        -------
        array, shape = (n_samples, n_samples2)

        Raises
        ------
        ValueError if the degrees of freedom is less than 2, since the covariance does not exist in this case.
        This could happen if `fit` is not called and the provided `df` is less than 2.
        """
        # Don't fill in Xp because WhiteKernel will not work correctly
        # if Xp is None:
        #     Xp = X

        if not self._fit:  # Unfitted; predict based on GP prior
            if self.df0 <= 2:
                raise ValueError('df must be greater than 2 for the covariance to exist')
            cov_factor = self.compute_cov_factor(scale_sq=self.scale0**2, df=self.df0)
            if self.kernel is None:
                kernel = self._default_kernel
            else:
                kernel = self.kernel
        else:
            cov_factor = self.cov_factor_
            kernel = self.kernel_

        return cov_factor * kernel(X, Xp)
    
    @staticmethod
    def num_y(y):
        """Computes the number of curves in y"""
        ny = 1
        if y.ndim == 2:
            ny = y.shape[1]
        return ny

    @staticmethod
    def avg_y(y):
        """Computes the average of y over the set of curves

        Parameters
        ----------
        y : array, shape = (n_samples, [n_curves])
            The data

        Returns
        -------
        avg_y : array, shape = (n_samples,)
            The average of y over the set of curves
        """
        if y.ndim == 1:
            return np.copy(y)
        elif y.ndim == 2:
            return np.average(y, axis=1)
        else:
            raise ValueError('y must be two-dimensional, not shape={}'.format(y.shape))

    def _calibrate_kernel(self):
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            optima = np.array(optima)
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)
    
    def fit(self, X, y):
        R"""Fits the process to data (X, y) and updates all hyperparameters.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The input variables where the response is observed
        y : array, shape = (n_samples, [n_curves])
            The response values

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = clone(self._default_kernel)
        else:
            self.kernel_ = clone(self.kernel)
        self._rng = check_random_state(self.random_state)

        # X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        if self.copy_X_train:
            try:
                self.X_train_ = X.copy()
            except AttributeError:
                self.X_train_ = np.copy(X)

            try:
                self.y_train_ = y.copy()
            except AttributeError:
                self.y_train_ = np.copy(y)
        else:
            self.X_train_ = X
            self.y_train_ = y
        self.basis_train_ = self.basis(self.X_train_)

        self._calibrate_kernel()
        self.corr_ = self.kernel_(X)

        if self.decomposition == 'cholesky':
            self.corr_L_ = self.corr_sqrt_ = cholesky(self.corr_ + self.nugget * np.eye(len(X)))
            sqrt_R = self.corr_sqrt_
        elif self.decomposition == 'eig':
            eig, Q = eigh(self.corr_ + self.nugget * np.eye(len(X)))
            self._eigh_tuple_ = eig, Q
            sqrt_R = eig, Q  # Passing tuple makes matrix inversion easier later on
            self.corr_L_ = self.corr_sqrt_ = Q @ np.diag(np.sqrt(eig))
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')

        self.center_ = self.compute_center(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_,
            center0=self.center0, disp0=self.disp0, decomposition=self.decomposition
        )
        self.disp_ = self.compute_disp(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_, disp0=self.disp0,
            decomposition=self.decomposition
        )
        self.df_ = self.compute_df(y=self.y_train_, df0=self.df0)
        scale_sq = self.compute_scale_sq(
            y=self.y_train_, sqrt_R=sqrt_R, basis=self.basis_train_,
            center0=self.center0, disp0=self.disp0, df0=self.df0, scale0=self.scale0,
            decomposition=self.decomposition
        )
        self.scale_ = np.sqrt(scale_sq)
        self.cov_factor_ = self.cbar_sq_mean_ = self.compute_cov_factor(scale_sq=scale_sq, df=self.df_)
        self._fit = True
        return self

    def underlying_properties(self, X, return_std=False, return_cov=False):
        y_mean = self.mean(X)
        if return_cov:
            y_cov = self.cov(X)
            return y_mean, y_cov
        elif return_std:
            y_std = np.sqrt(np.diag(self.cov(X)))
            return y_mean, y_std
        else:
            return y_mean

    @docstrings.get_sectionsf('BaseConjugateProcess_predict')
    @docstrings.dedent
    def predict(self, X, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=False):
        """
        Predict using the Gaussian process regression model at the points `X`

        Calling `predict` before calling `fit` will use the GP prior.
        In addition to the mean of the predictive distribution, its standard deviation (return_std=True)
        or covariance (return_cov=True) can be returned. Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Locations at which to predict the new y values
        return_std : bool, optional (default = False)
            Whether the marginal standard deviation of the predictive process is to be returned
        return_cov : bool, optional (default = False)
            Whether the covariance matrix of the predictive process is to be returned
        Xc : array, shape = (n_conditional_samples, n_features)
            Locations at which to condition. Defaults to `X` used in fit. This *does not*
            affect the `X` used to update hyperparameters.
        y : array, shape = (n_conditional_samples, [n_curves])
            Points upon which to condition. Defaults to the `y` used in `fit`. This *does not*
            affect the `y` used to update hyperparameters.
        pred_noise : bool, optional
            Adds `nugget` to the diagonal of the covariance matrix if `return_cov == True`.

        Returns
        -------
        y_mean : array, shape = (n_curves, n_samples)
            Mean of predictive distribution at query points
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution at query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError('Only one of return_std or return_cov may be True')

        if not self._fit:  # Unfitted; predict based on GP prior
            return self.underlying_properties(X=X, return_std=return_std, return_cov=return_cov)

        decomp = self.decomposition

        if Xc is None:
            Xc = self.X_train_
            if decomp == 'cholesky':
                sqrt_R = self.corr_sqrt_
            elif decomp == 'eig':
                sqrt_R = self._eigh_tuple_
            else:
                raise ValueError('decomposition must be "cholesky" or "eig"')
        else:
            # corr_chol = cholesky(self.kernel_(Xc) + self.nugget * np.eye(len(Xc)))
            kk = self.kernel_(Xc) + self.nugget * np.eye(len(Xc))
            if decomp == 'cholesky':
                sqrt_R = cholesky(kk)
            elif decomp == 'eig':
                sqrt_R = eigh(kk)  # eig, Q
            else:
                raise ValueError('decomposition must be "cholesky" or "eig"')
        if y is None:
            y = self.y_train_

        # Use X and y from fit for hyperparameters
        m_old = self.mean(Xc)
        m_new = self.mean(X)

        # Now use X and y from arguments for conditioning/predictions
        R_on = self.kernel_(Xc, X)
        R_no = R_on.T
        R_nn = self.kernel_(X)  # Only use one argument, otherwise, e.g., WhiteKernel won't work right

        if y.ndim == 1:
            y = y[:, None]

        # Use given y for prediction
        # alpha = cho_solve((corr_chol, True), (y - m_old[:, None]))
        alpha = self.solve_sqrt(sqrt_R, (y - m_old[:, None]), decomposition=decomp)
        m_pred = np.squeeze(m_new[:, None] + R_no @ alpha)
        if return_std or return_cov:
            # half_quad = solve_triangular(corr_chol, R_on, lower=True)
            # R_pred = R_nn - half_quad.T @ half_quad
            R_pred = R_nn - R_no @ self.solve_sqrt(sqrt_R, R_on, decomposition=decomp)
            if pred_noise:
                R_pred += self.nugget * np.eye(len(X))
            # Use y from fit for hyperparameters
            var = self.compute_cov_factor(scale_sq=self.scale_**2, df=self.df_)
            K_pred = np.squeeze(var * R_pred)
            if return_std:
                return m_pred, np.sqrt(np.diag(K_pred))
            return m_pred, K_pred
        return m_pred

    def sample_y(self, X, n_samples=1, random_state=0, underlying=False):
        """Draw samples from Gaussian process and evaluate at X. (Taken from scikit-learn's gp module)

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
        n_samples : int, optional (default = 1)
        random_state : int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.

        Returns
        -------
        y_samples : array, shape = (n_samples, [n_curves])
            Output samples from the GP at input points X.
        """
        rng = check_random_state(random_state)

        if underlying:
            y_mean, y_cov = self.underlying_properties(X=X, return_cov=True)
        else:
            y_mean, y_cov = self.predict(X, return_cov=True)

        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = \
                [rng.multivariate_normal(y_mean[:, i], y_cov,
                                         n_samples).T[:, np.newaxis]
                 for i in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, X=None, y=None):
        raise NotImplementedError

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        R"""A method to find the best kernel hyperparameters. Taken from scikit-learn.
        """
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict,
                              ConvergenceWarning)
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min


@docstrings.dedent
class ConjugateGaussianProcess(BaseConjugateProcess):
    R"""A conjugacy-based Gaussian Process class.

    Parameters
    ----------
    %(BaseConjugateProcess.parameters)s
    """

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, X=None, y=None):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If `None`, and fit() has been called, the precomputed
            log_marginal_likelihood of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        X : array, shape = (n_samples, n_features), optional
            The input data to use for the kernel. Defaults to `X` passed in `fit`.
        y : array, shape = (n_samples, [n_curves]), optional
            The observed data to use. Defaults to `y` passed in `fit`.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None and self._fit:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None"
                )
            return self.log_marginal_likelihood_value_

        if not hasattr(self, 'kernel_') or self.kernel_ is None:
            if self.kernel is None:
                kernel = self._default_kernel
            else:
                kernel = self.kernel
        else:
            kernel = self.kernel_
        kernel = kernel.clone_with_theta(theta)
        X = self.X_train_ if X is None else X
        y = self.y_train_ if y is None else y

        if eval_gradient:
            R, R_gradient = kernel(X, eval_gradient=True)
        else:
            R = kernel(X)
            R_gradient = None

        R[np.diag_indices_from(R)] += self.nugget

        decomp = self.decomposition

        if decomp == 'cholesky':
            try:
                sqrt_R = cholesky(R)  # Line 2
            except np.linalg.LinAlgError:
                return (-np.inf, np.zeros_like(theta)) \
                    if eval_gradient else -np.inf
        elif decomp == 'eig':
            sqrt_R = eigh(R)  # eig, Q
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')


        # Support multi-dimensional output of self.y_train_
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # ---------------------------------
        # Conjugacy-specific code.
        center0, disp0, df0, scale0 = self.center0, self.disp0, self.df0, self.scale0
        df = self.compute_df(y=y, df0=df0, eval_gradient=False)
        basis = self.basis(X)
        if eval_gradient:
            center, grad_center = self.compute_center(
                y, sqrt_R, basis, center0=center0, disp0=disp0,
                eval_gradient=eval_gradient, dR=R_gradient, decomposition=decomp
            )
            scale2, dscale2 = self.compute_scale_sq(
                y=y, sqrt_R=sqrt_R, basis=basis, center0=center0, disp0=disp0,
                df0=df0, scale0=scale0, eval_gradient=eval_gradient, dR=R_gradient,
                decomposition=decomp
            )
            grad_var = self.compute_cov_factor(scale_sq=dscale2, df=df)
            grad_mean = basis @ grad_center
        else:
            center = self.compute_center(y, sqrt_R, basis, center0=center0, disp0=disp0, decomposition=decomp)
            scale2 = self.compute_scale_sq(
                y=y, sqrt_R=sqrt_R, basis=basis, center0=center0, disp0=disp0,
                df0=df0, scale0=scale0, decomposition=decomp
            )
            grad_center, grad_var, grad_mean = None, None, None
        mean = basis @ center
        var = self.compute_cov_factor(scale_sq=scale2, df=df)

        # Convert from correlation matrix to covariance and subtract mean
        # to make all calculations below identical to scikit learn implementation
        # L = np.sqrt(var) * corr_L
        if decomp == 'cholesky':
            L = np.sqrt(var) * sqrt_R
            logdet_K = 2 * np.log(np.diag(L)).sum()
        elif decomp == 'eig':
            eig, Q = sqrt_R
            L = var * eig, Q  # Technically not lower triangular, but use L anyways
            logdet_K = np.log(var * eig).sum()
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')

        K, K_gradient = var * R, None
        if eval_gradient:
            K_gradient = var * R_gradient + grad_var * R[:, :, None]
        y_train = y - mean[:, None]
        N = K.shape[0]
        # ---------------------------------
        # Resume likelihood calculation

        # alpha = cho_solve((L, True), y_train)  # Line 3
        alpha = self.solve_sqrt(L, y_train, decomposition=decomp)

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        # log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= 0.5 * logdet_K
        log_likelihood_dims -= N / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GP for ML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            # tmp -= cho_solve((L, True), np.eye(N))[:, :, np.newaxis]
            tmp -= self.solve_sqrt(L, np.eye(N), decomposition=decomp)[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)

            # Beyond scikit-learn: Add gradient wrt mean
            log_likelihood_gradient_dims -= grad_mean.T @ alpha

            # Sum over output dimension
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
            return log_likelihood, log_likelihood_gradient
        return log_likelihood

    def likelihood(self, log=True, X=None, y=None, theta=None):
        # Multiple corr can be passed to quickly get likelihoods for many correlation parameters
        if X is None:
            X = self.X_train_
        if y is None:
            y = self.y_train_

        # corr = self.kernel(X, **self.kernel_kws)
        kernel = self.kernel_.clone_with_theta(theta)
        corr = kernel(X)
        corr = corr + self.nugget * np.eye(corr.shape[-1])
        corr_chol = cholesky(corr)
        
        # Setup best guesses for mean and cov
        center0, disp0, df0, scale0 = self.center0, self.disp0, self.df0, self.scale0
        df = self.compute_df(y=y, df0=df0)
        basis = self.basis(X)
        mean = basis @ self.compute_center(y, corr_chol, basis, center0=center0, disp0=disp0)
        # sd = self.compute_std(y=y, chol=corr_chol, basis=basis, beta0=beta0, disp0=disp0, df0=df0, scale0=scale0)
        scale2 = self.compute_scale_sq(
            y=y, chol=corr_chol, basis=basis, center0=center0, disp0=disp0,
            df0=df0, scale0=scale0)
        var = self.compute_cov_factor(scale_sq=scale2, df=df)
        cov = var * corr
        dist = st.multivariate_normal(mean=mean, cov=cov)
        log_like = np.sum(dist.logpdf(y))
        if log:
            return log_like
        return np.exp(log_like)


@docstrings.dedent
class ConjugateStudentProcess(BaseConjugateProcess):
    R"""A conjugacy-based Student-t Process class.

    Parameters
    ----------
    %(BaseConjugateProcess.parameters)s
    """

    def cov(self, X, Xp=None):

        if not self._fit:  # Unfitted; predict based on GP prior
            df = self.df0
            scale = self.scale0
            disp = self.disp0
            if self.kernel is None:
                kernel = self._default_kernel
            else:
                kernel = self.kernel
        else:
            df = self.df_
            scale = self.scale_
            disp = self.disp_
            kernel = self.kernel_

        if df <= 2:
            raise ValueError('df must be greater than 2 for the covariance to exist')

        # Call kernel before potentially reassigning Xp, else, e.g., WhiteKernel will not work properly
        corr = kernel(X, Xp)

        if Xp is None:
            Xp = X

        var = self.compute_cov_factor(scale_sq=scale**2, df=df)
        return var * (corr + self.basis(X) @ disp @ self.basis(Xp).T)

    @docstrings.dedent
    def predict(self, X, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=False):
        R"""

        Parameters
        ----------
        %(BaseConjugateProcess_predict.parameters)s
        """

        pred = super(ConjugateStudentProcess, self).predict(
            X=X, return_std=return_std, return_cov=return_cov, Xc=Xc, y=y, pred_noise=pred_noise)

        decomp = self.decomposition
        if not self._fit:  # Unfitted; predict based on GP prior
            disp = self.disp0
            var = self.compute_cov_factor(scale_sq=self.scale0 ** 2, df=self.df0)
            basis = self.basis(X)
        else:
            disp = self.disp_
            var = self.cov_factor_
            basis_new = self.basis(X)

            if Xc is None:
                basis_old = self.basis_train_
                if decomp == 'cholesky':
                    sqrt_R = self.corr_sqrt_
                elif decomp == 'eig':
                    sqrt_R = self._eigh_tuple_
                else:
                    raise ValueError('decomposition must be "cholesky" or "eig"')
                R_no = self.kernel_(X, self.X_train_)
            else:
                basis_old = self.basis(Xc)
                R_no = self.kernel_(X, Xc)
                kk = self.kernel_(Xc) + self.nugget * np.eye(len(Xc))
                # corr_chol = cholesky(kk)
                if decomp == 'cholesky':
                    sqrt_R = cholesky(kk)
                elif decomp == 'eig':
                    sqrt_R = eigh(kk)  # eig, Q
                else:
                    raise ValueError('decomposition must be "cholesky" or "eig"')
            # The conditional basis
            # basis = basis_new - R_no @ cho_solve((corr_chol, True), basis_old)
            basis = basis_new - R_no @ self.solve_sqrt(sqrt_R, basis_old, decomposition=decomp)

        mean_cov = var * (basis @ disp @ basis.T)  # From integrating out the mean
        if return_std:
            mean, std = pred
            std += np.sqrt(np.diag(mean_cov))
            return mean, std
        if return_cov:
            mean, cov = pred
            cov += mean_cov
            return mean, cov
        return pred

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, X=None, y=None):
        if y is None:
            y = self.y_train_
        if X is None:
            X = self.X_train_

        ny = self.num_y(y)
        if not hasattr(self, 'kernel_') or self.kernel_ is None:
            if self.kernel is None:
                kernel = self._default_kernel
            else:
                kernel = self.kernel
        else:
            kernel = self.kernel_
        kernel = kernel.clone_with_theta(theta)
        if eval_gradient:
            R, dR = kernel(X, eval_gradient)
        else:
            R, dR = kernel(X), None

        R[np.diag_indices_from(R)] += self.nugget
        N = R.shape[0]

        decomp = self.decomposition

        if decomp == 'cholesky':
            try:
                sqrt_R = cholesky(R)  # Line 2
            except np.linalg.LinAlgError:
                return (-np.inf, np.zeros_like(theta)) \
                    if eval_gradient else -np.inf
        elif decomp == 'eig':
            sqrt_R = eigh(R)  # eig, Q
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')

        center0, disp0, df0, scale0 = self.center0, self.disp0, self.df0, self.scale0
        df = self.compute_df(y=y, df0=df0)
        basis = self.basis(X)
        if eval_gradient:
            disp, grad_disp = self.compute_disp(
                y=y, sqrt_R=sqrt_R, basis=basis, disp0=disp0,
                eval_gradient=eval_gradient, dR=dR, decomposition=decomp
            )
            scale_sq, grad_scale_sq = self.compute_scale_sq(
                y=y, sqrt_R=sqrt_R, basis=basis, center0=center0, disp0=disp0,
                df0=df0, scale0=scale0, eval_gradient=eval_gradient, dR=dR, decomposition=decomp
            )
        else:
            disp = self.compute_disp(y=y, sqrt_R=sqrt_R, basis=basis, disp0=disp0, decomposition=decomp)
            scale_sq = self.compute_scale_sq(
                y=y, sqrt_R=sqrt_R, basis=basis, center0=center0, disp0=disp0, df0=df0,
                scale0=scale0, decomposition=decomp
            )
            grad_disp, grad_scale_sq = None, None
        scale = np.sqrt(scale_sq)

        def log_norm(df_, scale_, disp_):
            """Normalization constant of the normal scaled inverse chi squared distribution"""
            norm = loggamma(df_ / 2.) - df_ / 2. * np.log(df_ * scale_ / 2.)
            log_det = np.linalg.slogdet(2 * np.pi * disp_)[1]
            if log_det != -np.inf:
                norm += 0.5 * log_det
            return norm

        if decomp == 'cholesky':
            logdet_R = 2 * np.log(np.diag(sqrt_R)).sum()
        elif decomp == 'eig':
            eig, Q = sqrt_R
            logdet_R = np.log(eig).sum()
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')

        log_like = log_norm(df, scale, disp) - log_norm(df0, scale0, disp0) \
            - ny / 2. * (N * np.log(2*np.pi) + logdet_R)

        if eval_gradient:
            # cho_solve only cares about first dimension of dR. Gradient parameters are in the last dimension.
            # log_like_gradient = - (ny / 2.) * np.trace(cho_solve((corr_L, True), dR), axis1=0, axis2=1)
            log_like_gradient = - (ny / 2.) * np.trace(
                self.solve_sqrt(sqrt_R, dR, decomposition=decomp), axis1=0, axis2=1
            )
            log_like_gradient -= (df / 2.) * grad_scale_sq / scale_sq

            if not np.all(disp == 0):
                log_like_gradient += 0.5 * np.einsum('ij,ijp->p', inv(disp), grad_disp)

            return log_like, log_like_gradient

        return log_like


def _default_ref(X, ref):
    return ref * np.ones(X.shape[0])


def _default_ratio(X, ratio):
    return ratio * np.ones(X.shape[0])


@docstrings.dedent
class TruncationProcess:
    R"""

    Parameters
    ----------
    kernel : sklearn.Kernel

    ratio : scalar or callable
    ref : scalar or callable
    excluded : 1d array, optional
        The set of orders to ignore when constructing process for y_order and dy_order, i.e., the geometric sum
        will not include these values
    ratio_kws : dict, optional
    kernel_kws : dict, optional
    nugget : float, optional
    verbose : bool, optional

    Other Parameters
    ----------------
    %(BaseConjugateProcess.parameters)s
    """

    def __init__(self, kernel=None, ratio=0.5, ref=1, excluded=None, ratio_kws=None, **kwargs):

        if not callable(ref):
            self.ref = lambda X, ref=ref: ref * np.ones(X.shape[0])
        else:
            self.ref = ref

        if not callable(ratio):
            self.ratio = lambda X, ratio=ratio: ratio * np.ones(X.shape[0])
        else:
            self.ratio = ratio

        # self.coeffs_process_class = BaseConjugateProcess
        # self.coeffs_process = self.coeffs_process_class(kernel=kernel, **kwargs)
        self.coeffs_process = BaseConjugateProcess(kernel=kernel, **kwargs)
        self.kernel = kernel
        self._log_like = None

        self.excluded = excluded
        self.ratio_kws = {} if ratio_kws is None else ratio_kws

        self._fit = False
        self.X_train_ = None
        self.y_train_ = None
        self.orders_ = None
        self.dX_ = None
        self.dy_ = None
        self.coeffs_ = None
        # self.coeffs_process_ = None

    def mean(self, X, start=0, end=np.inf):
        coeff_mean = self.coeffs_process.mean(X=X)
        ratio_sum = geometric_sum(x=self.ratio(X, **self.ratio_kws), start=start, end=end, excluded=self.excluded)
        return self.ref(X) * ratio_sum * coeff_mean

    def cov(self, X, Xp=None, start=0, end=np.inf):
        coeff_cov = self.coeffs_process.cov(X=X, Xp=Xp)
        Xp = X if Xp is None else Xp  # Must reassign *after* calling cov
        ratio_mat = self.ratio(X, **self.ratio_kws)[:, None] * self.ratio(Xp, **self.ratio_kws)
        ratio_sum = geometric_sum(x=ratio_mat, start=start, end=end, excluded=self.excluded)
        ref_mat = self.ref(X)[:, None] * self.ref(Xp)
        return ref_mat * ratio_sum * coeff_cov

    def basis(self, X, start=0, end=np.inf):
        cn_basis = self.coeffs_process.basis(X=X)
        ratio = self.ratio(X, **self.ratio_kws)[:, None]
        ratio_sum = geometric_sum(x=ratio, start=start, end=end, excluded=self.excluded)
        return self.ref(X)[:, None] * ratio_sum * cn_basis

    def underlying_properties(self, X, order, return_std=False, return_cov=False):
        y_mean = self.mean(X, start=order+1)
        if return_cov:
            y_cov = self.cov(X, start=order+1)
            return y_mean, y_cov
        elif return_std:
            y_std = np.sqrt(np.diag(self.cov(X, start=order+1)))
            return y_mean, y_std
        else:
            return y_mean

    def fit(self, X, y, orders, dX=None, dy=None):
        self.X_train_ = X
        self.y_train_ = y
        self.orders_ = orders
        orders_mask = ~ np.isin(orders, self.excluded)

        self.dX_ = dX
        self.dy_ = dy

        # Extract the coefficients based on best ratio value and setup/fit the iid coefficient process
        ratio = self.ratio(X, **self.ratio_kws)
        ref = self.ref(X)
        if np.atleast_1d(ratio).ndim > 1:
            raise ValueError('ratio must return a 1d array or a scalar')
        if np.atleast_1d(ref).ndim > 1:
            raise ValueError('ref must return a 1d array or a scalar')
        self.coeffs_ = coefficients(y=y, ratio=ratio, ref=ref, orders=orders)[:, orders_mask]
        # self.coeffs_process_ = self.coeffs_process_class(kernel=self.kernel, **self.coeffs_process_kwargs)
        self.coeffs_process.fit(X=X, y=self.coeffs_)
        self._fit = True
        return self

    def predict(self, X, order, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=False, kind='both'):
        """Returns the predictive GP at the points X

        Parameters
        ----------
        X : (M, d) array
            Locations at which to predict the new y values
        order : int
            The order of the GP to predict
        return_std : bool
            Whether the marginal standard deviation of the predictive process is to be returned
        return_cov : bool
            Whether the covariance matrix of the predictive process is to be returned
        Xc : (N, d) array
            Locations at which to condition. Defaults to `X` used in fit. This *does not*
            affect the `X` used to update hyperparameters.
        y : (N, n) array
            Points upon which to condition. Defaults to the `y` used in `fit`. This *does not*
            affect the `y` used to update hyperparameters.
        pred_noise : bool
            Adds `noise_sd` to the diagonal of the covariance matrix if `return_cov == True`.
        kind : str

        Returns
        -------
        mean, (mean, std), or (mean, cov), depending on `return_std` and `return_cov`
        """

        if not self._fit:
            return self.underlying_properties(X, order, return_cov=return_cov, return_std=return_std)

        if Xc is None:
            Xc = self.X_train_
        if y is None:
            if order not in self.orders_:
                raise ValueError('order must be in orders passed to `fit`')
            if self.y_train_.ndim == 1:
                y = self.y_train_
            else:
                y = np.squeeze(self.y_train_[:, self.orders_ == order])

        if kind not in ['both', 'interp', 'trunc']:
            raise ValueError('kind must be one of "both", "interp" or "trunc"')

        m_pred, K_pred = 0, 0
        if kind == 'both' or kind == 'interp':
            # ----------------------------------------------------
            # Get mean & cov for (interpolating) prediction y_order
            #
            # Use X and y from fit for hyperparameters
            m_old = self.mean(X=Xc, start=0, end=order)
            m_new = self.mean(X=X, start=0, end=order)

            # Use X and y from arguments for conditioning/predictions
            K_oo = self.cov(start=0, end=order, X=Xc, Xp=Xc)
            K_on = self.cov(start=0, end=order, X=Xc, Xp=X)
            K_no = K_on.T
            K_nn = self.cov(start=0, end=order, X=X, Xp=X)

            # Use given y for prediction
            alpha = solve(K_oo, y - m_old)
            m_pred += m_new + K_no @ alpha
            if return_std or return_cov:
                K_pred += K_nn - K_no @ solve(K_oo, K_on)
            #
            # ----------------------------------------------------

        if kind == 'both' or kind == 'trunc':
            # ----------------------------------------------------
            # Get the mean & cov for truncation error
            #
            m_new_trunc = self.mean(X=X, start=order + 1, end=np.inf)
            K_nn_trunc = self.cov(X=X, Xp=X, start=order + 1, end=np.inf)

            X_trunc = self.dX_
            if X_trunc is not None:  # truncation error is constrained
                m_old_trunc = self.mean(X=X_trunc, start=order+1, end=np.inf)
                K_oo_trunc = self.cov(X=X_trunc, Xp=X_trunc, start=order+1, end=np.inf)
                K_on_trunc = self.cov(X=X_trunc, Xp=X, start=order+1, end=np.inf)
                K_no_trunc = K_on_trunc.T

                alpha_trunc = solve(K_oo_trunc, (self.dy_ - m_old_trunc))
                m_pred += m_new_trunc + K_no_trunc @ alpha_trunc
                if return_std or return_cov:
                    K_pred += K_nn_trunc - K_no_trunc @ solve(K_oo_trunc, K_on_trunc)
            else:  # truncation is not constrained
                m_pred += m_new_trunc
                if return_std or return_cov:
                    K_pred += K_nn_trunc

        if return_cov:
            return m_pred, K_pred
        if return_std:
            return m_pred, np.sqrt(np.diag(K_pred))
        return m_pred

    def log_marginal_likelihood(self, theta, eval_gradient=False, X=None, y=None, orders=None, **ratio_kws):
        if X is None:
            X = self.X_train_
        if y is None:
            y = self.y_train_
        if orders is None:
            orders = self.orders_
        ref = self.ref(X)
        ratio = self.ratio(X, **ratio_kws)

        orders_mask = ~ np.isin(orders, self.excluded)
        coeffs = coefficients(y=y, ratio=ratio, ref=ref, orders=orders)[:, orders_mask]
        result = self.coeffs_process.log_marginal_likelihood(theta, eval_gradient=eval_gradient, X=X, y=coeffs)
        if eval_gradient:
            coeff_log_like, coeff_log_like_gradient = result
        else:
            coeff_log_like = result

        orders_in = orders[orders_mask]
        n = len(orders_in)
        det_factor = np.sum(n * np.log(np.abs(ref)) + np.sum(orders_in) * np.log(np.abs(ratio)))
        y_log_like = coeff_log_like - det_factor
        return y_log_like


class TruncationGP(TruncationProcess):
    R"""A Gaussian Process Truncation class"""

    def __init__(self, kernel=None, ratio=0.5, ref=1, excluded=None, ratio_kws=None, **kwargs):
        super().__init__(
            kernel=kernel, ref=ref, ratio=ratio, excluded=excluded, ratio_kws=ratio_kws, **kwargs)
        self.coeffs_process = ConjugateGaussianProcess(kernel=kernel, **kwargs)


class TruncationTP(TruncationProcess):
    R"""A Student-t Process Truncation class"""

    def __init__(self, kernel=None, ratio=0.5, ref=1, excluded=None, ratio_kws=None, **kwargs):
        super().__init__(
            kernel=kernel, ratio=ratio, ref=ref, excluded=excluded, ratio_kws=ratio_kws, **kwargs)
        self.coeffs_process = ConjugateStudentProcess(kernel=kernel, **kwargs)

    def predict(self, X, order, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=False, kind='both'):
        pred = super(TruncationTP, self).predict(
            X=X, order=order, return_std=return_std, return_cov=return_cov,
            Xc=Xc, y=y, pred_noise=pred_noise
        )

        if not return_std and not return_cov:
            return pred

        if Xc is None:
            Xc = self.X_train_

        var, disp = self.coeffs_process.cov_factor_, self.coeffs_process.disp_
        basis_lower, basis_trunc = np.zeros((X.shape[0], disp.shape[0])), np.zeros((X.shape[0], disp.shape[0]))

        if kind == 'both' or kind == 'interp':
            # Use Xc from argument to define old points
            K_oo = self.cov(X=Xc, Xp=Xc, start=0, end=order)
            K_no = self.cov(X=X, Xp=Xc, start=0, end=order)

            basis_lower_old = self.basis(X=Xc, start=0, end=order)
            basis_lower_new = self.basis(X=X, start=0, end=order)
            basis_lower = basis_lower_new - K_no @ solve(K_oo, basis_lower_old)

        if kind == 'both' or kind == 'trunc':
            X_trunc = self.dX_
            if X_trunc is not None:  # truncation error is constrained
                K_oo_trunc = self.cov(X=X_trunc, Xp=X_trunc, start=order+1, end=np.inf)
                K_no_trunc = self.cov(X=X, Xp=X_trunc, start=order+1, end=np.inf)

                basis_trunc_old = self.basis(X=X_trunc, start=order+1, end=np.inf)
                basis_trunc_new = self.basis(X=X, start=order+1, end=np.inf)
                basis_trunc = basis_trunc_new - K_no_trunc @ solve(K_oo_trunc, basis_trunc_old)
            else:  # not constrained
                basis_trunc = self.basis(start=order + 1, end=np.inf, X=X)

        mean_cov = var * (basis_lower + basis_trunc) @ disp @ (basis_lower + basis_trunc).T

        if return_std:
            mean, std = pred
            return mean, std + np.sqrt(np.diag(mean_cov))
        if return_cov:
            mean, cov = pred
            return mean, cov + mean_cov


class TruncationPointwise:
    R"""A conjugacy-based implementation of the pointwise convergence model from Furnstahl et al. (2015)

    Implements the following model

    .. math::

        y_k = y_{\mathrm{ref}} \sum_{n=0}^k c_n Q^n

    where the :math:`c_n` are iid Gaussian random variables and :math:`\bar c^2` has a scaled inverse chi squared
    conjugate prior

    .. math::

        c_n \,|\, \bar c^2 & \sim N(0, \bar c^2) \\
        \bar c^2 & \sim \chi^{-2}(\nu_0, \tau_0^2)

    Conditioning on the partial sums :math:`y_0`, :math:`\dots,` :math:`y_k`, allow
    one to estimate :math:`\bar c`, and thus the full summation :math:`y_\infty`.

    Parameters
    ----------
    df : float >= 0
        The degrees of freedom hyperparameter :math:`\nu_0` for the scaled inverse chi squared prior on :math:`\bar c`
    scale : float > 0
        The scale hyperparameter :math:`\tau_0` for the scaled inverse chi squared prior on :math:`\bar c`
    excluded : int or array, optional
        The orders to be excluded from both the hyperparameter updating and from the truncation error distribution.
        Defaults to `None`.
    """

    def __init__(self, df=1, scale=1, excluded=None):
        self.df0 = df
        self.scale0 = scale
        self.excluded = excluded

        self._fit = False
        self.y_ = None
        self.ratio_ = None
        self.ref_ = None
        self.orders_ = None
        self.orders_mask_ = None
        self._orders_masked = None
        self.coeffs_ = None
        self.coeffs_dist_ = None
        self.df_ = None
        self.scale_ = None
        self.y_masked_ = None
        self.dist_ = None

    @classmethod
    def _compute_df(cls, c, df0):
        return df0 + c.shape[-1]

    @classmethod
    def _compute_scale(cls, c, df0, scale0):
        c_sq = (c ** 2).sum(-1)
        df = cls._compute_df(c, df0)
        return np.sqrt((df0 * scale0**2 + c_sq) / df)

    @staticmethod
    def _num_orders(y):
        if y.ndim == 1:
            return 1
        elif y.ndim == 2:
            return y.shape[-1]

    def _compute_order_indices(self, orders):
        if orders is None:
            return slice(None)
        orders = np.atleast_1d(orders)
        return np.squeeze([np.nonzero(self._orders_masked == order) for order in orders])

    def fit(self, y, ratio, ref=1, orders=None):
        R"""

        Parameters
        ----------
        y
        ratio
        ref
        orders

        Returns
        -------

        """
        if y.ndim == 1:
            y = y[:, None]

        ratio, ref = np.atleast_1d(ratio, ref)

        self.y_ = y
        self.ratio_ = ratio
        self.ref_ = ref

        if orders is None:
            orders = np.arange(y.shape[-1])

        if y.shape[-1] != orders.size:
            raise ValueError('The last dimension of `y` must have the same size as `orders`')

        self.orders_ = orders
        self.orders_mask_ = orders_mask = ~ np.isin(orders, self.excluded)
        self.coeffs_ = coefficients(y=y, ratio=ratio, ref=ref, orders=orders)[:, orders_mask]
        self.df_ = self._compute_df(c=self.coeffs_, df0=self.df0)
        self.scale_ = self._compute_scale(c=self.coeffs_, df0=self.df0, scale0=self.scale0)

        self.y_masked_ = y[:, orders_mask]
        self._orders_masked = orders_masked = orders[orders_mask]
        ratio_sums = np.array([geometric_sum(ratio**2, k+1, np.inf, excluded=self.excluded)
                               for k in orders_masked]).T
        trunc_scale = ref[:, None] * np.sqrt(ratio_sums) * self.scale_[:, None]
        self.coeffs_dist_ = st.t(loc=0, scale=self.scale_, df=self.df_)
        self.dist_ = st.t(loc=self.y_masked_, scale=trunc_scale, df=self.df_)
        self._fit = True
        return self

    def interval(self, alpha, orders=None):
        R"""A convenience method to call `interval` on the truncation error distribution object.

        Parameters
        ----------
        alpha
        orders

        Returns
        -------

        """
        alpha = np.array(alpha)
        if alpha.ndim == 1:
            alpha = alpha[:, None, None]
        interval = np.array(self.dist_.interval(alpha))
        idx = self._compute_order_indices(orders)
        return interval[..., idx]

    def pdf(self, y, orders=None):
        R"""A convenience method to call `pdf` on the truncation error distribution object.

        Parameters
        ----------
        y
        orders

        Returns
        -------

        """
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y[:, None, None]
        idx = self._compute_order_indices(orders)
        return self.dist_.pdf(y)[..., idx]

    def logpdf(self, y, orders=None):
        R"""A convenience method to call `logpdf` on the truncation error distribution object.

        Parameters
        ----------
        y
        orders

        Returns
        -------

        """
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y[:, None, None]
        idx = self._compute_order_indices(orders)
        return self.dist_.logpdf(y)[..., idx]

    def std(self):
        R"""A convenience method to call `std` on the truncation error distribution object.

        Returns
        -------

        """
        return self.dist_.std()

    def log_likelihood(self, ratio=None, ref=None):
        R"""Computes the log likelihood for the ratio and ref parameters given the data passed to `fit`.

        That is

        .. math::
            pr(\vec{y}_k \, | \, Q, y_{ref}) & = \frac{pr(\vec{c}_k)}{\prod_n y_{ref} Q^n} \\
            pr(\vec{c}_k) & = \frac{\Gamma(\nu/2)}{\Gamma(\nu_0/2)}
                              \sqrt{\frac{1}{(2\pi)^n} \frac{(\nu_0 \tau_0^2 / 2)^{\nu_0}}{(\nu \tau^2 / 2)^{\nu}}}

        Parameters
        ----------
        ratio : scalar or array, shape = (n_points,)
            The ratio, or EFT expansion parameter, in the geometric sum, used to extract the coefficients.
        ref : scalar or array, shape = (n_points,)
            The multiplicative reference scale used to extract the coefficients.

        Returns
        -------
        float
            The log likelihood
        """
        if not self._fit:
            raise ValueError('Must call fit before calling log_likelihood')

        if ratio is None:
            ratio = self.ratio_
        if ref is None:
            ref = self.ref_

        y, orders, mask = self.y_, self.orders_, self.orders_mask_
        coeffs = coefficients(y=y, ratio=ratio, ref=ref, orders=orders)[:, mask]
        df0, scale0 = self.df0, self.scale0
        df = self._compute_df(c=coeffs, df0=df0)
        scale = self._compute_scale(c=coeffs, df0=df0, scale0=scale0)

        n = self._num_orders(coeffs)
        log_like = loggamma(df / 2.) - 0.5 * n * np.log(2 * np.pi)
        if df0 > 0:  # Ignore this infinite constant for scale invariant prior, df0 == 0
            log_like += 0.5 * np.sum(df0 * np.log(df0 * scale0 ** 2 / 2.)) - loggamma(df0 / 2.)
        log_like -= 0.5 * np.sum(df * np.log(df * scale**2 / 2.))
        log_like -= np.sum(np.log(np.abs(ref)) + np.sum(orders[mask]) * np.log(ratio))  # From change of variables
        return log_like

    def credible_diagnostic(self, data, dobs, band_intervals=None, band_dobs=None, beta=True):
        dist = self.dist_
        dobs = np.atleast_1d(dobs)
        if data.ndim == 1:
            data = data[:, None]
        lower, upper = dist.interval(dobs[:, None, None])

        def diagnostic(data_, lower_, upper_):
            indicator = (lower_ < data_) & (data_ < upper_)  # 1 if in, 0 if out
            return np.average(indicator, axis=1)   # The diagnostic

        # D_CI = np.apply_along_axis(
        #         diagnostic, axis=0, arr=data, lower_=lower,
        #         upper_=upper)
        D_CI = diagnostic(data, lower, upper)

        if band_intervals is not None:
            if band_dobs is None:
                band_dobs = dobs
            band_dobs = np.atleast_1d(band_dobs)

            N = self.y_.shape[0]
            if beta:
                band_intervals = np.atleast_1d(band_intervals)
                # Band shape: (len(dobs), 2, len(X))
                bands = np.zeros((len(band_intervals), 2, len(band_dobs)))
                for i, p in enumerate(band_intervals):
                    bands[i] = np.array(
                        [hpd(sp.stats.beta, p, N*s+1, N-N*s+1)
                         for s in band_dobs]).T
                # bands = np.transpose(bands, [0, 1, 2])
            else:
                band_dist = st.binom(n=N, p=band_dobs)
                band_intervals = np.atleast_2d(band_intervals)
                bands = np.asarray(band_dist.interval(band_intervals.T)) / N
                bands = np.transpose(bands, [1, 0, 2])
            return D_CI, bands
        return D_CI
