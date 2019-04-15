from __future__ import division
from .helpers import coefficients, predictions, gaussian, stabilize, HPD, default_attributes, mahalanobis, rbf
import numpy as np
from numpy.linalg import solve, cholesky
import scipy as sp
from scipy.linalg import cho_solve, solve_triangular, inv
from scipy.special import loggamma
import scipy.stats as st
from scipy.optimize import fmin_l_bfgs_b
from statsmodels.sandbox.distributions.mv_normal import MVT
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import ConvergenceWarning

import warnings
from operator import itemgetter


__all__ = [
    'SGP', 'PowerProcess', 'PowerSeries', 'ConjugateProcess',
    'ConjugateGaussianProcess', 'ConjugateStudentProcess', 'TruncationGP', 'TruncationTP']


class ConjugateProcess:
    
    def __init__(self, kernel=None, center=0, disp=1, df=1, scale=1, sd=None, basis=None, kernel_kws=None,
                 nugget=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, copy_X_train=True, random_state=None):
        R"""A conjugate Gaussian Process model.


        Parameters
        ----------
        kernel : callable
            The kernel for the correlation matrix. The covariance matrix is the kernel multiplied by the squared scale.
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
        verbose : bool, optional (default: True)
            Whether to print info about the chosen kernel parameters when fitting.
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

        self.X_train_ = None
        self.y_train_ = None
        self.corr_L_ = None
        self.corr_ = None
        self.noise_sd = None

        self.nugget = nugget
        # self.verbose = verbose
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.optimizer = optimizer

        # Setup kernel keywords for fitting via maximum likelihood
        # self.all_kernel_kws = None
        # if kernel_kws is None:
        #     kernel_kws = {}
        # if isinstance(kernel_kws, (list, np.ndarray)):
        #     self.all_kernel_kws = np.array(kernel_kws)
        # elif isinstance(kernel_kws, dict):
        #     self.kernel_kws = kernel_kws
        # else:
        #     raise ValueError('kernel_kws must be a list or dict')

        # if kernel_kws is None:
        #     kernel_kws = {}
        # self.kernel_kws = kernel_kws

        # self.optimize = False
        # if isinstance(kernel_kws, (list, np.ndarray)):
        #     self.optimize = True

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
    def compute_center(cls, y, chol, basis, center0, disp0, eval_gradient=False, dR=None):
        R"""Computes the regression coefficients' center hyperparameter :math:`\eta` updated based on data

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
            The data to condition upon
        chol : array, shape = (n_samples, n_samples)
            The cholesky decomposition of the correlation matrix
        basis : array, shape = (n_samples, n_param)
            The basis matrix that multiplies the regression coefficients beta to create the GP mean.
        center0 : scalar or array, shape = (n_param)
            The prior regression coefficients for the mean
        disp0 : scalar or array, shape = (n_param, n_param)
            The prior dispersion for the regression coefficients
        eval_gradient : bool
        dR

        Returns
        -------
        center : scalar or array, shape = (n_param)
            The posterior regression coefficients for the mean
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

        invR_y_avg = cho_solve((chol, True), y_avg)
        disp = cls.compute_disp(y=y, chol=chol, basis=basis, disp0=disp0)
        factor = solve(disp0, center0) + ny * basis.T @ invR_y_avg
        center = disp @ factor

        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            invR_basis = cho_solve((chol, True), basis)
            invR_diff = cho_solve((chol, True), basis @ center - y_avg)
            d_center = ny * disp @ np.einsum('ji,jkp,k->ip', invR_basis, dR, invR_diff)
            return center, d_center
        return center

    @classmethod
    def compute_disp(cls, y, chol, basis, disp0, eval_gradient=False, dR=None):
        R"""The dispersion hyperparameter :math:`V` updated based on data.

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
            The data to condition upon
        chol : (n_samples, n_samples)-shaped array
            The lower Cholesky decomposition of the correlation matrix
        basis : (n_samples, n_param)-shaped array
            The basis for the `p` regression coefficients `beta`
        disp0 : (n_param, n_param)-shaped array
            The prior dispersion
        eval_gradient : bool
        dR

        Returns
        -------
        disp : (p, p)-shaped array
            The updated dispersion hyperparameter
        d_disp : array, shape = (
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
        quad = mahalanobis(basis.T, 0, chol) ** 2
        disp = inv(inv(disp0) + ny * quad)
        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            invRBV = cho_solve((chol, True), basis) @ disp
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
        eval_gradient
        dR

        Returns
        -------
        df : scalar
            The updated degrees of freedom
        d_df : array, size = (n_kernel_params,), optional
            The gradient of the updated degrees of freedom with respect to the kernel parameters
        """
        df = df0 + y.size
        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is True')
            return df, np.zeros(dR.shape[-1])
        return df

    @classmethod
    def compute_scale_sq_v2(cls, y, chol, basis, center0, disp0, df0, scale0, eval_gradient=False, dR=None):
        R"""The squared scale hyperparameter :math:`\tau^2` updated based on data.

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
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
        eval_gradient : bool, optional
            Whether to return to the gradient with respect to kernel hyperparameters. Optional, defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params), optional
            The gradient of the correlation matrix

        Returns
        -------
        scale_sq : scalar
            The updated scale hyperparameter squared
        d_scale_sq : array, shape = (n_kernel_params,), optional
            The gradient of scale^2 with respect to the kernel hyperparameters. Only returned if eval_gradient is True.
        """
        if df0 == np.inf:
            if eval_gradient:
                return scale0, np.zeros(dR.shape[-1])
            return scale0

        avg_y = cls.avg_y(y)
        ny = cls.num_y(y)

        if np.all(disp0 == 0):
            invR_diff0 = cho_solve((chol, True), 2 * avg_y - basis @ center0)
            mean_terms = - ny * center0 @ basis.T @ invR_diff0
        else:
            center = cls.compute_center(y=y, chol=chol, basis=basis, center0=center0, disp0=disp0)
            disp = cls.compute_disp(y=y, chol=chol, basis=basis, disp0=disp0)
            mean_terms = center0 @ inv(disp0) @ center0 - center @ inv(disp) @ center
        if y.ndim == 1:
            y = y[None, :]
        invR_y = cho_solve((chol, True), y.T)
        quad = np.sum(y @ invR_y, axis=-1)
        df = cls.compute_df(y=y, df0=df0)
        scale_sq = (df0 * scale0**2 + mean_terms + quad) / df

        if eval_gradient:
            if dR is None:
                raise ValueError('dR must be given if eval_gradient is true')
            d_scale_sq = - np.einsum('ij,jkp,ki->p', invR_y.T, dR, invR_y)

            if np.all(disp0 == 0):
                center = cls.compute_center(y=y, chol=chol, basis=basis, center0=center0, disp0=disp0)
            invR_diff = cho_solve((chol, True), 2 * avg_y - basis @ center)
            invR_basis_center = cho_solve((chol, True), basis) @ center
            d_scale_sq += ny * np.einsum('i,ijp,j->p', invR_basis_center, dR, invR_diff)

            d_scale_sq /= df
            return scale_sq, d_scale_sq
        return scale_sq

    @classmethod
    def compute_scale_sq(cls, y, chol, basis, center0, disp0, df0, scale0, eval_gradient=False, dR=None):
        R"""The squared scale hyperparameter :math:`\tau^2` updated based on data.

        Parameters
        ----------
        y : array, shape = (n_curves, n_samples)
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
        eval_gradient : bool
            Whether to return to the gradient with respect to kernel hyperparameters. Optional, defaults to False.
        dR : array, shape = (n_samples, n_samples, n_kernel_params)
            The gradient of the correlation matrix

        Returns
        -------
        scale_sq : scalar
            The updated scale hyperparameter squared
        d_scale_sq : array, shape = (n_kernel_params,), optional
            The gradient of scale^2 with respect to the kernel hyperparameters. Only returned if eval_gradient is True.
        """
        if df0 == np.inf:
            if eval_gradient:
                return scale0, np.zeros(dR.shape[-1])
            return scale0

        if y.ndim == 1:
            y = y[None, :]
        avg_y = cls.avg_y(y)
        ny = cls.num_y(y)

        y_centered = y - avg_y
        invR_yc = cho_solve((chol, True), y_centered.T)
        quad = np.sum(y_centered @ invR_yc, axis=-1)

        avg_y_centered = avg_y - basis @ center0
        disp = cls.compute_disp(y=y, chol=chol, basis=basis, disp0=disp0, eval_gradient=False)
        mat = np.eye(chol.shape[0]) - ny * cho_solve((chol, True), basis) @ disp @ basis.T
        mat_invR_avg_yc = ny * mat @ cho_solve((chol, True), avg_y_centered)
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

    # @classmethod
    # def compute_std(cls, y, chol, basis, center0, disp0, df0, scale0):
    #     """The mean value for the marginal standard deviation given y.
    #
    #     It turns out that for both the GP and TP, `sd**2` is the conversion factor to go
    #     from the correlation matrix to the covariance matrix.
    #
    #     Note: if the correlation matrix does equal 1 when `X == Xp`, `sd` **will not**
    #     be the standard deviation at `X`. Instead, one must look at `cov` directly.
    #     """
    #     std = scale0
    #     if df0 != np.inf:
    #         scale_sq = cls.compute_scale_sq(
    #             y=y, chol=chol, basis=basis, center0=center0, disp0=disp0, df0=df0, scale0=scale0)
    #         df = cls.compute_df(y=y, df0=df0)
    #         std = np.sqrt(df * scale_sq / (df - 2))
    #     return std

    @staticmethod
    def scale_sq_to_marginal_variance(scale_sq, df):
        """Converts the squared scale hyperparameter :math:`\tau^2` to the marginal variance

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
        return self.compute_center(
            y=self.y_train_, chol=self.corr_L_, basis=self.basis_train_, center0=self.center0, disp0=self.disp0)

    def disp(self):
        """The dispersion hyperparameter updated by the call to `fit`.
        """
        return self.compute_disp(y=self.y_train_, chol=self.corr_L_, basis=self.basis_train_, disp0=self.disp0)

    def df(self):
        """The degrees of freedom hyperparameter for the standard deviation updated by the call to `fit`
        """
        return self.compute_df(y=self.y_train_, df0=self.df0)

    def scale(self):
        """The scale hyperparameter for the standard deviation updated by the call to `fit`
        """
        scale_sq = self.compute_scale_sq(y=self.y_train_, chol=self.corr_L_, basis=self.basis_train_,
                                         center0=self.center0, disp0=self.disp0, df0=self.df0, scale0=self.scale0)
        return np.sqrt(scale_sq)

    # def std(self):
    #     """The mean value for the marginal standard deviation given y.
    #
    #     It turns out that for both the GP and TP, `sd**2` is the conversion factor to go
    #     from the correlation matrix to the covariance matrix.
    #
    #     Note: if the correlation matrix does equal 1 when `X == Xp`, `sd` **will not**
    #     be the standard deviation at `X`. Instead, one must look at `cov` directly.
    #     """
    #     return self.compute_std(y=self.y_train_, chol=self.corr_L_, basis=self.basis_train_,
    #                             center0=self.center0, disp0=self.disp0, df0=self.df0, scale0=self.scale0)

    def mean(self, X=None):
        """The MAP value for the mean of the process at inputs X with hyperparameters updated by y.

        This does not interpolate the y values. For that functionality, use `predict`.
        """
        if X is None:
            return self.X_train_
        return self.basis(X) @ self.center()

    def cov(self, X=None, Xp=None):
        if X is None:
            X = self.X_train_
        if Xp is None:
            Xp = X
        corr = self.kernel_(X, Xp)
        return self.std()**2 * corr
    
    @staticmethod
    def num_y(y):
        ny = 1
        if y.ndim == 2:
            ny = y.shape[0]
        return ny

    @staticmethod
    def avg_y(y):
        if y.ndim == 1:
            return np.copy(y)
        elif y.ndim == 2:
            return np.average(y, axis=0)

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
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)
    
    def fit(self, X, y):
        R"""Fits the hyperparameters to the data (X, y) and updates all hyperparameters.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The input variables where the response is observed
        y : array, shape = (n_curves, n_samples)
            The response values

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        self.basis_train_ = self.basis(self.X_train_)

        # if self.optimize:
        #     # corr_vec = np.array([self.kernel(X, X, **kws) for kws in self.all_kernel_kws])
        #     # log_like = self.likelihood(log=True, y=y, corr=corr_vec, noise_sd=noise_sd)
        #     log_like = np.array([self.likelihood(log=True, X=X, y=y, **kws) for kws in self.kernel_kws])
        #     kernel_kws = self.kernel_kws[np.argmax(log_like)]
        #     if self.verbose:
        #         print('Setting kernel kwargs to {}'.format(kernel_kws))
        #     self.kernel_kws_ = kernel_kws
        # else:
        #     self.kernel_kws_ = self.kernel_kws
        self._calibrate_kernel()

        # self.corr_ = self.kernel(X=X, Xp=None, **self.kernel_kws_)
        self.corr_ = self.kernel_(X=X)
        self.corr_L_ = cholesky(self.corr_ + self.nugget * np.eye(len(X)))

        self.center_ = self.center()
        self.disp_ = self.disp()
        self.df_ = self.df()
        self.scale_ = self.scale()
        # self.std_ = self.std()
        return self

    def predict(self, X, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=True):
        """Returns the predictive GP at the points X

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Locations at which to predict the new y values
        return_std : bool
            Whether the marginal standard deviation of the predictive process is to be returned
        return_cov : bool
            Whether the covariance matrix of the predictive process is to be returned
        Xc : array, shape = (n_conditional_samples, n_features)
            Locations at which to condition. Defaults to `X` used in fit. This *does not*
            affect the `X` used to update hyperparameters.
        y : array, shape = (n_curves, n_conditional_samples)
            Points upon which to condition. Defaults to the `y` used in `fit`. This *does not*
            affect the `y` used to update hyperparameters.
        pred_noise : bool
            Adds `noise_sd` to the diagonal of the covariance matrix if `return_cov == True`.

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

        kwargs = self.kernel_kws_
        if Xc is None:
            Xc = self.X_train_
            corr_chol = self.corr_L_
        else:
            corr_chol = cholesky(self.kernel(Xc, Xc, **kwargs) + self.nugget * np.eye(len(Xc)))
        if y is None:
            y = self.y_train_

        # Use X and y from fit for hyperparameters
        m_old = self.mean(Xc)
        m_new = self.mean(X)

        # Now use X and y from arguments for conditioning/predictions
        R_on = self.kernel(Xc, X, **kwargs)
        R_no = R_on.T
        R_nn = self.kernel(X, X, **kwargs)

        # Use given y for prediction
        alpha = cho_solve((corr_chol, True), (y - m_old).T)
        m_pred = m_new + (R_no @ alpha).T
        if return_std or return_cov:
            half_quad = solve_triangular(corr_chol, R_on, lower=True)
            R_pred = R_nn - np.dot(half_quad.T, half_quad)
            if pred_noise:
                R_pred += self.noise_sd**2 * np.eye(len(X))
            # Use y from fit for hyperparameters
            std = self.std()
            K_pred = np.squeeze(std**2 * R_pred)
            if return_std:
                return m_pred, np.sqrt(np.diag(K_pred))
            return m_pred, K_pred
        return m_pred

    def sample_y(self, X, n_samples=1, random_state=0):
        """Taken from scikit-learn's gp module"""
        rng = check_random_state(random_state)

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

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        raise NotImplementedError

    def likelihood(self, log=True, X=None, y=None, **kernel_kws):
        raise NotImplementedError

    # def ratio_likelihood(self, ratio, y, corr_chol, orders=None):
    #     if y.ndim < 2:
    #         raise ValueError('y must be at least 2d, not {}'.format(y.shape))
    #     if orders is None:
    #         orders = np.arange(y.shape[0])
    #     ys = y / ratio[:, None, ...] ** orders[:, None]
    #     log_likes = np.array([self.likelihood(log=True, y=yi, corr_chol=corr_chol) for yi in ys])
    #     log_likes -= np.sum(orders) * np.sum(np.log(ratio), axis=-1)[:, None]
    #     return log_likes

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
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

        
class ConjugateGaussianProcess(ConjugateProcess):

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta)

        if eval_gradient:
            R, R_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            R = kernel(self.X_train_)

        R[np.diag_indices_from(R)] += self.nugget
        try:
            corr_L = cholesky(R)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        X = self.X_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        center0, disp0, df0, scale0 = self.center0, self.disp0, self.df0, self.scale0
        df = self.compute_df(y=y_train, df0=df0, eval_gradient=False)
        basis = self.basis(X)
        if eval_gradient:
            center, dcenter = self.compute_center(y_train, corr_L, basis, center0=center0, disp0=disp0,
                                                  eval_gradient=eval_gradient, dR=R_gradient)
            scale2, dscale2 = self.compute_scale_sq(
                y=y_train, chol=corr_L, basis=basis, center0=center0, disp0=disp0,
                df0=df0, scale0=scale0, eval_gradient=eval_gradient, dR=R_gradient)
            dvar = self.scale_sq_to_marginal_variance(scale_sq=dscale2, df=df)
            dmean = basis @ dcenter
        else:
            center = self.compute_center(y_train, corr_L, basis, center0=center0, disp0=disp0)
            scale2, dscale2 = self.compute_scale_sq(
                y=y_train, chol=corr_L, basis=basis, center0=center0, disp0=disp0,
                df0=df0, scale0=scale0)
        mean = basis @ center
        var = self.scale_sq_to_marginal_variance(scale_sq=scale2, df=df)

        # Convert from correlation matrix to covariance and subtract mean
        # to make all calculations below identical to scikit learn implementation
        L = np.sqrt(var) * corr_L
        K = var * R
        if eval_gradient:
            # R_gradient *= std**2
            K_gradient = var * R_gradient + dvar * R[:, :, None]
        y_train = y_train - mean[:, None]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)

            # Add gradient wrt mean
            log_likelihood_gradient_dims -= dmean.T @ alpha

            # Sum over output dimension
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def likelihood(self, log=True, X=None, y=None, **kernel_kws):
        # Multiple corr can be passed to quickly get likelihoods for many correlation parameters
        if X is None:
            X = self.X_train_
        if y is None:
            y = self.y_train_

        corr = self.kernel(X, **self.kernel_kws)
        corr = corr + self.nugget * np.eye(corr.shape[-1])
        corr_chol = cholesky(corr)
        
        # Setup best guesses for mean and cov
        beta0, disp0, df0, scale0 = self.beta0, self.disp0, self.df0, self.scale0
        basis = self.basis(X)
        mean = basis @ self.compute_beta(y, corr_chol, basis, beta0=beta0, disp0=disp0)
        sd = self.compute_std(y=y, chol=corr_chol, basis=basis, beta0=beta0, disp0=disp0, df0=df0, scale0=scale0)
        cov = sd ** 2 * corr
        dist = st.multivariate_normal(mean=mean, cov=cov)
        log_like = np.sum(dist.logpdf(y))
        if log:
            return log_like
        return np.exp(log_like)

    
class ConjugateStudentProcess(ConjugateProcess):

    def cov(self, X=None, Xp=None):
        if X is None:
            X = self.X_train_
        if Xp is None:
            Xp = X
        corr = self.kernel(X, Xp, **self.kernel_kws)
        return self.std()**2 * (corr + self.basis(X) @ self.disp() @ self.basis(Xp).T)

    def predict(self, X, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=True):
        pred = super(ConjugateStudentProcess, self).predict(
            X=X, return_std=return_std, return_cov=return_cov, Xc=Xc, y=y, pred_noise=pred_noise)

        basis = self.basis(X)
        mean_cov = self.std()**2 * (basis @ self.disp() @ basis.T)
        if return_std:
            mean, std = pred
            std += np.sqrt(np.diag(mean_cov))
            return mean, std
        if return_cov:
            mean, cov = pred
            cov += mean_cov
            return mean, cov
        return pred

    def likelihood(self, log=True, X=None, y=None, **kernel_kws):
        if X is None:
            X = self.X_train_
        if y is None:
            y = self.y_train_
        # if corr.ndim == 2:
        #     corr = corr[None, :, :]

        ny = self.num_y(y)
        # corr = corr + noise_sd**2 * np.eye(corr.shape[-1])
        # corr_chol_vec = np.linalg.cholesky(corr)
        corr = self.kernel(X, X, **kernel_kws)
        corr_chol = cholesky(corr + self.nugget * np.eye(corr.shape[-1]))

        beta0, disp0, df0, scale0 = self.beta0, self.disp0, self.df0, self.scale0
        df = self.compute_df(y=y, df0=df0)
        basis = self.basis(X)
        # disp_vec = [self.compute_disp(y=y, chol=chol, basis=basis, disp0=disp0) for chol in corr_chol_vec]
        # scale_vec = [self.compute_scale(y=y, chol=chol, basis=basis, beta0=beta0, disp0=disp0, df0=df0, scale0=scale0)
        #              for chol in corr_chol_vec]
        disp = self.compute_disp(y=y, chol=corr_chol, basis=basis, disp0=disp0)
        scale = self.compute_scale(y=y, chol=corr_chol, basis=basis, beta0=beta0, disp0=disp0, df0=df0, scale0=scale0)

        def log_norm(df_, scale_, disp_):
            """Normalization constant of the normal scaled inverse chi squared distribution"""
            norm = loggamma(df_ / 2.) - df_ / 2. * np.log(df_ * scale_ / 2.)
            log_det = np.linalg.slogdet(2 * np.pi * disp_)[1]
            if log_det != -np.inf:
                norm += 0.5 * log_det
            return norm

        log_det_corr = 2 * np.sum(np.log(2 * np.pi * np.diagonal(corr_chol)))
        log_like = log_norm(df, scale, disp) - log_norm(df0, scale0, disp0) - ny / 2. * log_det_corr

        # log_like = np.array([log_norm(df, scale, disp) for scale, disp in zip(scale_vec, disp_vec)])
        # log_det_corr = 2 * np.sum(np.log(2 * np.pi * np.diagonal(corr_chol_vec, axis1=-2, axis2=-1)), axis=-1)
        # log_like -= ny / 2. * log_det_corr + log_norm(df0, scale0, disp0)

        if log:
            return log_like
        return np.exp(log_like)


class TruncationProcess:

    def __init__(self, kernel=None, ratio=0.5, ref=1, excluded=None, ratio_kws=None, kernel_kws=None,
                 nugget=1e-10, verbose=True, **kwargs):
        R"""

        Parameters
        ----------
        kernel
        ratio
        ref
        excluded : 1d array
            The set of orders to ignore when constructing process for y_order and dy_order, i.e., the geometric sum
            will not include these values
        ratio_kws
        kernel_kws
        nugget
        verbose
        kwargs
        """
        if not callable(ref):
            self.ref = lambda X, *args, **kws: ref * np.ones(X.shape[0])
        else:
            self.ref = ref

        if not callable(ratio):
            self.ratio = lambda X, *args, **kws: ratio * np.ones(X.shape[0])
        else:
            self.ratio = ratio

        kwargs['nugget'] = nugget
        kwargs['verbose'] = False  # Handled by this class
        self.coeffs_process_class = ConjugateProcess
        self.coeffs_process_kwargs = kwargs
        self.coeffs_process = ConjugateProcess()
        self.kernel = kernel
        self._log_like = None

        self.X_train_ = None
        self.y_train_ = None
        self.coeffs = None
        self.orders = None
        self.excluded = None
        self.dX = None
        self.dy = None
        self.excluded = excluded
        self.nugget = nugget
        self.verbose = verbose

        self.all_kernel_kws = None
        if kernel_kws is None:
            kernel_kws = {}
        if isinstance(kernel_kws, (list, np.ndarray)):
            self.all_kernel_kws = np.array(kernel_kws)
        elif isinstance(kernel_kws, dict):
            self.kernel_kws = kernel_kws
        else:
            raise ValueError('kernel_kws must be a list or dict')

        self.all_ratio_kws = None
        if ratio_kws is None:
            ratio_kws = {}
        if isinstance(ratio_kws, (list, np.ndarray)):
            self.all_ratio_kws = np.array(ratio_kws)
        elif isinstance(ratio_kws, dict):
            self.ratio_kws = ratio_kws
        else:
            raise ValueError('ratio_kws must be a list or dict')

    @staticmethod
    def geometric_sum(x, start, end, excluded=None):
        """The geometric sum of x from `i=start` to `i=end` (inclusive)

        .. math::
            S = \sum_{i=start}^{end} x^i

        with the i in `exclude` excluded from the sum.

        Parameters
        ----------
        x : array
            The value to be summed
        start : int
            The start index of the sum
        end : int
            The end index of the sum (inclusive)
        excluded : int or 1d array-like of ints
            The indices to exclude from the sum

        Returns
        -------
        S : array
            The geometric sum
        """
        if end < start:
            raise ValueError('end must be greater than or equal to start')

        s = (x**start - x**(end+1)) / (1 - x)
        if excluded is not None:
            excluded = np.atleast_1d(excluded)
            for n in excluded:
                if (n >= start) and (n <= end):
                    s -= x**n
        return s

    def mean(self, X, start=0, end=np.inf):
        coeff_mean = self.coeffs_process.mean(X=X)
        ratio_sum = self.geometric_sum(x=self.ratio(X), start=start, end=end, excluded=self.excluded)
        return self.ref(X) * ratio_sum * coeff_mean

    def cov(self, X, Xp=None, start=0, end=np.inf):
        Xp = X if Xp is None else Xp
        coeff_cov = self.coeffs_process.cov(X=X, Xp=Xp)
        ratio_mat = self.ratio(X)[:, None] * self.ratio(Xp)
        ratio_sum = self.geometric_sum(x=ratio_mat, start=start, end=end, excluded=self.excluded)
        ref_mat = self.ref(X)[:, None] * self.ref(Xp)
        return ref_mat * ratio_sum * coeff_cov

    def basis(self, X, start=0, end=np.inf):
        cn_basis = self.coeffs_process.basis(X=X)
        ratio_sum = self.geometric_sum(x=self.ratio(X)[:, None], start=start, end=end, excluded=self.excluded)
        return self.ref(X)[:, None] * ratio_sum * cn_basis

    def fit(self, X, y, orders, dX=None, dy=None):
        self.X_train_ = X
        self.y_train_ = y
        self.orders = orders
        orders_mask = ~ np.isin(orders, self.excluded)

        self.dX = dX
        self.dy = dy

        # Find best kernel_kws and ratio_kws, if necessary
        if (self.all_kernel_kws is not None) or (self.all_ratio_kws is not None):
            # Setup corr_vec and ratios lists to feed into likelihood function
            if self.all_kernel_kws is not None:
                corr_vec = np.array([self.kernel(X, X, **kws) for kws in self.all_kernel_kws])
                placeholder_kernel_kws = self.all_kernel_kws[0]
            else:
                corr_vec = self.kernel(X, **self.kernel_kws)[None, :, :]
                placeholder_kernel_kws = self.kernel_kws

            if self.all_ratio_kws is not None:
                ratios = np.array([self.ratio(X, **kws) for kws in self.all_ratio_kws])
            else:
                ratios = self.ratio(X, **self.ratio_kws)[None, :]

            # Set a placeholder process for computing the likelihood
            self.coeffs_process = self.coeffs_process_class(
                kernel=self.kernel, kernel_kws=placeholder_kernel_kws, **self.coeffs_process_kwargs)
            log_like = self.likelihood(
                log=True, X=X, y=y, corr=corr_vec, orders=orders,
                ratio=ratios, ref=self.ref(X))
            self._log_like = log_like

            # Pick the best values by maximizing the likelihood
            best_idx = np.unravel_index(np.argmax(log_like), (len(ratios), len(corr_vec)))
            ratio_kws = self.ratio_kws if (self.all_ratio_kws is None) else self.all_ratio_kws[best_idx[0]]
            kernel_kws = self.kernel_kws if (self.all_kernel_kws is None) else self.all_kernel_kws[best_idx[1]]
            if self.verbose:
                print('Setting kernel kwargs to {}'.format(kernel_kws))
                print('Setting ratio kwargs to {}'.format(ratio_kws))
            self.kernel_kws = kernel_kws
            self.ratio_kws = ratio_kws

        # Extract the coefficients based on best ratio value and setup/fit the iid coefficient process
        self.coeffs = coefficients(partials=y, ratio=self.ratio(X, **self.ratio_kws), X=X,
                                   ref=self.ref(X), orders=orders)[orders_mask]
        self.coeffs_process = self.coeffs_process_class(
            kernel=self.kernel, kernel_kws=self.kernel_kws, **self.coeffs_process_kwargs)
        self.coeffs_process.fit(X=X, y=self.coeffs)
        return self

    def predict(self, X, order, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=True):
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
        y : (n, N) array
            Points upon which to condition. Defaults to the `y` used in `fit`. This *does not*
            affect the `y` used to update hyperparameters.
        pred_noise : bool
            Adds `noise_sd` to the diagonal of the covariance matrix if `return_cov == True`.

        Returns
        -------
        mean, (mean, std), or (mean, cov), depending on `return_std` and `return_cov`
        """

        if Xc is None:
            Xc = self.X_train_
        if y is None:
            y = np.squeeze(self.y_train_[self.orders == order])

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
        alpha = solve(K_oo, (y - m_old).T)
        m_pred = m_new + (K_no @ alpha).T
        K_pred = None
        if return_std or return_cov:
            K_pred = K_nn - K_no @ solve(K_oo, K_on)
        #
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Get the mean & cov for truncation error
        #
        m_new_trunc = self.mean(X=X, start=order + 1, end=np.inf)
        K_nn_trunc = self.cov(start=order + 1, end=np.inf, X=X, Xp=X)

        X_trunc = self.dX
        if X_trunc is not None:  # truncation error is constrained
            m_old_trunc = self.mean(X=X_trunc, start=order+1, end=np.inf)
            K_oo_trunc = self.cov(X=X_trunc, Xp=X_trunc, start=order+1, end=np.inf)
            K_on_trunc = self.cov(X=X_trunc, Xp=X, start=order+1, end=np.inf)
            K_no_trunc = K_on_trunc.T

            alpha_trunc = solve(K_oo_trunc, (self.dy - m_old_trunc))
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

    def likelihood(self, log=True, X=None, y=None, orders=None, ratio_kws=None, **kernel_kws):
        X = self.X_train_ if X is None else X
        y = self.y_train_ if y is None else y
        orders = self.orders if orders is None else orders
        ratio_kws = {} if ratio_kws is None else ratio_kws

        for v in [X, y, orders]:
            if v is None:
                raise ValueError('All of X, y, and orders must be given if model is not fit')

        ref = self.ref(X)
        ratio = self.ratio(X, **ratio_kws)

        orders_mask = ~ np.isin(orders, self.excluded)
        coeffs = coefficients(partials=y, ratio=ratio, X=X, ref=ref, orders=orders)[orders_mask]
        coeff_log_like = self.coeffs_process.likelihood(log=True, X=X, y=coeffs, **kernel_kws)

        orders = orders[orders_mask]
        det_factor = np.sum(len(orders) * np.log(np.abs(ref)) + np.sum(orders) * np.log(np.abs(ratio)), axis=-1)
        y_log_like = coeff_log_like - det_factor

        # if corr is None:
        #     corr = np.array([self.kernel(X, X, **kws) for kws in self.all_kernel_kws])

        # if callable(ref):
        #     ref = ref(X)
        # if callable(ratio):
        #     ratio = ratio(X)

        # if ratio.ndim == 1:
        #     ratio = ratio[None, :]
        # if ref.ndim == 1:
        #     ref = ref[None, :]

        # Ensure each can be looped over below
        # ratio, ref = np.broadcast_arrays(ratio, ref)
        #
        # # Compute pr(c | \ell) for each set of coefficients determined by ref, ratio
        # orders_mask = orders != excluded
        # coeffs = [coefficients(partials=y, ratio=ratio_, X=X, ref=ref_, orders=orders)[orders_mask]
        #           for ratio_, ref_ in zip(ratio, ref)]
        # coeff_log_like = np.array([self.coeffs_process.likelihood(log=True, X=X, y=c, corr=corr, noise_sd=noise_sd)
        #                           for c in coeffs])
        #
        # # Now convert to pr(y | \ell, ratio) with determinant factor
        # orders = orders[orders_mask]
        # det_factor = np.sum(len(orders) * np.log(np.abs(ref)) + np.sum(orders) * np.log(np.abs(ratio)), axis=-1)
        # y_log_like = (coeff_log_like.T - det_factor).T  # Transpose handles possible kernel kw dimension

        if log:
            return y_log_like
        return np.exp(y_log_like)


class TruncationGP(TruncationProcess):

    def __init__(self, kernel, ref, ratio, ratio_kws=None, kernel_kws=None, **kwargs):
        super().__init__(
            kernel=kernel, ref=ref, ratio=ratio, ratio_kws=ratio_kws, kernel_kws=kernel_kws, **kwargs)
        self.coeffs_process_class = ConjugateGaussianProcess
        self.coeffs_process = ConjugateGaussianProcess()


class TruncationTP(TruncationProcess):

    def __init__(self, kernel=None, ratio=0.5, ref=1, ratio_kws=None, kernel_kws=None, **kwargs):
        super().__init__(
            kernel=kernel, ratio=ratio, ref=ref, ratio_kws=ratio_kws, kernel_kws=kernel_kws, **kwargs)
        self.coeffs_process_class = ConjugateStudentProcess
        self.coeffs_process = ConjugateStudentProcess()

    def predict(self, X, order, excluded=None, return_std=False, return_cov=False, Xc=None, y=None, pred_noise=True):
        pred = super(TruncationTP, self).predict(
            X=X, order=order, return_std=return_std, return_cov=return_cov,
            Xc=Xc, y=y, pred_noise=pred_noise
        )

        if not return_std and not return_cov:
            return pred

        if Xc is None:
            Xc = self.X_train_

        # Use Xc from argument to define old points
        K_oo = self.cov(start=0, end=order, excluded=excluded, X=Xc, Xp=Xc)
        K_no = self.cov(start=0, end=order, excluded=excluded, X=X, Xp=Xc)

        basis_lower_old = self.basis(start=0, end=order, excluded=excluded, X=Xc)
        basis_lower_new = self.basis(start=0, end=order, excluded=excluded, X=X)
        basis_lower = basis_lower_new - K_no @ solve(K_oo, basis_lower_old)

        if self.Xt is not None:  # truncation error is constrained
            K_oo_trunc = self.cov(start=order+1, end=np.inf, excluded=excluded, X=self.Xt, Xp=self.Xt)
            K_no_trunc = self.cov(start=order+1, end=np.inf, excluded=excluded, X=X, Xp=self.Xt)

            basis_trunc_old = self.basis(start=order+1, end=np.inf, excluded=excluded, X=self.Xt)
            basis_trunc_new = self.basis(start=order+1, end=np.inf, excluded=excluded, X=X)
            basis_trunc = basis_trunc_new - K_no_trunc @ solve(K_oo_trunc, basis_trunc_old)
        else:  # not constrained
            basis_trunc = self.basis(start=order + 1, end=np.inf, excluded=excluded, X=X)
        mean_cov = (basis_lower + basis_trunc) @ self.coeffs_process.disp() @ (basis_lower + basis_trunc).T
        mean_cov *= self.coeffs_process.std() ** 2

        if return_std:
            mean, std = pred
            return mean, std + np.sqrt(np.diag(mean_cov))
        if return_cov:
            mean, cov = pred
            return mean, cov + mean_cov


class TruncationPointwise:

    def __init__(self):
        pass




















































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

        right_quad = solve_triangular(R_chol, y.T, lower=True)
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

        Rinv_y = cho_solve((R_chol, True), avg_y)
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
