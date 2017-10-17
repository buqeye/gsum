# Testing for now
import numpy as np
import pymc3 as pm
from pymc3.distributions.dist_math \
    import bound, logpow, factln, Cholesky
from pymc3.distributions.distribution \
    import Continuous, Discrete, draw_values, generate_samples
from scipy import stats, linalg
import theano
import theano.tensor as tt


class MatrixNormal(Continuous):
    R"""
    Matrix-valued normal log-likelihood.

    ===============  =====================================
    Support          :math:`x \in \mathbb{R}^{q \times p}`
    Mean             :math:`\mu`
    Right Variance   :math:`T_R^{-1}`
    Left Variance    :math:`T_L^{-1}`
    ===============  =====================================

    Parameters
    ----------
    mu : array
        Vector of means.
    rcov : array
        Right (or row) pxp covariance matrix. Defines variance within rows.
        Exactly one of rcov or rchol is needed.
    rchol : array
        Cholesky decomposition of pxp covariance matrix. Defines variance
        within rows. Exactly one of rcov or rchol is needed.
    lcov : array
        Left (or column) qxq covariance matrix. Defines variance within
        columns. Exactly one of lcov or lchol is needed.
    lchol : array
        Cholesky decomposition of qxq covariance matrix. Defines variance
        within columns. Exactly one of lcov or lchol is needed.

    Examples
    --------
    ?
    """

    def __init__(self, mu=0, rcov=None, rchol=None, rtau=None,
                 lcov=None, lchol=None, ltau=None, *args, **kwargs):

        self.setup_matrices(rcov, rchol, rtau, lcov, lchol, ltau)

        shape = kwargs.pop('shape', None)
        assert len(shape) == 2, "only 2d tuple inputs work right now: qxp"
        self.shape = shape

        super(MatrixNormal, self).__init__(shape=shape, *args, **kwargs)

        self.mu = tt.as_tensor_variable(mu)

        self.mean = self.median = self.mode = self.mu

        # self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        # self.solve_upper = tt.slinalg.Solve(A_structure="upper_triangular")
        self.solve_lower = tt.slinalg.solve_lower_triangular
        self.solve_upper = tt.slinalg.solve_upper_triangular

    def setup_matrices(self, rcov, rchol, rtau, lcov, lchol, ltau):
        # Step methods and advi do not catch LinAlgErrors at the
        # moment. We work around that by using a cholesky op
        # that returns a nan as first entry instead of raising
        # an error.
        cholesky = Cholesky(nofail=True, lower=True)

        # Right (or row) matrices
        if len([i for i in [rtau, rcov, rchol] if i is not None]) != 1:
            raise ValueError('Incompatible parameterization. '
                             'Specify exactly one of rtau, rcov, '
                             'or rchol.')
        if rcov is not None:
            self.p = rcov.shape[0]  # How many points along vector
            self._rcov_type = 'cov'
            rcov = tt.as_tensor_variable(rcov)
            if rcov.ndim != 2:
                raise ValueError('rcov must be two dimensional.')
            self.rchol_cov = cholesky(rcov)
            self.rcov = rcov
            # self._n = self.rcov.shape[-1]
        elif rtau is not None:
            raise ValueError('rtau not supported at this time')
            self.p = rtau.shape[0]
            self._rcov_type = 'tau'
            rtau = tt.as_tensor_variable(rtau)
            if rtau.ndim != 2:
                raise ValueError('rtau must be two dimensional.')
            self.rchol_tau = cholesky(rtau)
            self.rtau = rtau
            # self._n = self.rtau.shape[-1]
        else:
            self.p = rchol.shape[0]
            self._rcov_type = 'chol'
            if rchol.ndim != 2:
                raise ValueError('rchol must be two dimensional.')
            self.rchol_cov = tt.as_tensor_variable(rchol)
            # self._n = self.rchol_cov.shape[-1]

        # Left (or column) matrices
        if len([i for i in [ltau, lcov, lchol] if i is not None]) != 1:
            raise ValueError('Incompatible parameterization. '
                             'Specify exactly one of ltau, lcov, '
                             'or lchol.')
        if lcov is not None:
            self.q = lcov.shape[0]
            self._lcov_type = 'cov'
            lcov = tt.as_tensor_variable(lcov)
            if lcov.ndim != 2:
                raise ValueError('lcov must be two dimensional.')
            self.lchol_cov = cholesky(lcov)
            self.lcov = lcov
            # self._n = self.lcov.shape[-1]
        elif ltau is not None:
            raise ValueError('ltau not supported at this time')
            self.q = ltau.shape[0]
            self._lcov_type = 'tau'
            ltau = tt.as_tensor_variable(ltau)
            if ltau.ndim != 2:
                raise ValueError('ltau must be two dimensional.')
            self.lchol_tau = cholesky(ltau)
            self.ltau = ltau
            # self._n = self.ltau.shape[-1]
        else:
            self.q = lchol.shape[0]
            self._lcov_type = 'chol'
            if lchol.ndim != 2:
                raise ValueError('lchol must be two dimensional.')
            self.lchol_cov = tt.as_tensor_variable(lchol)
            # self._n = self.lchol_cov.shape[-1]

    def random(self, point=None, size=None):
        if size is None:
            size = list(self.shape)

        mu, rchol, lchol = draw_values([self.mu, self.rchol_cov, self.lchol_cov], point=point)
        standard_normal = np.random.standard_normal(size)

        return mu + lchol @ standard_normal @ rchol.T

    def _trquaddist(self, value):
        """Compute Tr[rcov^-1 @ (x - mu).T @ lcov^-1 @ (x - mu)] and
        the logdet of rcov and lcov."""
        mu = self.mu

        delta = value - mu

        lchol_cov = self.lchol_cov
        rchol_cov = self.rchol_cov

        rdiag = tt.nlinalg.diag(rchol_cov)
        ldiag = tt.nlinalg.diag(lchol_cov)
        # Check if the covariance matrix is positive definite.
        rok = tt.all(rdiag > 0)
        lok = tt.all(ldiag > 0)
        ok = rok and lok

        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        rchol_cov = tt.switch(rok, rchol_cov, 1)
        lchol_cov = tt.switch(lok, lchol_cov, 1)

        # Find exponent piece by piece
        right_quaddist = self.solve_lower(lchol_cov, delta)
        quaddist = tt.nlinalg.matrix_dot(right_quaddist.T, right_quaddist)
        quaddist = self.solve_lower(rchol_cov, quaddist)
        quaddist = self.solve_upper(rchol_cov.T, quaddist)
        trquaddist = tt.nlinalg.trace(quaddist)

        half_rlogdet = tt.sum(tt.log(rdiag))  # logdet(M) = 2*Tr(log(L))
        half_llogdet = tt.sum(tt.log(ldiag))  # Using Cholesky: M = L L^T

        return trquaddist, half_rlogdet, half_llogdet, ok

    def logp(self, value):
        trquaddist, half_rlogdet, half_llogdet, ok = self._trquaddist(value)
        q = self.q
        p = self.p
        norm = - 0.5 * q * p * pm.floatX(np.log(2 * np.pi))
        return bound(
                norm - 0.5 * trquaddist - q * half_rlogdet - p * half_llogdet,
                ok)
