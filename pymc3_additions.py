# Testing for now
import pymc3 as pm
from pm.distributions.multivariate import MvNormal, _QuadFormBase
import theano
import theano.tensor as tt


class MatNormal(MvNormal):
    R"""
    Matrix-valued normal log-likelihood.
    """

    def __init__(self, mu, rcov, lcov, *args, **kwargs):

        self.n_output = lcov.shape[1]  # How many output vectors
        self.n_mv = rcov.shape[0]      # Length of the vectors
        super(MatNormal, self).__init__(mu=mu, cov=rcov,
                                        shape=(self.n_output, self.n_mv),
                                        *args, **kwargs)

        self.M = mu

        # Step methods and advi do not catch LinAlgErrors at the
        # moment. We work around that by using a cholesky op
        # that returns a nan as first entry instead of raising
        # an error.
        cholesky = Cholesky(nofail=True, lower=True)

        lcov = tt.as_tensor_variable(cov)
        if lcov.ndim != 2:
            raise ValueError('lcov must be two dimensional.')
        self.lchol_cov = cholesky(lcov)
        self.lcov = lcov

    def random(self, point=None, size=None):
        # mv_normal = super(MatNormal, self).random(self, point, size)

        if size is None:
            size = []
        else:
            try:
                size = list(size)
            except TypeError:
                size = [size]

        if self._cov_type == 'cov':
            mu, cov = draw_values([self.mu, self.cov], point=point)
            if mu.shape[-1] != cov.shape[-1]:
                raise ValueError("Shapes for mu and cov don't match")

            try:
                dist = stats.multivariate_normal(
                    mean=mu, cov=cov, allow_singular=True)
            except ValueError:
                size.append(mu.shape[-1])
                return np.nan * np.zeros(size)
            return dist.rvs(size)
        
        return self.M + np.dot(mv_normal, self.lchol_cov.T)

    def lchol_logdet(self, delta):
        lchol_cov = self.lchol_cov
        diag = tt.nlinalg.diag(lchol_cov)
        # Check if the covariance matrix is positive definite.
        ok = tt.all(diag > 0)
        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        lchol_cov = tt.switch(ok, lchol_cov, 1)

        logdet = tt.sum(tt.log(diag))
        return logdet

    def logp(self, value):
        quaddist, logdet, ok = self._quaddist(value)
        l_logdet = lchol_logdet(value - self.mu)
        tr_quaddist = np.dot(self.)
        k = value.shape[-1].astype(theano.config.floatX)
        norm = - 0.5 * self.n_output * k * pm.floatX(np.log(2 * np.pi))
        return bound(norm - 0.5 * quaddist - logdet - l_logdet, ok)
