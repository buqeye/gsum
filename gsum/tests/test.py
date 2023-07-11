import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
# import unittest
from gsum import ConjugateGaussianProcess
from gsum.helpers import *
from gsum.helpers import pivoted_cholesky


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import DotProduct

from sklearn.utils.testing \
    import (assert_greater, assert_array_less,
            assert_almost_equal, assert_equal, assert_raise_message,
            assert_array_almost_equal, assert_array_equal)

import pytest


def f(x):
    return x * np.sin(x)


X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
X2 = np.atleast_2d([2., 4., 5.5, 6.5, 7.5]).T
y = f(X).ravel()

fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
kernel_ill_conditioned = RBF(length_scale=15.0, length_scale_bounds="fixed")
kernels = [
    RBF(length_scale=1.0),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) *
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) *
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
    C(1e-5, (1e-5, 1e2)),
    # C(0.1, (1e-2, 1e2)) *
    # RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
    # WhiteKernel(1e-2, (1e-5, 1e2))
]
non_fixed_kernels = [kernel for kernel in kernels
                     if kernel != fixed_kernel]

# kernels_ill_conditioned = [RBF(length_scale=20.0)]


@pytest.mark.parametrize('kernel', kernels)
def test_gpr_interpolation(kernel):
    # Test the interpolating property for different kernels.
    print(kernel)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0., decimal=10)


@pytest.mark.parametrize('kernel', kernels)
@pytest.mark.parametrize('decomposition', ['cholesky', 'eig'])
def test_cgp_interpolation(kernel, decomposition):
    # Test the interpolating property for different kernels.
    print(kernel)
    gpr = ConjugateGaussianProcess(kernel=kernel, nugget=0, decomposition=decomposition).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0., decimal=10)


Ls = [
    np.array([
        [7., 0, 0, 0, 0, 0],
        [9, 13, 0, 0, 0, 0],
        [4, 10, 6, 0, 0, 0],
        [18, 1, 2, 14, 0, 0],
        [5, 11, 20, 3, 17, 0],
        [19, 12, 16, 15, 8, 21]
    ]),
    np.array([
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 6.]
    ]),
    np.array([
        [6, 0, 0],
        [3, 2, 0],
        [4, 1, 5.]
    ]),
]

pchols = [
    np.array([
        [3.4444, -1.3545, 4.084, 1.7674, -1.1789, 3.7562],
        [8.4685, 1.2821, 3.1179, 12.9197, 0.0000, 0.0000],
        [7.5621, 4.8603, 0.0634, 7.3942, 4.0637, 0.0000],
        [15.435, -4.8864, 16.2137, 0.0000, 0.0000, 0.0000],
        [18.8535, 22.103, 0.0000, 0.0000, 0.0000, 0.0000],
        [38.6135, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ]),
    np.array([
        [0.4558, 0.3252, 0.8285],
        [2.6211, 2.4759, 0.0000],
        [8.7750, 0.0000, 0.0000]
    ]),
    np.array([
        [3.7033, 4.7208, 0.0000],
        [2.1602, 2.1183, 1.9612],
        [6.4807, 0.0000, 0.0000]
    ]),
]


@pytest.mark.parametrize('L,pchol', zip(Ls, pchols))
def test_oracle_examples(L, pchol):
    """Inputs taken from Tensorflow-Probability, which was taken from GPyTorch"""
    mat = np.matmul(L, L.T)
    np.testing.assert_allclose(pchol, pivoted_cholesky(mat), atol=1e-4)


# class TestGaussianKernel(unittest.TestCase):
#     """Test the Gaussian kernel.
#     """
#
#
#     def test_gaussian_kernel(self):
#         ls = 2.5
#
#         X = np.asarray([
#             [0, 0],
#             [0.5, 0],
#             [1, 2],
#             [1, 2.7],
#             [3, 3],
#             [3.001, 3.001]
#             ])
#
#         pm_cov = ExpQuad(2, ls)
#         cov = gaussian(X, ls=ls)
#
#         np.testing.assert_allclose(pm_cov(X).eval(), cov)
#
#     def test_gaussian_kernel2(self):
#         ls = 2.5
#
#         X = np.linspace(0, 1, 100)[:, None]
#
#         pm_cov = ExpQuad(1, ls)
#         cov = gaussian(X, ls=ls)
#
#         np.testing.assert_allclose(pm_cov(X).eval(), cov)
#
# if __name__ == '__main__':
#     unittest.main()
