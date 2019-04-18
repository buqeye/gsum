import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
# import unittest
from gsum import ConjugateGaussianProcess
from gsum.helpers import *


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
kernels = [RBF(length_scale=1.0),
           fixed_kernel,
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
           C(1.0, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
           C(1.0, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
           C(1e-5, (1e-5, 1e2)),
           C(0.1, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
           WhiteKernel(1e-2, (1e-5, 1e2))
           ]
non_fixed_kernels = [kernel for kernel in kernels
                     if kernel != fixed_kernel]


@pytest.mark.parametrize('kernel', kernels)
def test_gpr_interpolation(kernel):
    # Test the interpolating property for different kernels.
    print(kernel)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0., decimal=10)


@pytest.mark.parametrize('kernel', kernels)
def test_cgp_interpolation(kernel):
    # Test the interpolating property for different kernels.
    print(kernel)
    gpr = ConjugateGaussianProcess(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0., decimal=10)


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
