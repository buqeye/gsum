import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import unittest
from gsum import *
from gsum.helpers import *
from pymc3.gp.cov import ExpQuad


class TestGaussianKernel(unittest.TestCase):
    """Test the Gaussian kernel.
    """


    def test_gaussian_kernel(self):
        ls = 2.5

        X = np.asarray([
            [0, 0],
            [0.5, 0],
            [1, 2],
            [1, 2.7],
            [3, 3],
            [3.001, 3.001]
            ])

        pm_cov = ExpQuad(2, ls)
        cov = gaussian(X, ls=ls)

        np.testing.assert_allclose(pm_cov(X).eval(), cov)

    def test_gaussian_kernel2(self):
        ls = 2.5

        X = np.linspace(0, 1, 100)[:, None]

        pm_cov = ExpQuad(1, ls)
        cov = gaussian(X, ls=ls)

        np.testing.assert_allclose(pm_cov(X).eval(), cov)

if __name__ == '__main__':
    unittest.main()
