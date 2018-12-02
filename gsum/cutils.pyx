cimport cython
from cython_gsl cimport *
import numpy as np
cimport numpy as np


cdef extern from "gsl/gsl_linalg.h":
    # Pivoted Cholesky Decomposition
    int  gsl_linalg_pcholesky_decomp(gsl_matrix * A, gsl_permutation * P) nogil


cdef extern from "gsl/gsl_permutation.h": 
    int gsl_permute_matrix (const gsl_permutation * p, gsl_matrix * A)


def pivoted_cholesky(arr):
    """Returns the pivoted cholesky decomposition of arr.

    Parameters
    ----------
    arr : 2darray
        The matrix to decompose

    Returns
    -------
    The pivoted cholesky decomposition G such that G.G^T = arr
    """
    # Variable sized array info stolen from
    # http://docs.cython.org/en/latest/src/userguide/memoryviews.html#pass-data-from-a-c-function-via-pointer
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.
    n = arr.shape[0]
    cdef double[::1] aa = arr.ravel().copy()
    cdef double[::1] ii = np.eye(n).ravel()
    cdef gsl_permutation *P
    # GSL wants its special gsl matrices
    P = gsl_permutation_alloc(n)
    A = gsl_matrix_view_array(&aa[0], n, n)
    I = gsl_matrix_view_array(&ii[0], n, n)

    # This constructs LDL^T where L is lower triangular with ones on the diagonal
    # Turns aa into a matrix with L on the lower triangle, D on the diagonal, and ones on the upper triangle
    gsl_linalg_pcholesky_decomp(&A.matrix, P)

    # Now create lower triangular matrix R from aa described above
    chol = np.tril(np.array(aa).reshape(arr.shape))
    d = np.sqrt(np.diag(chol))
    chol[range(n), range(n)] = 1.
    # Multiply column i by d_i
    chol *= d
    
    # P is some permutation object but I want a matrix
    gsl_permute_matrix(P, &I.matrix)  # Permute the identity to turn ii into a permutation matrix
    perm = np.array(ii).reshape(arr.shape)
    return perm @ chol
