"""auxlib module provides several useful low-level functions as number thresholding or a couple of linear algebra
operations"""

import numpy as np

_epsilon = 0.0001


def to_colour_1(x):
    """Convert number to float in range [0,1]

    Parameters
    ----------
    x : convertible to float
        number that will be cut to interval [0,1]

    Returns
    -------
    float
        0 if x < 0, 1 if x > 1. In other cases x converted to float.
    """
    x = float(x)
    x = 0. if x < 0 else x
    x = 1. if x > 1 else x
    return x


def to_colour_255(x):
    """Convert number to int in range [0,255]

    Parameters
    ----------
    x : convertible to int
        number that will be cut to interval [0,255]

    Returns
    -------
    int
        0 if x < 0, 255 if x > 255. In other cases x converted to int
    """
    x = int(x)
    x = 0 if x < 0 else x
    x = 255 if x > 255 else x
    return x


def positive(x):
    """Make number be strictly positive

    Parameters
    ----------
    x : convertible to float
        a number

    Returns
    -------
    float
        _epsilon if x is not strictly positive, otherwise x
    """
    return _epsilon if x <= 0 else float(x)


def negative(x):
    """Make number be non-positive

    Parameters
    ----------
    x : convertible to float
        a number

    Returns
    -------
    float
        0 if x is positive, otherwise x converted to float
    """
    return 0. if x >= 0 else float(x)


def check_positivity(r):
    """Check if all numbers in a container are strictly positive

    Parameters
    ----------
    r : iterable
        an object with int or float values. Each element is checked whether it is positive

    Returns
    -------
    bool
        True if all elements in r are strictly positive, False otherwise
    """
    for i in r:
        if i <= 0:
            return False
    return True


def find_vals(a, r):
    """Get density coefficients from logarithmic basis matrix and pixel values.

    Parameters
    ----------
    a : ndarray
        logarithmic basis matrix, shape (3,2)
    r : ndarray
        pixel values, shape (3,)

    Returns
    -------
    ndarray
        density coefficients, shape (2,)
    """
    m00, m01, m11, v0, v1 = 0, 0, 0, 0, 0
    for i in range(3):
        m00 += a[i, 0] ** 2
        m01 += a[i, 0] * a[i, 1]
        m11 += a[i, 1] ** 2
        v0 += r[i] * a[i, 0]
        v1 += r[i] * a[i, 1]
    return -np.linalg.solve(np.array([[m00, m01], [m01, m11]]), np.array([v0, v1]))


def get_physical_normal(n):
    """Given unit vector, find the nearest physical unit vector.
       A physical unit vector can't have all component positive,
       can't have all components negative, and can't have exactly
       one zero component.

    Parameters
    ----------
    n : ndarray
        unit vector, shape (3,)

    Returns
    -------
    ndarray
        physical unit vector, shape (3,)
    """

    n = np.array(n)

    one_zero = (n[0] == 0 and n[1] * n[2] != 0) or\
               (n[1] == 0 and n[2] * n[0] != 0) or\
               (n[2] == 0 and n[0] * n[1] != 0)

    if not check_positivity(n) and not check_positivity(-n) and not one_zero:
        return n

    # print("Best fitting plane non-physical, attempting to correct...")
    m = n

    if check_positivity(n) or check_positivity(-n):
        minimum = np.abs(n[0])
        index = 0
        for i in range(1, 3):
            if np.abs(n[i]) < minimum:
                index = i
                minimum = np.abs(n[i])

        n[index] = 0
        n = n / np.linalg.norm(n)

    one_zero = (n[0] == 0 and n[1] * n[2] != 0) or\
               (n[1] == 0 and n[2] * n[0] != 0) or\
               (n[2] == 0 and n[0] * n[1] != 0)

    # Correction for normal vector with exactly one zero component
    if one_zero:
        minimum = 2
        index = 0
        for i in range(3):
            if n[i] != 0 and np.abs(n[i]) < minimum:
                index = i
                minimum = np.abs(n[i])

        n[index] = 0
        n = n / np.linalg.norm(n)

    # print("Correction error is: ", np.linalg.norm(m - n))
    return n


def get_basis_from_normal(n):
    """Finds physical colour basis from given physical (positive) versor (unit vector).

    Parameters
    ----------
    n : ndarray
        physical unit vector, shape (3,)

    Returns
    -------
    ndarray
        (log) basis, shape (2,3)
    """

    v = {}
    v[0] = [0, abs(n[2]), abs(n[1])]
    v[1] = [abs(n[2]), 0, abs(n[0])]
    v[2] = [abs(n[1]), abs(n[0]), 0]

    good = [(n[(i + 1) % 3] * n[(i + 2) % 3] <= 0) and\
            (n[(i + 1) % 3] != 0 or n[(i + 2) % 3] != 0) for i in range(3)]

    basis = [v[i] for i in range(3) if good[i]]
    assert(len(basis) > 1)

    basis = np.array(basis)
    basis = [v / np.linalg.norm(v) for v in basis]
    return basis[:2]


def orthonormal_rotation(v):
    """Gives a (orthonormal) rotation matrix transforming [1, 0, 0] to the given vector.

    Parameters
    ----------
    v : ndarray
        versor (unit vector), shape (3,)
    Returns
    -------
    ndarray
        rotation matrix, shape (3,3)
    """
    if v[0] == 1.:
        return np.identity(3)

    u = np.array([0., v[1], v[2]]) / np.array((v[1] ** 2 + v[2] ** 2))
    u = u - v * np.dot(u, v)
    u = u / np.linalg.norm(u)
    w = np.cross(v, u)

    return np.transpose(np.array([v, u, w]))


def find_vector(mat):
    """Special eigenvector of a special matrix.

    Finds eigenvector associated with the smallest eigenvalue and turns it into a unit vector (versor). Assumes that
    matrix is in special coordinates such first row and first column (`mat[0,:]`, `mat[:,0]`) do not matter

    Parameters
    ----------
    mat : ndarray
        (3,3) list or numpy array

    Returns
    -------
    ndarray
        normed eigenvector, shape (3,)
    """
    eig = np.linalg.eig([[mat[1, 1], mat[1, 2]], [mat[2, 1], mat[2, 2]]])

    minimum = eig[0][0]
    index = 0
    for i in range(1, 2):
        if eig[0][i] < minimum:
            minimum = eig[0][i]
            index = i

    n = [0, eig[1][0][index], eig[1][1][index]]
    n = n / np.linalg.norm(n)

    return n
