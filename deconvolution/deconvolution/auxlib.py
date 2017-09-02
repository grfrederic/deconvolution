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
    """Given versor (unit vector), find the nearest positive versor.

    Parameters
    ----------
    n : ndarray
        versor (unit vector), shape (3,)

    Returns
    -------
    ndarray
        positive versor, shape (3,)
    """
    if (n[0] > 0 and n[1] > 0 and n[2] > 0) or (n[0] < 0 and n[1] < 0 and n[2] < 0):
        print("Best fitting plane non-physical, attempting to correct...")
        minimum = n[0] ^ 2
        index = 0
        for i in range(1, 3):
            if n[i] ^ 2 < minimum:
                index = i
                minimum = n[i] ^ 2

        m = n
        n[index] = 0
        n = n / np.linalg.norm(n)
        print("Correction error is: ", np.linalg.norm(m - n))

    # Correction for normal vector with exactly one zero component
    if (n[0] == 0 and n[1] * n[2] != 0) or (n[1] == 0 and n[2] * n[0] != 0) or (n[2] == 0 and n[0] * n[1] != 0):
        print("Best fitting plane non-physical, attempting to correct...")
        minimum = 2
        index = 0
        for i in range(3):
            if n[i] != 0 and n[i] ** 2 < minimum:
                index = i
                minimum = n[i] ** 2

        m = n
        n[index] = 0
        n = n / np.linalg.norm(n)
        print("Correction error is: ", np.linalg.norm(m - n))

    return n


def get_basis_from_normal(n):
    """Finds physical colour basis from given physical (positive) versor (unit vector).

    Parameters
    ----------
    n : ndarray
        physical versor, shape (3,)

    Returns
    -------
    ndarray
        (log) basis, shape (2,3)
    """
    axial_flag = False
    if n[0] * n[1] * n[2] == 0:
        axial_flag = True

    if axial_flag:
        index = 0
        for i in range(3):
            if n[i] != 0:
                index = i

        x = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        y = [[0, 0, 0], [0, 0, 0]]
        j = 0
        for i in range(3):
            if i != index:
                y[j] = x[j]
                j += 1

        return y

    # If n is not a versor
    x = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    best = [0., 0., 0.]
    for i in x:
        pr = i - np.dot(i, n) * n
        if pr[0] > 0 and pr[1] > 0 and pr[2] > 0:
            best = pr

    y = [1., 1., 1.] - np.dot([1., 1., 1.], n) * n + 2 * best / min(best)
    x = best

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    k = min([(y[i] / x[i] if x[i] != 0 else 1e5) for i in range(3)])
    y -= k * x

    k = min([(x[i] / y[i] if y[i] != 0 else 1e5) for i in range(3)])
    x -= k * y

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    return [x, y]


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

    return np.array([v, u, w])


def find_vector(mat):
    """Find the eigenvector associated with the smallest eigenvalue.

    ????????

    Parameters
    ----------
    mat : ndarray
        (2??,2??) list or numpy array

    Returns
    -------
    ndarray
        eigenvector, shape (3,)
    """
    # here ->
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
