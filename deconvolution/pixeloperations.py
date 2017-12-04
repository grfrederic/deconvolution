"""
pixeloperations provides an implementation of PixelOperations class used by ImageFrame to interact with basis
and background
"""

import numpy as np
import deconvolution.auxlib as aux
import deconvolution.exceptions as ex
from copy import deepcopy

_infzero = 0.00001
_white255 = np.array([255, 255, 255], dtype=float)
_white1 = np.array([1, 1, 1], dtype=float)


def _entries_in_closed_interval(vect):
    """Checks if all list components are in [0,1]

    Parameters
    ----------
    vect : array_like
        Container with values to check

    Returns
    -------
    bool
        True if all components lie in the interval [0,1]. False otherwise
    """
    if len(vect) == 0:
        return True
    try:
        return not (np.amax(vect) > 1 or np.amin(vect) < 0)
    except TypeError:
        return False


def _entries_in_half_closed_interval(vect):
    """Checks if all list components are in (0,1]

    Parameters
    ----------
    vect : array_like
        Container with values to check

    Returns
    -------
    bool
        True if all components lie in the interval [0,1]. False otherwise
    """
    if len(vect) == 0:
        return True
    try:
        return not (np.amax(vect) > 1 or np.amin(vect) <= 0)
    except TypeError:
        return False


def _array_to_colour_255(arr):
    """Changes array of numbers into arrays of colour entries

    Parameters
    ---------
    arr : ndarray
        numpy array of shape (x,y,3) with float or int values

    Returns
    -------
    ndarray
        array entries converted to integers from [0,255], shape (x,y,3)

    See Also
    --------
    _array_to_colour_1

    """
    if len(arr) == 0:
        return np.array([], dtype=np.uint8)
    return np.array(np.minimum(np.maximum(arr, 0), 255), dtype=np.uint8)


def _array_to_colour_1(arr):
    """Changes array of numbers into arrays of colour entries

    Parameters
    ----------
    arr: ndarray
        shape (x,y,3)

    Returns
    -------
    ndarray
        array entries converted to floats from [0,1]
    """
    if len(arr) == 0:
        return np.array([], dtype=np.float)
    return np.array(np.minimum(np.maximum(arr, 0), 1), dtype=float)


def _array_positive(arr):
    """Changes numbers smaller than arbitrary infinitesimal number to that number

    Parameters
    ----------
    arr : ndarray
        an array that may contain non-positive entries

    Returns
    -------
    ndarray
        an array of the same shape with strictly positive entries
    """
    return np.array(np.maximum(arr, _infzero), dtype=float)


class PixelOperations:
    def __init__(self, basis=None, background=None):
        """
        Class used to interact with basis and background (e.g. transforming pixels using this)

        Parameters
        ----------
        basis : array_like
            a list (or numpy array) with three-dimensional vectors. Can have 0, 1, 2 or 3 vectors
        background : array_like
            array with shape (3,) or list with three entries. Entries should be numbers from interval (0, 1]

        See Also
        --------
        PixelOperations.set_basis
        PixelOperations.set_background

        Notes
        -----
        It is equivalent to setting basis and background using setters
        """
        self.__basis, self.__basis_dim, self.__background, self.__basis_log_matrix = None, None, None, None

        self.set_basis(basis)
        self.set_background(background)

    def set_basis(self, basis):
        """Sets basis

        Parameters
        ----------
        basis : array_like
            a list (or numpy array) with three-dimensional vectors. Can have 0, 1, 2 or 3 vectors

        Raises
        ------
        BasisException
            Erroneous basis
        """

        if basis is None:
            basis = []

        basis = np.array(basis, dtype=float)

        if basis.shape not in [(0,), (1, 3), (2, 3), (3, 3)]:
            raise ex.BasisException("Basis has invalid dimensions, and was not set.")

        if not _entries_in_closed_interval(basis):
            raise ex.BasisException("Check components of the base vectors.")

        self.__basis = _array_positive(basis)
        self.__basis_dim = len(basis)

        if self.check_basis():
            self.__basis_log_matrix = np.transpose(-np.log(self.__basis))
            if np.linalg.matrix_rank(self.__basis_log_matrix) < self.get_basis_dim():
                raise ex.BasisException("Base vectors are (pseudo)linearly dependent.")

    def set_background(self, background=None):
        """Sets background

        Parameters
        ----------
        background : array_like
            array with shape (3,) or list with three entries. Entries should be numbers from interval (0, 1]

        Raises
        ------
        ValueError
            Erroneous background vector
        """

        if background is None:
            self.__background = _white1
            return

        background = np.array(background, dtype=float)

        if background.shape != (3,):
            raise ValueError("Check background vector shape.")

        if not _entries_in_half_closed_interval(background):
            raise ValueError("Check components of the background vector.")

        self.__background = _array_positive(_array_to_colour_1(background))

    def check_basis(self):
        """Checks if the basis is complete (has exactly two or three vectors)

        Returns
        -------
        bool
            True if it is, False otherwise
        """
        return self.__basis.shape in [(2, 3), (3, 3)]

    def get_basis_dim(self):
        """Returns number of the base vectors

        Returns
        -------
        int
            number of base vectors
        """
        return self.__basis_dim

    def get_basis(self):
        """Returns copy of the basis

        Returns
        -------
        ndarray
            array with basis. It can be empty or have shape (x,3) where x is 1, 2 or 3
        """
        return deepcopy(self.__basis)

    def get_background(self):
        """Returns copy of the background vector

        Returns
        -------
        ndarray
            background (numpy array)
        """
        return deepcopy(self.__background)

    def __transform_image2(self, image, mode):
        """Using basis with two vectors produce new images

        Parameters
        ----------
        image : ndarray
            shape (x,y,3)
        mode : array_like
            elements can be 0 (image generated from white light and two stains), 1 (white light and first stain),
            2 (white light and second stain) or -1 (remove both stains to obtain the rest)

        Returns
        -------
        list
            list of ndarrays with shape same as image according to mode

        Raises
        ------
        ValueError
            if image has wrong shape
        """
        r = np.array(image, dtype=float)

        dim1, dim2, dim3 = r.shape
        if dim3 != 3:
            raise ValueError("Basis has wrong shape.")

        v, u = self.__basis
        vf, uf = np.zeros_like(r), np.zeros_like(r)
        vf[:], uf[:] = v, u

        a, b = map(_array_positive, self.get_coef(r))
        af = np.repeat(a, 3).reshape(r.shape)
        bf = np.repeat(b, 3).reshape(r.shape)

        dec = []
        for i in mode:
            if i == 0:
                dec.append(_array_to_colour_255(_white255 * (vf ** af) * (uf ** bf)))
            elif i == 1:
                dec.append(_array_to_colour_255(_white255 * (vf ** af)))
            elif i == 2:
                dec.append(_array_to_colour_255(_white255 * (uf ** bf)))
            elif i == -1:
                dec.append(_array_to_colour_255(r * (vf ** -af) * (uf ** -bf)))

        return dec

    def __transform_image3(self, image, mode):
        """Using basis with three vectors produce new images

        Parameters
        ----------
        image : ndarray
            shape (x,y,3)
        mode : array_like
            elements can be 0 (image generated from white light and two stains), 1 (white light and first stain),
            2 (white light and second stain), 3 (white light and third stain)
            or -1 (remove all stains to obtain the rest)

        Returns
        -------
        list
            list of ndarrays with shape same as image according to mode

        See Also
        --------
        PixelOperations.__transform_image2
        """
        r = np.array(image, dtype=float)

        v, u, w = self.__basis
        vf, uf, wf = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
        vf[:], uf[:], wf[:] = v, u, w

        a, b, c = map(_array_positive, self.get_coef(r))
        af = np.repeat(a, 3).reshape(r.shape)
        bf = np.repeat(b, 3).reshape(r.shape)
        cf = np.repeat(c, 3).reshape(r.shape)

        dec = []
        for i in mode:
            if i == 0:
                dec.append(_array_to_colour_255(_white255 * (vf ** af) * (uf ** bf) * (wf ** cf)))
            elif i == 1:
                dec.append(_array_to_colour_255(_white255 * (vf ** af)))
            elif i == 2:
                dec.append(_array_to_colour_255(_white255 * (uf ** bf)))
            elif i == 3:
                dec.append(_array_to_colour_255(_white255 * (wf ** cf)))
            elif i == -1:
                dec.append(_array_to_colour_255(r * (vf ** -af) * (uf ** -bf) * (wf ** -cf)))

        return dec

    def transform_image(self, image, mode=None):
        """Transforms given image array and gives output accordingly to iterable mode

        Parameters
        ----------
        image : ndarray
            shape (x,y,3)
        mode : array_like
            elements can be 0 (image generated from white light and two stains), 1 (white light and first stain),
            2 (white light and second stain), 3 (white light and third stain. Note that this works only if a basis
            with three vectors is used) or -1 (remove all stains to obtain the rest)

        Returns
        -------
        list
            list of ndarrays with shape same as image according to mode. Type is np.uint8.

        Raises
        ------
        BasisException
            No basis has been set
        See Also
        --------
        PixelOperations.__transform_image2
        PixelOperations.__transform_image3
        """

        if self.__basis_dim == 2:
            return self.__transform_image2(image, [1, 2] if mode is None else mode)
        elif self.__basis_dim == 3:
            return self.__transform_image3(image, [1, 2, 3] if mode is None else mode)
        else:
            raise ex.BasisException("No proper basis set.")

    def __get_coef2(self, pixel):
        """Finds exponentials in which stains are present. Returns non-negative values"""
        r = np.array(pixel, dtype=float)
        r = _array_positive(_array_to_colour_1(r/255.))
        r = r/self.__background

        sol = aux.find_vals(self.__basis_log_matrix, np.log(r))
        sol = np.maximum(0, sol)
        return sol

    def __get_coef3(self, pixel):
        """Finds exponentials in which stains are present. Returns non-negative values"""
        r = np.array(pixel, dtype=float)
        r = _array_positive(_array_to_colour_1(r/255.))
        r /= self.__background

        sol = np.linalg.solve(self.__basis_log_matrix, -np.log(r))
        sol = np.maximum(0, sol)
        return sol

    def get_coef(self, image):
        """For a given image returns deconvolution coefficient field

        Parameters
        ----------
        image : numpy array
            array representing image, shape (x,y,3) and entries [0,255]

        Returns
        -------
        list
            length of the list is number of vectors in the basis. Each entry is numpy array with shape (x,y)
            representing field of exponent coefficients
        Raises
        ------
        BasisException
            No basis has been set
        ValueError
            Image channel number not supported.
        """
        if image.shape[-1] != 3:
            raise ValueError("Pixel dimensionality is wrong. Maybe it has an alpha channel?")

        if self.get_basis_dim() == 2:
            fv = np.vectorize(self.__get_coef2, signature='(n)->(k)')
        elif self.get_basis_dim() == 3:
            fv = np.vectorize(self.__get_coef3, signature='(n)->(k)')
        else:
            raise ex.BasisException("Basis of dimension 2 or 3 has not been set.")

        return np.array(fv(image)).transpose((2, 0, 1))
