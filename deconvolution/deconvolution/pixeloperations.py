import numpy as np
import auxlib as aux

_infzero = 0.00001
_white255 = np.array([255, 255, 255], dtype=float)
_white1 = np.array([1, 1, 1], dtype=float)

def _proper_vector_check(vect):
    """
    Checks if all list components are in [0,1]
    :param vect (list or list of lists):
    :return: true/false (bool)
    """
    return not (np.amax(vect) > 1 or np.amin(vect) < 0)


def _array_to_colour_255(arr):
    """
    Changes array of floats into arrays of colour entries
    :param arr: numpy array of shape (x,y,3)
    :return: array entries converted to integers from [0,255]
    """
    return np.array(np.minimum(np.maximum(arr, 0), 255), dtype=int)


def _array_to_colour_1(arr):
    """
    Changes array of floats into arrays of colour entries
    :param arr: numpy array of shape (x,y,3)
    :return: array entries converted to floats from [0,1]
    """
    return np.array(np.minimum(np.maximum(arr, 0), 1), dtype=float)


def _array_positive(arr):
    """
    Changes zeros to inf zeros
    :param arr:
    :return: numpy array with the same shape
    """
    return np.array(np.maximum(arr, _infzero), dtype=float)


class PixelOperations:
    def __init__(self, basis=None, background=None):
        """
        Initialising function
        :param basis:
        :param background:
        """
        self.__basis, self.__basis_dim, self.__background, self.__basis_log_matrix = None, None, None, None

        self.set_basis(basis)
        self.set_background(background)

    def set_basis(self, basis):
        """
        Sets basis
        :param basis:
        """
        if basis is None:
            self.__basis = []
            self.__basis_dim = 0

        elif not _proper_vector_check(basis):
            raise Exception("Check components of the base vectors")

        else:
            self.__basis_dim = len(basis)

        self.__basis = _array_positive(basis)

        if self.check_basis():
            self.__basis_log_matrix = np.transpose(-np.log(self.__basis))
            if np.linalg.matrix_rank(self.__basis_log_matrix) < self.get_basis_dim():
                raise Exception("Base vectors are (pseudo)linearly dependent")

    def set_background(self, background=None):
        """
        Sets background
        :param background:
        """
        if background is None:
            self.__background = _white1
            return

        if not _proper_vector_check(background):
            raise Exception("Check components of the background vector")

        self.__background = _array_positive(_array_to_colour_1(background))

    def check_basis(self):
        """
        Checks if the basis has two or three vectors
        :return: bool
        """
        return self.__basis.shape in [(2, 3), (3, 3)]

    def get_basis_dim(self):
        """
        Returns number of base vectors
        :return: uint
        """
        return self.__basis_dim

    def get_basis(self):
        """
        Returns basis
        :return: basis
        """
        return self.__basis

    def get_background(self):
        """
        Returns background vector
        :return: background (numpy array)
        """
        return self.__background

    def __transform_image2(self, image, mode):
        r = np.array(image, dtype=float)

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
            elif i == 3:
                dec.append(_array_to_colour_255(r * (vf ** -af) * (uf ** -bf)))

        return dec

    def __transform_image3(self, image, mode):
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
            elif i == 4:
                dec.append(_array_to_colour_255(r * (vf ** -af) * (uf ** -bf) * (wf ** -cf)))

        return dec

    def transform_image(self, image, mode=None):
        """
        Transforms given image array and gives output accordingly to iterable mode
        :param image:
        :param mode:
        :return: list of images with dimension of mode
        """
        if self.__basis_dim == 2:
            return self.__transform_image2(image, [1, 2] if mode is None else mode)
        elif self.__basis_dim == 3:
            return self.__transform_image3(image, [1, 2, 3] if mode is None else mode)
        else:
            raise Exception("You can't transform image until you have a proper basis")

    def __get_coef2(self, pixel):
        r = np.array(pixel, dtype=float)
        r = _array_positive(_array_to_colour_1(r/255.))
        r = r/self.__background

        return aux.find_vals(self.__basis_log_matrix, np.log(r))

    def __get_coef3(self, pixel):
        r = np.array(pixel, dtype=float)
        r = _array_positive(_array_to_colour_1(r/255.))
        r /= self.__background

        sol = np.linalg.solve(self.__basis_log_matrix, -np.log(r))
        sol = np.maximum(0, sol)
        return sol

    def get_coef(self, image):
        """
        For a given image returns deconvolution coefficient field
        :param image:
        :return: list of numpy arrays; length of the basis dimension
        """
        if image.shape[-1] != 3:
            raise Exception("Image is corrupted - pixel dimensionality is wrong")

        if self.get_basis_dim() == 2:
            fv = np.vectorize(self.__get_coef2, signature='(n)->(k)')
        elif self.get_basis_dim() == 3:
            fv = np.vectorize(self.__get_coef3, signature='(n)->(k)')
        else:
            raise Exception("Basis of dimension 2 or 3 has not been set yet")

        return np.array(fv(image)).transpose((2, 0, 1))
