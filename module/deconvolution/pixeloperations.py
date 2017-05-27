import numpy as np
import auxlib as aux


def _proper_vector_check(vect):
    """
    Checks if all list components are in [0,1]
    :param vect (iterable):
    :return: true/false (bool)
    """
    return not (max(vect) > 1 or min(vect) < 0)

_white = np.array([255, 255, 255], dtype=float)


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
        for vect in basis:
            if not _proper_vector_check(vect):
                raise Exception("Check components of the base vectors")
        self.__basis = basis
        self.__basis_dim = len(basis)

        for i in range(basis.shape[0]):
            for j in range(basis.shape[1]):
                basis[i][j] = aux.positive(basis[i][j])

        if basis.shape in [(2, 3), (3, 3)]:
            self.__basis_log_matrix = np.transpose(-np.log(basis))

    def set_background(self, background=None):
        """
        Sets background
        :param background:
        """
        if not _proper_vector_check(background):
            raise Exception("Check components of the background vector")

        self.__background = np.array(map(aux.positive, map(aux.to_colour_1, background)))

    def check_background(self):
        """
        Checks if there is background
        :return: bool
        """
        return self.__background is not None

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

    def __transform_pixel2(self, pixel, mode):
        r = np.array(pixel, dtype=float)
        v, u = np.array(self.__basis[0], dtype=float), np.array(self.__basis[1], dtype=float)
        a, b = map(aux.positive, self.get_coef(r))

        dec = []

        for i in mode:
            if i == 0:
                dec.append(map(aux.to_colour_255, _white * (v ** a) * (u ** b)))
            if i == 1:
                dec.append(map(aux.to_colour_255, _white * (v ** a)))
            if i == 2:
                dec.append(map(aux.to_colour_255, _white * (u ** b)))
            if i == 3:
                dec.append(map(aux.to_colour_255, r * (v ** -a) * (u ** -b)))

        return dec

    def __transform_pixel3(self, pixel, mode):  # for 3 subs, used in out_images
        r = np.array(pixel, dtype=float)
        v, u = np.array(self.__basis[0], dtype=float), np.array(self.__basis[1], dtype=float)
        w = np.array(self.__basis[2], dtype=float)

        a, b, c = map(aux.positive, self.get_coef(r))

        dec = []

        for i in mode:
            if i == 0:
                dec.append(map(aux.to_colour_255, _white * (v ** a) * (u ** b) * (w ** c)))
            if i == 1:
                dec.append(map(aux.to_colour_255, _white * (v ** a)))
            if i == 2:
                dec.append(map(aux.to_colour_255, _white * (u ** b)))
            if i == 3:
                dec.append(map(aux.to_colour_255, _white * (w ** c)))
            if i == 4:
                dec.append(map(aux.to_colour_255, r * (v ** -a) * (u ** -b) * (w ** -c)))

        return dec

    def transform_pixel(self, pixel, mode=None):
        """
        Transforms given pixel and gives output accordingly to iterable mode
        :param pixel:
        :param mode:
        :return: list with dimension of mode
        """
        if self.__basis_dim == 2:
            return self.__transform_pixel2(pixel, [1, 2] if mode is None else mode)
        elif self.__basis_dim == 3:
            return self.__transform_pixel3(pixel, [1, 2, 3] if mode is None else mode)
        else:
            raise Exception("You can't transform pixels until you set basis")

    def __get_coef2(self, pixel):
        r = np.array(pixel, dtype=float)
        r = np.array(map(aux.to_colour_1, r/255.), dtype=float)
        r = np.array(map(aux.positive, r), dtype=float)

        if self.check_background():
            return aux.find_vals(self.__basis_log_matrix, np.log(r / self.__background))
        return aux.find_vals(self.__basis_log_matrix, np.log(r))

    def __get_coef3(self, pixel):
        r = np.array(pixel, dtype=float)
        r = np.array(map(aux.to_colour_1, r/255.), dtype=float)
        r = np.array(map(aux.positive, r), dtype=float)

        if self.check_background():
            r /= self.__background

        r = np.array(map(aux.positive, r), dtype=float)
        r = -np.array(map(np.log, r), dtype=float)

        if np.linalg.matrix_rank(self.__basis_log_matrix) < 3:
            raise Exception("Error: Basis vectors are (pseudo)linearly dependent")

        sol = np.linalg.solve(self.__basis_log_matrix, r)

        for i in range(len(sol)):
            sol[i] = max(0, sol[i])
        return sol

    def get_coef(self, pixel):
        """
        For a given pixel returns deconvolution
        :param pixel:
        :return: coefficient list with length of the basis dimension
        """
        if pixel.shape != (3,):
            raise Exception("Pixel is corrupted - dimensionality is wrong")
        if not self.check_basis():
            raise Exception("Basis has not been set yet")
        if self.get_basis_dim() == 2:
            return self.__get_coef2(pixel)
        elif self.get_basis_dim() == 3:
            return self.__get_coef3(pixel)
