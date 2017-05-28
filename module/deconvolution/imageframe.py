import numpy as np
from multiprocessing import Pool
from PIL import Image
import auxlib as aux


def _is0(x):
    if x == 0:
        return 1
    else:
        return 0


def _is1(x):
    if x == 1:
        return 1
    else:
        return 0


def _is2(x):
    if x == 2:
        return 1
    else:
        return 0


class ImageFrame:
    def __init__(self, image=None, threads=1, verbose=True):
        """
        Initialising function
        :param image: PIL Image, image to deconvolve
        :param threads: int, number of threads to use
        :param verbose: bool, default True
        """
        self.__verbose = True
        if not verbose:
            self.__verbose = False

        self.__image, self.__threads, self.__inertia_matrix = np.asarray(image).copy(), threads, None

    def set_image(self, image):
        """
        Sets new image for deconvolution.
        :param image: PIL Image
        """
        self.__image = np.asarray(image).copy()

    def __source_set(self):
        """
        Source set?
        :return: bool
        """
        return self.__image is not None

    def __source_sampled(self):
        """
        Source sampled?
        :return: bool
        """
        return self.__inertia_matrix is not None

    def sample_source(self, pixel_operations, sample_density=5):
        """
        Creates __inertia_matrix used for finding optimal bases.
        :param pixel_operations: used for background info
        :param sample_density: precision of sampling
        """
        if not self.__source_set:
            raise Exception("No source set")

        step = (2 ** (8 - sample_density))
        rgb_sample = np.zeros(shape=(2 ** sample_density, 2 ** sample_density, 2 ** sample_density))

        if pixel_operations.check_background:
            def sample_pixel(pix):
                rgb = np.array(pix, dtype=float) / pixel_operations.background
                rgb = np.array(map(aux.to_colour_255, rgb))
                rgb = 1. * rgb / step
                rgb_sample[tuple(map(int, rgb.tolist()))] += 1
        else:
            def sample_pixel(pix):
                rgb = np.array(pix, dtype=float) / step
                rgb_sample[tuple(map(int, rgb.tolist()))] += 1

        def sample_row(row):
            map(sample_pixel, row)

        Pool(self.__threads).map(sample_row, self.__image)

        w0 = np.ones(2 ** sample_density)
        w1 = np.log(np.identity(2 ** sample_density) + 0.5) - np.log(2) * sample_density
        w2 = w1 * w1

        def w_vec(i):
            if i == 0:
                return w0
            if i == 1:
                return w1
            if i == 2:
                return w2

        def w_mat(i, j):
            return np.outer(w_vec(_is0(i) + _is0(j)), w_vec(_is1(i) + _is1(j)), w_vec(_is2(i) + _is2(j)))

        self.__inertia_matrix = np.fromfunction(lambda i, j: np.sum(w_mat(i, j) * rgb_sample), (3, 3))

    def __find_substance_two(self, pixel_operations):
        """
        Used when TWO substances are needed, and NONE is known.
        :param pixel_operations: used for setting new basis 
        """
        if self.__verbose:
            print("Searching for best fitting substances...")

        eig = np.linalg.eig(self.__inertia_matrix)

        minimum = eig[0][0]
        index = 0
        for i in range(1, 3):
            if eig[0][i] < minimum:
                minimum = eig[0][i]
                index = i

        n = [eig[1][0][index], eig[1][1][index], eig[1][2][index]]
        n = n / np.linalg.norm(n)
        n = aux.get_physical_normal(n)

        basis = aux.get_basis_from_normal(n)

        subs = np.ones(shape=(2, 3))
        for i in range(3):
            subs[0][i] = np.exp(-basis[0][i])
            subs[1][i] = np.exp(-basis[1][i])

        pixel_operations.set_basis(subs)

        if self.__verbose:
            print("Finished search.")
            print("Found substances:")
            print(subs)

    def __find_substance_one(self, pixel_operations):
        """
        Used when ONE additional substance is needed, and ONE is already known.
        :param pixel_operations: used for setting new basis and getting known substance
        """
        if self.__verbose:
            print("Searching for second best fitting substance...")

        if pixel_operations.get_basis_dim() == 0:  # Basis requires AT LEAST 1 vector in basis
            raise Exception("Empty basis")

        basis = pixel_operations.get_basis()
        v = np.array([-np.log(basis[0][0]), -np.log(basis[0][1]), -np.log(basis[0][2])])
        v = v / np.linalg.norm(v)

        rot = aux.orthonormal_rotation(v)

        rot_inert = np.dot(np.dot(rot, self.__inertia_matrix), np.transpose(rot))

        n = np.dot(np.transpose(rot), aux.find_vector(rot_inert))
        n = aux.get_physical_normal(n)

        basis = aux.get_basis_from_normal(n)
        u = basis[0] if np.dot(v, basis[0]) < np.dot(v, basis[1]) else basis[1]

        basis = [v, u]

        subs = np.ones(shape=(2, 3))
        for i in range(3):
            subs[0][i] = np.exp(-basis[0][i])
            subs[1][i] = np.exp(-basis[1][i])

        pixel_operations.set_basis(subs)

        if self.__verbose:
            print("Finished search.")
            print("Found substances:")
            print(subs)

    def complete_basis(self, pixel_operations):
        """
        Checks dimensionality and completes basis accordingly.
        Performs sampling automatically if needed.
        :param pixel_operations: used for interaction with basis
        """
        if not self.__source_set():
            raise Exception("Set source first")

        if not self.__source_sampled():
            self.sample_source(pixel_operations)

        dim = pixel_operations.get_basis_dim()
        if dim == 0:
            self.__find_substance_two(pixel_operations)
        elif dim == 1:
            self.__find_substance_one(pixel_operations)
        elif self.__verbose:
            print("Basis already complete")

    def resolve_dependencies(self, pixel_operations=None, belligerency=0.3):
        """
        Tries to minimize mutual dependence of deconvolved substance density fields
        :param pixel_operations: used for interaction with basis
        :param belligerency: aggressiveness of corrections
        """
        if pixel_operations.get_basis_dim() != 2:
            raise Exception("Exactly two element basis needed for resolve_dependencies")

        # collecting data
        if self.__verbose:
            print("Decomposition info:\n")

        surf = len(self.__image[0]) * len(self.__image[0])

        a, b = pixel_operations.get_coef(self.__image)
        a_neg_sum = np.sum(np.maximum(a, 0))
        a_pos_sum = np.sum(np.minimum(a, 0))
        a_mean = a_pos_sum/surf

        b_neg_sum = np.sum(np.maximum(b, 0))
        b_pos_sum = np.sum(np.minimum(b, 0))
        b_mean = b_pos_sum/surf

        if self.__verbose:
            print("First substance:")
            print("Negativity: ", a_neg_sum/aux.positive(a_pos_sum), "Mean value: ", a_mean)
            print("Second substance:")
            print("Negativity: ", b_neg_sum/aux.positive(b_pos_sum), "Mean value: ", b_mean, "\n")

        def safe_div(x, y):
            if x > belligerency * a_mean and y > belligerency * b_mean:
                return x/y
            else:
                return 1e20

        safe_vec_div = np.vectorize(safe_div)

        # resolving
        k1 = np.min(safe_vec_div(a, b))
        k2 = np.min(safe_vec_div(b, a))

        if self.__verbose:
            print("Mutual dependencies k1, k2 =", k1, ", ", k2)

        basis = pixel_operations.get_basis()
        pixel_operations.set_basis(
            [[basis[0][0]*(basis[1][0]**k2), basis[0][1]*(basis[1][1]**k2), basis[0][2]*(basis[1][2]**k2)],
             [basis[1][0]*(basis[0][0]**k1), basis[1][1]*(basis[0][1]**k1), basis[1][2]*(basis[0][2]**k1)]]
        )

        if self.__verbose:
            print("Corrected substances:")
            print(pixel_operations.get_basis())

    def out_scalars(self, pixel_operations=None):
        """
        Get scalar density fields of substances.
        :return: list of scalar fields (numpy arrays) 
        """
        return pixel_operations.get_coef(self.__image)

    def out_images(self, pixel_operations=None, mode=None):
        """
        Get list of deconvolved images.
        :param pixel_operations: basis interaction
        :param mode: which images to return (see ref. table in documentation TODO)
        :return: list of PIL Images
        """
        if pixel_operations.get_basis_dim() < 2:
            raise Exception("At least two elements in basis needed")

        if self.__source_set():
            print("Error: source has to be set first")

        if self.__verbose:
            print("Returning deconvolved images...")

        if pixel_operations.get_basis_dim() == 2:
            if mode is None:
                mode = [1, 2]

        if pixel_operations.get_basis_dim() == 3:
            if mode is None:
                mode = [1, 2, 3]

        out_tmp = pixel_operations.transform_pixel(self.__image, mode)
        return [Image.fromarray(out_tmp[i]) for i in range(len(mode))]
