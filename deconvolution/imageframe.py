"""
imageframe provides ImageFrame class that is a backend of deconvolution algorithm
"""

import numpy as np
#from multiprocessing import Pool
from PIL import Image
import deconvolution.auxlib as aux
from deconvolution.pixeloperations import WrongBasisError
from functools import reduce


class ImageFrame:
    def __init__(self, image=None, threads=1, verbose=False):
        """Class that can be used to deconvolve an image

        Parameters
        ----------
        image : PIL Image
            image to deconvolve
        threads : int,
            number of threads to use
        verbose : bool
            whether to print internal processing information to std out (e.g. for debugging purposes or process control)
        """
        self.__verbose = False
        if verbose:
            self.__verbose = True

        self.__image = None
        if image is not None:
            self.set_image(image)

        self.__threads, self.__inertia_matrix = threads, None

    def set_image(self, image):
        """Sets new image to deconvolve.
        Parameters
        ----------
        image : PIL Image
            image to deconvolve
        """
        self.__image = np.asarray(image).copy()

    def __source_set(self):
        """Check whether an image has been set

        Returns
        -------
        bool
            True if it has been set, False otherwise
        """
        return self.__image is not None

    def __source_sampled(self):
        """Check whether the inertia matrix has been calculated

        Returns
        -------
        bool
            True if the inertia matrix calculation has been finished
        """
        return self.__inertia_matrix is not None

    def sample_source(self, pixel_operations, sample_density=5):
        """Creates inertia matrix used for finding optimal bases.

        Parameters
        ----------
        pixel_operations : PixelOperations
            an object used for manipulation with basis and background
        sample_density : int
            precision of sampling

        See Also
        --------
        PixelOperations
        """
        if not self.__source_set:
            raise Exception("No source set")

        if self.__verbose:
            print("Sampling source...")

        step = (2 ** (8 - sample_density))
        rgb_sample = np.zeros(shape=(2 ** sample_density, 2 ** sample_density, 2 ** sample_density))

        def sample_pixel(pix):
            rgb = np.array(pix, dtype=float) / pixel_operations.get_background()
            rgb = np.array(map(aux.to_colour_255, rgb))
            rgb = 1. * rgb / step
            rgb_sample[tuple(map(int, rgb.tolist()))] += 1

        def sample_row(row):
            map(sample_pixel, row)

        # bad code
        map(sample_row, self.__image)

        w0 = np.ones(2 ** sample_density)
        w1 = np.log(2) * sample_density - np.log(np.arange(2 ** sample_density) + 0.5)
        w2 = w1 * w1

        def w_vec(i):
            if i == 0:
                return w0
            elif i == 1:
                return w1
            elif i == 2:
                return w2

        def w_mat(i, j):
            return reduce(np.multiply.outer,
                          [w_vec((i == 0) + (j == 0)), w_vec((i == 1) + (j == 1)), w_vec((i == 2) + (j == 2))]
                          )

        self.__inertia_matrix = [[np.sum(w_mat(i, j) * rgb_sample) for i in range(3)] for j in range(3)]

        if self.__verbose:
            print("Done.")

    def __find_substance_two(self, pixel_operations):
        """Used when two substances are needed, and none of them is known.

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for setting new basis

        See Also
        --------
        PixelOperations
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
        """Used when one additional substance is needed, and one is already known.

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for setting new basis and getting known substance

        See Also
        --------
        PixelOperations
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
        """Checks dimensionality and completes basis accordingly. Performs sampling automatically if needed.

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for interactions with basis

        See Also
        --------
        PixelOperations

        Raises
        ------
        Exception
            source has not been set yet (note - this will be replaced by another type of error)
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
        """Tries to minimize mutual dependence of deconvolved substance density fields

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for setting new basis and getting known substance
        belligerency : float
            high values lead to greater contrast in stains

        See Also
        --------
        PixelOperations

        Raises
        ------
        WrongBasisError
            basis has wrong number of vectors
        Exception
            second substance is extremely negative compared to the `belligerency`. It should never happen
        """
        if pixel_operations.get_basis_dim() != 2:
            raise WrongBasisError("Exactly two element basis needed for resolve_dependencies")

        # collecting data
        if self.__verbose:
            print("Decomposition info:\n")

        surf = len(self.__image[0]) * len(self.__image[0])

        a, b = pixel_operations.get_coef(self.__image)
        a_neg_sum = np.sum(np.minimum(a, 0))
        a_pos_sum = np.sum(np.maximum(a, 0))
        a_mean = a_pos_sum/surf

        b_neg_sum = np.sum(np.minimum(b, 0))
        b_pos_sum = np.sum(np.maximum(b, 0))
        b_mean = b_pos_sum/surf

        if self.__verbose:
            print("First substance:")
            print("Negativity: {} Mean value: {}".format(a_neg_sum/aux.positive(a_pos_sum), a_mean))
            print("Second substance:")
            print("Negativity: {} Mean value: {}".format(b_neg_sum/aux.positive(b_pos_sum), b_mean))

        min_x = belligerency * a_mean
        min_y = belligerency * b_mean

        if min_y <= 0:
            raise Exception("Something went horribly wrong. Feel free to bash developers")

        def safe_div(x, y):
            if x > min_x and y > min_y:
                return x/y
            else:
                return 1e20

        safe_vec_div = np.vectorize(safe_div)

        # resolving
        k1 = np.min(safe_vec_div(a, b))
        k2 = np.min(safe_vec_div(b, a))

        if self.__verbose:
            print("Mutual dependencies k1, k2 = {}, {}".format(k1, k2))

        basis = pixel_operations.get_basis()
        pixel_operations.set_basis(
            [[basis[0][0]*(basis[1][0]**k2), basis[0][1]*(basis[1][1]**k2), basis[0][2]*(basis[1][2]**k2)],
             [basis[1][0]*(basis[0][0]**k1), basis[1][1]*(basis[0][1]**k1), basis[1][2]*(basis[0][2]**k1)]]
        )

        if self.__verbose:
            print("Corrected substances:")
            print(pixel_operations.get_basis())

    def out_scalars(self, pixel_operations=None):
        """Get scalar density fields of substances.

        Returns
        -------
        list
            list of numpy arrays (length is the dimensionality of basis), each with exponent field of coefficient
        """
        return pixel_operations.get_coef(self.__image)

    def out_images(self, pixel_operations=None, mode=None):
        """Get list of deconvolved images.

        Parameters
        ----------
        pixel_operations : PixelOperations
            object for interaction with basis
        mode : array_like
            if list contains:
                0 - image generated from white light and all stains
                1 - white light and first stain
                2 - white light and second stain
                (3) - white light and third stain (only if basis with three vectors has been set)

                3 (4) - use image and remove both stains to obtain the rest (4 if three vectors has been set)

        Returns
        -------
        list
            list of PIL Images

        Raises
        ------
        WrongBasisError
            basis needs to have at least two vectors
        Exception
            you need image to deconvolve
        """
        if pixel_operations.get_basis_dim() < 2:
            raise WrongBasisError("At least two elements in basis needed")

        if not self.__source_set():
            raise Exception("Error: source has to be set first")

        if self.__verbose:
            print("Returning deconvolved images...")

        if pixel_operations.get_basis_dim() == 2:
            if mode is None:
                mode = [1, 2]

        if pixel_operations.get_basis_dim() == 3:
            if mode is None:
                mode = [1, 2, 3]

        out_tmp = pixel_operations.transform_image(self.__image, mode=mode)
        return [Image.fromarray(out_tmp[i]) for i in range(len(mode))]
