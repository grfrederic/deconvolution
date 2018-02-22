"""
imageframe provides ImageFrame class that is a backend of deconvolution algorithm
"""

import numpy as np
# from multiprocessing import Pool

from PIL import Image
import deconvolution.auxlib as aux
import deconvolution.exceptions as ex
from functools import reduce


class ImageFrame:
    def __init__(self, image=None, threads=1, verbose=False, sample_density=5):
        """Class that can be used to deconvolve an image

        Parameters
        ----------
        image : PIL Image
            image to deconvolve
        threads : int,
            number of threads to use
        verbose : bool
            whether to print internal processing information to std out (e.g. for debugging purposes or process control)
        sample_density : int
            precision of sampling
        """
        self._verbose = False
        self.set_verbose(verbose)
        self._sample_density = 0
        self.sample_density = sample_density

        self._image = None
        if image is not None:
            self.set_image(image)

        self._threads, self._inertia_matrix = threads, None

    @property
    def sample_density(self):
        return self._sample_density

    @sample_density.setter
    def sample_density(self, value):
        if not isinstance(value, int):
            raise TypeError("sample_density must be int")
        if value not in range(2, 9):
            raise ValueError("sample density must be in interval [2,8]")
        self._sample_density = value

    def set_verbose(self, verbose):
        """Change verbosity.

        Parameters
        ----------
        verbose : bool
            set to True prints to the std output internal actions
        """
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError("Variable verbose has to be bool.")

    def get_verbose(self):
        """Returns verbosity.

        Returns
        -------
        bool
            True if verbosity is turned on, False otherwise
        """
        return self._verbose

    def get_threads(self):
        """Returns number of threads to be used.

        Returns
        -------
        int
            number of threads used in deconvolution
        """
        return self._threads

    def set_image(self, image):
        """Sets new image to deconvolve.
        Parameters
        ----------
        image : PIL Image
            image to deconvolve

        Raises
        ----------
        ValueError
            Image has to be a PIL Image
        """
        if isinstance(image, Image.Image):
            self._image = np.asarray(image).copy()
        else:
            raise ValueError("image has to be a PIL Image.")

    def get_image(self):
        """Returns copy of the stored image.

        Returns
        -------
        ndarray
            copy of the array representing image
        """
        return np.copy(self._image)

    def get_inertia_matrix(self):
        """Returns copy of the calculated inertia matrix.

        Returns
        -------
        ndarray
            copy of the array representing the inertia matrix.
        """
        if not self._source_sampled():
            raise ex.ImageException("Source not sampled")

        return np.copy(self._inertia_matrix)

    def _source_set(self):
        """Check whether an image has been set

        Returns
        -------
        bool
            True if it has been set, False otherwise
        """
        return self._image is not None

    def _source_sampled(self):
        """Check whether the inertia matrix has been calculated

        Returns
        -------
        bool
            True if the inertia matrix calculation has been finished
        """
        return self._inertia_matrix is not None

    def sample_source(self, pixel_operations):
        """Creates inertia matrix used for finding optimal bases.

        Parameters
        ----------
        pixel_operations : PixelOperations
            an object used for manipulation with basis and background

        Raises
        --------
        ImageException
            No image has been set
        See Also
        --------
        PixelOperations
        """
        if not self._source_set():
            raise ex.ImageException("No source set")

        if self._verbose:
            print("Sampling source...")

        step = (2 ** (8 - self.sample_density))
        rgb_sample = np.zeros(shape=(2 ** self.sample_density, 2 ** self.sample_density, 2 ** self.sample_density))

        def sample_pixel(pix):
            rgb = np.array(pix, dtype=float) / pixel_operations.get_background()
            rgb = np.array([aux.to_colour_255(c) for c in rgb])
            rgb = 1. * rgb / step
            rgb_sample[tuple(map(int, rgb.tolist()))] += 1

        def sample_row(row):
            [sample_pixel(px) for px in row]

        [sample_row(row) for row in self._image]

        w0 = np.ones(2 ** self.sample_density)
        w1 = np.log((0.1 + np.arange(2 ** self.sample_density)) /
                    (0.1 + 2 ** self.sample_density))
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

        self._inertia_matrix = [[np.sum(w_mat(i, j) * rgb_sample) for i in range(3)] for j in range(3)]

        if self._verbose:
            print("Done.")

    def _find_substance_two(self, pixel_operations):
        """Used when two substances are needed, and none of them is known.

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for setting new basis

        See Also
        --------
        PixelOperations
        """
        if self._verbose:
            print("Searching for best fitting substances...")

        eig = np.linalg.eig(self._inertia_matrix)

        minimum = eig[0][0]
        index = 0
        for i in range(1, 3):
            if eig[0][i] < minimum:
                minimum = eig[0][i]
                index = i

        n = [eig[1][0][index], eig[1][1][index], eig[1][2][index]]
        n = aux.get_physical_normal(n)

        basis = aux.get_basis_from_normal(n)

        subs = np.ones(shape=(2, 3))
        for i in range(3):
            subs[0][i] = np.exp(-basis[0][i])
            subs[1][i] = np.exp(-basis[1][i])

        pixel_operations.set_basis(subs)

        if self._verbose:
            print("Finished search.")
            print("Found substances:")
            print(subs)

    def _find_substance_one(self, pixel_operations):
        """Used when one additional substance is needed, and one is already known.

        Parameters
        ----------
        pixel_operations : PixelOperations
            used for setting new basis and getting known substance

        Raises
        ----------
        BasisException
            Empty basis

        See Also
        --------
        PixelOperations
        """
        if self._verbose:
            print("Searching for second best fitting substance...")

        if pixel_operations.get_basis_dim() == 0:  # Basis requires AT LEAST 1 vector in basis
            raise ex.BasisException("Empty basis")

        basis = pixel_operations.get_basis()
        v = np.array([-np.log(basis[0][0]), -np.log(basis[0][1]), -np.log(basis[0][2])])
        v = v / np.linalg.norm(v)

        rot = aux.orthonormal_rotation(v)

        rot_inert = np.dot(np.dot(rot, self._inertia_matrix), np.transpose(rot))

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

        if self._verbose:
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
        ImageException
            image has not been set yet
        """

        if not self._source_set():
            raise ex.ImageException("Set source first")

        if not self._source_sampled():
            self.sample_source(pixel_operations)

        dim = pixel_operations.get_basis_dim()
        if dim == 0:
            self._find_substance_two(pixel_operations)
        elif dim == 1:
            self._find_substance_one(pixel_operations)
        elif self._verbose:
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
        BasisException
            basis has wrong number of vectors
        Exception
            second substance is extremely negative compared to the `belligerency`. This should never happen
        """
        if pixel_operations.get_basis_dim() != 2:
            raise ex.BasisException("Exactly two element basis needed for resolve_dependencies")

        if belligerency < 0:
            raise ValueError("Belligerency needs to be nonnegative.")

        # collecting data
        if self._verbose:
            print("Decomposition info:\n")

        surf = len(self._image[0]) * len(self._image[0])

        a, b = pixel_operations.get_coef(self._image)
        a_neg_sum = np.sum(np.minimum(a, 0))
        a_pos_sum = np.sum(np.maximum(a, 0))
        a_mean = a_pos_sum/surf

        b_neg_sum = np.sum(np.minimum(b, 0))
        b_pos_sum = np.sum(np.maximum(b, 0))
        b_mean = b_pos_sum/surf

        if self._verbose:
            print("First substance:")
            print("Negativity: {} Mean value: {}".format(a_neg_sum/aux.positive(a_pos_sum), a_mean))
            print("Second substance:")
            print("Negativity: {} Mean value: {}".format(b_neg_sum/aux.positive(b_pos_sum), b_mean))

        min_x = belligerency * a_mean
        min_y = belligerency * b_mean

        def safe_div(x, y):
            if x > min_x and y > min_y:
                return x/y
            else:
                return 1e20

        safe_vec_div = np.vectorize(safe_div)

        # resolving
        k1 = np.min(safe_vec_div(a, b))
        k2 = np.min(safe_vec_div(b, a))

        if k1 == 1e20 or k2 ==1e20:
            if self._verbose:
                print("Unable to resolve.")
            return


        if self._verbose:
            print("Mutual dependencies k1, k2 = {}, {}".format(k1, k2))

        basis = pixel_operations.get_basis()
        pixel_operations.set_basis(
            [[basis[0][0]*(basis[1][0]**k2), basis[0][1]*(basis[1][1]**k2), basis[0][2]*(basis[1][2]**k2)],
             [basis[1][0]*(basis[0][0]**k1), basis[1][1]*(basis[0][1]**k1), basis[1][2]*(basis[0][2]**k1)]]
        )

        if self._verbose:
            print("Corrected substances:")
            print(pixel_operations.get_basis())

    def out_scalars(self, pixel_operations=None):
        """Get scalar density fields of substances.

        Returns
        -------
        list
            list of numpy arrays (length is the dimensionality of basis), each with exponent field of coefficient
        """
        return pixel_operations.get_coef(self._image)

    def out_images(self, pixel_operations=None, mode=None):
        """Get list of deconvolved images.

        Parameters
        ----------
        pixel_operations : PixelOperations
            object for interaction with basis
        mode : array_like
            elements can be 0 (image generated from white light and two stains), 1 (white light and first stain),
            2 (white light and second stain), 3 (white light and third stain. Note that this works only if a basis
            with three vectors is used) or -1 (remove all stains to obtain the rest)

        Returns
        -------
        list
            list of PIL Images

        Raises
        ------
        BasisException
            basis needs to have at least two vectors
        ImageException
            you need image to deconvolve
        """
        if not self._source_set():
            raise ex.ImageException("Error: source has to be set first")

        if self._verbose:
            print("Returning deconvolved images...")

        out_tmp = pixel_operations.transform_image(self._image, mode=mode)
        return [Image.fromarray(out_tmp[i]) for i in range(len(mode))]
