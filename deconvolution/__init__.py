"""deconvolution is a Python module for performing (automatic) colour deconvolution.
"""

# -*- coding: utf-8 -*-
import numpy as np

import deconvolution.pixeloperations as po
import deconvolution.imageframe as fr


class Deconvolution:
    def complete_basis(self):
        """Automatically completes a zero or two vector basis to a two vector basis."""
        self.image_frame.complete_basis(self.pixel_operations)

    def resolve_dependencies(self, belligerency=0.3):
        """Tries to separate colour basis, so that output images are less dependent on each other.

        Parameters
        ----------
        belligerency : float
            aggressiveness of separation. Should be positive.
        """
        self.image_frame.resolve_dependencies(self.pixel_operations, belligerency=belligerency)

    def out_images(self, mode=None):
        """Get deconvolved images
        :param mode: which substances (or reconstructed/remainder), None means all

        Parameters
        ----------
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

        See Also
        --------
        ImageFrame.out_images
        """
        if not self.pixel_operations.check_basis():
            self.complete_basis()
            self.resolve_dependencies()

        return self.image_frame.out_images(pixel_operations=self.pixel_operations, mode=mode)

    def out_scalars(self):
        """Get deconvolved scalar density fields.

        Returns
        -------
        list
            list of numpy arrays (length is the dimensionality of basis), each with exponent field of coefficient

        See Also
        --------
        ImageFrame.out_scalars
        """
        if not self.pixel_operations.check_basis():
            self.complete_basis()
            self.resolve_dependencies()

        return self.image_frame.out_scalars(pixel_operations=self.pixel_operations)

    def set_source(self, in_image):
        """Sets image for deconvolution.

        Parameters
        ----------
        in_image : PIL Image
            input image
        """
        self.image_frame.set_image(in_image)

    # set basis and background check dimensionality before passing further
    def set_basis(self, basis):
        """Sets initial basis.

        Parameters
        ----------
        basis : array_like
            list of lists or numpy array. Basis can be of one of the following shapes: (0,), (1, 3), (2, 3), (3, 3).
        """
        basis = np.array(basis, dtype=float)

        if basis.shape not in [(0,), (1, 3), (2, 3), (3, 3)]:
            if self.verbose:
                print("Basis has invalid dimensions, and was not set.", basis)
            return 1

        self.pixel_operations.set_basis(basis)

    def set_background(self, background):
        """Sets background to be adjusted for.

        Parameters
        ----------
        background : array_like
            colour vector (three components, each in [0,1])
        """
        background = np.array(background, dtype=float)
        if np.shape(background) != (3,):
            if self.verbose:
                print("Background vector has invalid dimensions.")
                return 1

        self.pixel_operations.set_background(background)

    def set_verbose(self, verbose):
        """Change verbosity.

        Parameters
        ----------
        verbose : bool
            set to True prints to the std output internal actions
        """
        self.verbose = verbose

    def __init__(self, image=None, basis=None, verbose=False, background=None):
        """High-level class able to deconvolve PIL Image without any effort and gain other images and stain densities

        Parameters
        ----------
        image : PIL Image
            input image
        basis : array_like
            list of lists or numpy array. Basis can be of one of the following shapes: (0,), (1, 3), (2, 3), (3, 3).
        verbose : bool
            set to True prints to the std output internal actions
        background : array_like
            colour vector (three components, each in [0,1])

        Notes
        --------
        This class acts mainly as an interface. Most functions are implemented in either PixelOperations or ImageFrame.
        """
        self.pixel_operations = po.PixelOperations(basis=basis, background=background)
        self.image_frame = fr.ImageFrame(image=image, verbose=verbose)

        self.verbose = False
        self.sample_flag = False

        if verbose is not None:
            self.set_verbose(verbose)