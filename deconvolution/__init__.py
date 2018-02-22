"""deconvolution is a Python module for performing (automatic) colour deconvolution.
"""

# -*- coding: utf-8 -*-
import deconvolution.pixeloperations as po
import deconvolution.imageframe as fr


class Deconvolution:
    def complete_basis(self):
        """Automatically completes a zero or two vector basis to a two vector basis."""
        self.image_frame.complete_basis(self.pixel_operations)

    def resolve_dependencies(self, belligerency=0.1):
        """Tries to separate colour basis, so that output images are less dependent on each other.

        Parameters
        ----------
        belligerency : float
            aggressiveness of separation. Should be positive.
        """
        self.image_frame.resolve_dependencies(self.pixel_operations, belligerency=belligerency)

    def out_images(self, mode=None):
        """Get deconvolved images

        Parameters
        ----------
        mode : array_like
            elements can be 0 (image generated from white light and two stains), 1 (white light and first stain),
            2 (white light and second stain), 3 (white light and third stain. Note that this works only if a basis
            with three vectors is used) or -1 (remove all stains to obtain the rest)

        Returns
        -------
        list
            list of PIL Images, in order same as given by `mode`

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

    def set_basis(self, basis):
        """Sets initial basis.

        Parameters
        ----------
        basis : array_like
            list of lists or numpy array. Basis can be of one of the following shapes: (0,), (1, 3), (2, 3), (3, 3).
        """
        self.pixel_operations.set_basis(basis)

    def set_background(self, background):
        """Sets background to be adjusted for.

        Parameters
        ----------
        background : array_like
            colour vector (three components, each in [0,1])
        """
        self.pixel_operations.set_background(background)

    def set_verbose(self, verbose):
        """Change verbosity.

        Parameters
        ----------
        verbose : bool
            set to True prints to the std output internal actions
        """
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise ValueError("Variable verbose has to be bool.")

    def __init__(self, image=None, basis=None, verbose=False, background=None, sample_density=5):
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
        sample_density : int
            precision of sampling. Should be in interval [2, 8]

        Notes
        --------
        This class acts mainly as an interface. Most functions are implemented in either PixelOperations or ImageFrame.
        """

        self.pixel_operations = po.PixelOperations(basis=basis, background=background)
        self.image_frame = fr.ImageFrame(image=image, verbose=verbose, sample_density=sample_density)

        self.sample_flag = False
        self.verbose = False

        self.set_verbose(verbose)
