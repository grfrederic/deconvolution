# -*- coding: utf-8 -*-
import numpy as np

import pixeloperations as po
import imageframe as fr


class Deconvolution:
    def complete_basis(self):
        self.image_frame.complete_basis(self.pixel_operations)

    def resolve_dependencies(self, belligerency=0.3):
        self.image_frame.resolve_dependencies(belligerency=belligerency)

    def out_images(self, mode=None):
        return self.image_frame.out_images(pixel_operations=self.pixel_operations, mode=mode)

    def out_scalars(self, mode=None):
        return self.image_frame.out_scalars(pixel_operations=self.pixel_operations, mode=mode)

    def save_images(self, name=None, mode=None):
        return self.image_frame.save_images(pixel_operations=self.pixel_operations, name=name, mode=mode)

    def save_scalars(self, name=None, mode=None):
        return self.image_frame.save_scalars(pixel_operations=self.pixel_operations, name=name, mode=mode)

    def set_source(self, in_image):
        self.image_frame.set_image(in_image)

    def set_basis(self, basis):
        basis = np.array(basis, dtype=float)

        if basis.shape not in [(0,), (1, 3), (2, 3), (3, 3)]:
            if self.verbose:
                print("Basis has invalid dimensions, and was not set.", basis)
            return 1

        self.pixel_operations.set_basis(basis)

    def set_background(self, background):
        background = np.array(background, dtype=float)
        if np.shape(background) != (3,):
            if self.verbose:
                print("Background vector has invalid dimensions.")
                return 1

        self.pixel_operations.set_background(background)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def __init__(self, image=None, basis=None, verbose=None, background=None):

        self.pixel_operations = po.PixelOperations(basis=basis, background=background)
        self.image_frame = fr.ImageFrame(image=image)

        self.verbose = False
        self.sample_flag = False

        if verbose is not None:
            self.set_verbose(verbose)
