import numpy as np
import auxlib as aux


class ImageFrame:
    def __init__(self, image=None):
        """
        Initialising function
        :param image: image to deconvolve
        """
        self.image = image
