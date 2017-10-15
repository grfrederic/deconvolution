class BasisException(Exception):
    """Raised when basis is flawed. This could be caused by shape, type, parameter range or dimensional degeneracy."""


class ImageException(Exception):
    """Raised when an operation requiring an input image is called, but none has been set."""

