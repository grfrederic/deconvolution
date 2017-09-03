class BasisException(Exception):
    """Raised when basis is flawed. This could be caused by shape, type, parameter range or dimensional degeneracy.
    """

    def __init__(self, message):
        Exception.__init__(self, message)

class ImageException(Exception):
    """Raised when basis is flawed. This could be caused by shape, type, parameter range or dimensional degeneracy.
    """

    def __init__(self, message):
        Exception.__init__(self, message)