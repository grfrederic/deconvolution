import unittest
import numpy as np
import deconvolution.pixeloperations as px


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_proper_vector_check(self):
        """All components in [0,1] iff True"""
        for v in [[0, 1], [0.001, 0.1, 1], [0.1, 0.2, 0.3], [-0.], [0.55, 0.99]]:
            self.assertTrue(px._proper_vector_check(v))
            self.assertTrue(px._proper_vector_check(np.array(v)))

        for v in [[0, 1.001], [0.00, 0., -0.0001], [-0.1, 0, 0.3], [1e9], [0.55, 0.99, 10001, -1]]:
            self.assertFalse(px._proper_vector_check(v))
            self.assertFalse(px._proper_vector_check(np.array(v)))


if __name__ == '__main__':
    unittest.main()
