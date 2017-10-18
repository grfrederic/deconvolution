import unittest
import numpy as np
import deconvolution.pixeloperations as px


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_proper_vector_check(self):
        """All components in [0,1] iff True"""
        self.assertTrue(px._proper_vector_check([]))

        for v in [[0, 1], [0.001, 0.1, 1], [0.1, 0.2, 0.3], [-0.], [0.55, 0.99]]:
            self.assertTrue(px._proper_vector_check(v))
            self.assertTrue(px._proper_vector_check(np.array(v)))

        for v in [[0, 1.001], [0.00, 0., -0.0001], [-0.1, 0, 0.3], [1e9], [0.55, 0.99, 10001, -1]]:
            self.assertFalse(px._proper_vector_check(v))
            self.assertFalse(px._proper_vector_check(np.array(v)))

    def test_array_to_colour_255(self):
        input_cases = []
        output_cases = []

        # Sub test 1 - array with shape (2, 3, 3)
        input_cases.append(np.array([
            [[100, 200, 355], [10., 20., 223], [80, 60, 1.]],
            [[-5., 4, 2], [-6, -10, 12], [-4, 2, 4]]
        ]))
        output_cases.append(np.array([
            [[100, 200, 255], [10, 20, 223], [80, 60, 1]],
            [[0, 4, 2], [0, 0, 12], [0, 2, 4]]
        ], dtype=np.uint8))

        # Sub test 2 - array with shape (2, 2, 3)
        input_cases.append(np.array([
            [[-10, 12, 55], [100., -20., 1223]],
            [[1e9, 180, -1], [0.99, 0.001, 0.1]]
        ]))
        output_cases.append(np.array([
            [[0, 12, 55], [100, 0, 255]],
            [[255, 180, 0], [0, 0, 0]]
        ], dtype=np.uint8))

        # Sub test 2 - array with shape (2, 2, 1)
        input_cases.append(np.array([
            [[55], [1223]],
            [[-1], [1.1]]
        ]))
        output_cases.append(np.array([
            [[55], [255]],
            [[0], [1]]
        ], dtype=np.uint8))

        for inp, out in zip(input_cases, output_cases):
            t_out = px._array_to_colour_255(inp)
            self.assertTrue(np.allclose(t_out, out, rtol=1e-05, atol=1e-08))
            self.assertEqual(t_out.dtype, np.dtype(np.uint8))

    def test_array_to_colour_255_empty(self):
        x = px._array_to_colour_255(np.array([]))
        self.assertEqual(len(x), 0)
        self.assertEqual(x.dtype, np.uint8)

    def test_array_to_colour_1(self):
        input_cases = []
        output_cases = []

        # Sub test 1 - array with shape (2, 3, 3)
        input_cases.append(np.array([
            [[1, 0.2, -0.0001], [1., 1.01, 1.], [0.2, 0.33, .4]],
            [[.31, .11, 20], [1e9, .01, 12], [-0.1, -0.001, 0.51]]
        ]))
        output_cases.append(np.array([
            [[1, 0.2, 0], [1, 1, 1], [0.2, 0.33, .4]],
            [[.31, .11, 1], [1, .01, 1], [0, 0, 0.51]]
        ], dtype=np.float))

        # Sub test 2 - array with shape (2, 2, 1)
        input_cases.append(np.array([
            [[.01], [1.17]],
            [[-.1], [.68]]
        ]))
        output_cases.append(np.array([
            [[.01], [1]],
            [[0], [.68]]
        ], dtype=np.float))

        for inp, out in zip(input_cases, output_cases):
            t_out = px._array_to_colour_1(inp)
            self.assertTrue(np.allclose(t_out, out, rtol=1e-05, atol=1e-08))
            self.assertEqual(t_out.dtype, np.float)

    def test_array_to_colour_1_empty(self):
        x = px._array_to_colour_1(np.array([]))
        self.assertEqual(len(x), 0)
        self.assertEqual(x.dtype, np.float)

    def test_array_positive_1(self):
        """Only one negative value to change, 1D array"""
        inp = np.array([-0.1, 0.001, 120])
        t_out = px._array_positive(inp)
        self.assertGreater(t_out[0], 0)
        self.assertLess(t_out[0], 0.001)
        for coord in range(1, len(inp)):
            self.assertEqual(inp[coord], t_out[coord])

    def test_array_positive_2(self):
        """Only one zero to change, 1D array"""
        inp = np.array([0., 0.001, 120])
        t_out = px._array_positive(inp)
        self.assertGreater(t_out[0], 0)
        self.assertLess(t_out[0], 0.001)
        for coord in range(1, len(inp)):
            self.assertEqual(inp[coord], t_out[coord])

    def test_array_positive_3(self):
        """Nothing to change, 1D array"""
        inp = np.array([0.1, 0.001, 120])
        t_out = px._array_positive(inp)
        self.assertTrue(np.allclose(t_out, inp, rtol=1e-05, atol=1e-08))

    def test_array_positive_4(self):
        """Nothing to change, 2D array"""
        inp = np.array([[0.1, 0.001, 120], [0.11, 0.01, 1000]])
        t_out = px._array_positive(inp)
        self.assertTrue(np.allclose(t_out, inp, rtol=1e-05, atol=1e-08))


class TestPixelOperations(unittest.TestCase):
    def test_set_background_1(self):
        """Check if background vector is white for empty init"""
        pix_ops = px.PixelOperations()
        self.assertTrue(np.allclose(pix_ops.get_background(), px._white1, rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
