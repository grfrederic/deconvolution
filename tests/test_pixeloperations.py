import unittest
import numpy as np
import deconvolution.pixeloperations as px
import deconvolution.exceptions as ex


np.random.seed(10*8*2*3*9)


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_entries_in_closed_interval(self):
        """All components are numbers in [0,1] iff True"""
        self.assertTrue(px._entries_in_closed_interval([]))

        for v in [[0, 1], [0.001, 0.1, 1], [0.1, 0.2, 0.3], [-0.], [0.55, 0.99]]:
            self.assertTrue(px._entries_in_closed_interval(v))
            self.assertTrue(px._entries_in_closed_interval(np.array(v)))

        for v in [
            [0, 1.001], [0.00, 0., -0.0001], [-0.1, 0, 0.3],
            [1e9], [0.55, 0.99, 10001, -1], ["a", 1], ["a", "b"]
        ]:
            self.assertFalse(px._entries_in_closed_interval(v))
            self.assertFalse(px._entries_in_closed_interval(np.array(v)))

    def test_entries_in_half_closed_interval(self):
        """All components are numbers in (0,1] iff True"""
        self.assertTrue(px._entries_in_half_closed_interval([]))

        for v in [[0.001, 1], [0.001, 0.1, 1], [0.1, 0.2, 0.3], [0.55, 0.99]]:
            self.assertTrue(px._entries_in_half_closed_interval(v))
            self.assertTrue(px._entries_in_half_closed_interval(np.array(v)))

        for v in [
            [0, 1.001], [0.00, 0., 0.0001], [-0.1, 0, 0.3], [0., 0.1],
            [1e9], [0.55, 0.99, 10001, -1], ["a", 1], ["a", "b"]
        ]:
            self.assertFalse(px._entries_in_half_closed_interval(v))
            self.assertFalse(px._entries_in_half_closed_interval(np.array(v)))

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

    def test_set_background_wrong_shape(self):
        """Check if errors are raised when background has wrong shape"""
        for vec in [
            [0.1, 0.2, 0.3, 0.4], [0.1, 0.2], [],
            [[0.1, 0.2, 0.3], [0.1, 0.2]], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        ]:
            with self.assertRaises(ValueError):
                pix_ops = px.PixelOperations()
                pix_ops.set_background(vec)

    def test_set_background_wrong_entries(self):
        """Check if errors are raised when background has correct shape but wrong entries"""
        for vec in [
            [0.1, 0.2, 0.0], [0.1, 0.2, -0.1], [0.1, 0.2, 1.01],
            [1.0, 0.2, -0.3], [1.0001, -0.0001, -0.3],
            ["a", "b", 1]
        ]:
            with self.subTest(vec=vec):
                with self.assertRaises(ValueError):
                    pix_ops = px.PixelOperations()
                    pix_ops.set_background(vec)

    def test_set_background_correct_entries(self):
        """List of correct background vectors"""
        for vec in [[0.1, 0.02, 0.9], [0.11, 0.01, 0.99], [0.001, 0.12, 0.999], [0.1, 1, 0.1]]:
            pix_ops = px.PixelOperations()
            pix_ops.set_background(vec)
            self.assertTrue(np.allclose(pix_ops.get_background(), vec, rtol=1e-05, atol=1e-08))

            pix_ops1 = px.PixelOperations()
            pix_ops1.set_background(np.array(vec))
            self.assertTrue(np.allclose(pix_ops.get_background(), np.array(vec), rtol=1e-05, atol=1e-08))

    def test_set_basis_wrong_shape(self):
        """Shape (3, 2)"""
        pix_ops = px.PixelOperations()
        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis([[0.4, 0.5], [1, 0.8], [0.9, 0.12]])

    def test_set_basis_wrong_shape_2(self):
        """Shape (3, 4)"""
        pix_ops = px.PixelOperations()
        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis([[0.1, 0.3, 0.5, 0.7], [0.1, 0.20, 0.2, 0.3], [0.33, 0.45, 0.1, 0.2]])

    def test_basis_wrong_entries(self):
        """Negative entries"""
        pix_ops = px.PixelOperations()
        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis([[0.1, -0.1, 0.0], [0.01, -0.1, 0.1]])

    def test_basis_wrong_entries_2(self):
        """Entries greater than 1"""
        pix_ops = px.PixelOperations()
        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis([[0.1, 1.01, 0.0], [0.01, 0.4, 0.1]])

    def test_basis_dimension_0(self):
        """Empty list of basis vectors"""
        pix_ops = px.PixelOperations()
        pix_ops.set_basis([])
        self.assertEqual(pix_ops.get_basis().size, 0)

    def test_basis_dimension_1(self):
        """One vector in basis"""
        basis = [[1, 0.1, 0]]
        pix_ops = px.PixelOperations()
        pix_ops.set_basis(basis)

        self.assertTrue(np.allclose(pix_ops.get_basis(), np.array(basis), rtol=1e-05, atol=1e-05))
        self.assertGreater(pix_ops.get_basis().max(), 0)
        self.assertEqual(pix_ops.get_basis_dim(), 1)

    def test_basis_dimension_2(self):
        """Two vectors in basis"""
        basis = [[1, 0.1, 0], [0.3, 0.2, 0.1]]
        pix_ops = px.PixelOperations()
        pix_ops.set_basis(basis)

        self.assertTrue(np.allclose(pix_ops.get_basis(), np.array(basis), rtol=1e-05, atol=1e-05))
        self.assertGreater(pix_ops.get_basis().max(), 0)
        self.assertEqual(pix_ops.get_basis_dim(), 2)

    def test_basis_dimension_3(self):
        """Three vectors in basis"""
        basis = [[1, 0.1, 0], [0.2, 0.2, 0.2], [0.1, 0.5, 1]]
        pix_ops = px.PixelOperations()
        pix_ops.set_basis(basis)

        self.assertTrue(np.allclose(pix_ops.get_basis(), np.array(basis), rtol=1e-05, atol=1e-05))
        self.assertGreater(pix_ops.get_basis().max(), 0)
        self.assertEqual(pix_ops.get_basis_dim(), 3)

    def test_basis_pseudodependent_1(self):
        """Two vectors that are (pseudo)linearly dependent"""
        basis = [[0.1, 0.2, 0.3], [0.01, 0.04, 0.09]]
        pix_ops = px.PixelOperations()

        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis(basis)

    def test_basis_pseudodependent_2(self):
        """Three vectors that are (pseudo)linearly dependent"""
        basis = [[0.1, 0.2, 0.3], [0.01, 0.04, 0.09], [0.5, 0.1, 0.3]]
        pix_ops = px.PixelOperations()

        with self.assertRaises(ex.BasisException):
            pix_ops.set_basis(basis)

    def test_get_coef2(self):
        """Some trivial combinations of two vectors. In some cases with noise"""
        basis = [[0.1, 0.2, 0.3], [0.7, 0.5, 0.1]]
        pix_ops = px.PixelOperations(basis=basis)

        image_1dim = [
                [0.1*0.7, 0.2*0.5, 0.3*0.1],
                [0.1**2, 0.2**2, 0.3**2],
                [0.01+0.001, 0.04-0.001, 0.09+0.001],
                [0.1*0.7**2+0.000001, 0.2*0.5**2-0.00001, 0.3*0.1**2+0.00001]
            ]
        image = 255. * np.array([image_1dim])
        coef = pix_ops.get_coef(image)
        coef_np = np.array(coef)

        desi = [
            [[1, 2, 2, 1]],
            [[1, 0, 0, 2]]
        ]
        desi_np = np.array(desi)

        # Uncomment to see numerical values
        # print("\ntest_get_coef2:\n", desi_np, "\n\n\n", coef_np)

        self.assertEqual(coef_np.shape, desi_np.shape)
        self.assertTrue(np.allclose(coef_np, desi_np, rtol=4e-02, atol=0.02))

    def test_get_coef3(self):
        """Some trivial combinations of three vectors. In some cases with noise"""
        basis = [[0.1, 0.2, 0.3], [0.7, 0.5, 0.1], [1, 0.9, 0.2]]
        pix_ops = px.PixelOperations(basis=basis)

        image_1dim = [
            [0.1*0.7*1, 0.2*0.5*0.9, 0.3*0.1*0.2],
            [0.1**2, 0.2**2, 0.3**2],
            [0.01+0.0001, 0.04-0.0001, 0.09+0.00001],
            [0.1*0.7**2*1+0.000001, 0.2*0.5**2*0.9-0.000001, 0.3*0.1**2*0.2+0.000001]
        ]
        image = 255. * np.array([image_1dim])
        coef = pix_ops.get_coef(image)
        coef_np = np.array(coef)

        desi = [
            [[1, 2, 2, 1]],
            [[1, 0, 0, 2]],
            [[1, 0, 0, 1]]
        ]
        desi_np = np.array(desi)

        # Uncomment to see numerical values
        # print("\ntest_get_coef3:\n", desi_np, "\n\n\n", coef_np)

        self.assertEqual(coef_np.shape, desi_np.shape)
        self.assertTrue(np.allclose(coef_np, desi_np, rtol=5e-02, atol=0.03))

    def test_get_coef_four_channels(self):
        """Four channels, an exception should be raised"""
        basis = [[0.1, 0.2, 0.3], [0.7, 0.5, 0.1], [1, 0.9, 0.2]]
        pix_ops = px.PixelOperations(basis=basis)

        image_1dim = [
            [0.5, 0.1, 0.5, 0.7],
            [0.1, 0.2, 0.6, 0.12]
        ]
        image = 255. * np.array([image_1dim])

        with self.assertRaises(ValueError):
            pix_ops.get_coef(image)

    def test_get_coef_no_basis(self):
        """Basis had not been set before get_coef was invoked"""
        pix_ops = px.PixelOperations()

        image_1dim = [
            [0.5, 0.1, 0.5],
            [0.1, 0.2, 0.6]
        ]
        image = 255. * np.array([image_1dim])

        with self.assertRaises(ex.BasisException):
            pix_ops.get_coef(image)


class TestTransformImageTwoDim(unittest.TestCase):
    u = np.array([0.1, 0.2, 0.3])
    v = np.array([1, 0.3, 0.8])
    basis = [u, v]

    @staticmethod
    def white_matrix(x, y):
        return 255 * np.ones(shape=(x, y, 3))

    def test_transform_image__1(self):
        """Test for a uniformly colored image, 50x50"""
        a = self.white_matrix(50, 50) * self.u**0.2 * self.v**0.3
        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1 = self.white_matrix(50, 50) * self.u**0.2
        r2 = self.white_matrix(50, 50) * self.v**0.3

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))

    def test_transform_image__2(self):
        """Test for a uniformly colored image, 100x50"""
        a = self.white_matrix(100, 50) * self.u**0.2 * self.v**0.3
        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1 = self.white_matrix(100, 50) * self.u**0.2
        r2 = self.white_matrix(100, 50) * self.v**0.3

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))

    def test_transform_image__3(self):
        """Test for a non-uniformly colored image"""
        a_1 = self.white_matrix(50, 50) * self.u**0.2 * self.v**0.3
        a_2 = self.white_matrix(50, 50) * self.u**0.1 * self.v**0.5
        a = np.concatenate((a_1, a_2))

        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1_1 = self.white_matrix(50, 50) * self.u**0.2
        r1_2 = self.white_matrix(50, 50) * self.u**0.1
        r1 = np.concatenate((r1_1, r1_2))

        r2_1 = self.white_matrix(50, 50) * self.v**0.3
        r2_2 = self.white_matrix(50, 50) * self.v**0.5
        r2 = np.concatenate((r2_1, r2_2))

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))

    def test_bad_image(self):
        """Test for an image of shape (X,Y,4)"""

        a = 255 * np.ones(shape=(50, 50, 4))
        pix_ops = px.PixelOperations(basis=self.basis)
        with self.assertRaises(ValueError):
            pix_ops.transform_image(a)

    def test_noisy_image(self):
        """Test for an image with noise added"""
        a = 255 * np.ones(shape=(50, 50, 3))
        a *= (self.u**0.2) * (self.v**0.3)

        a += np.random.rand(50, 50, 3)

        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        r1 = np.array(255 * np.ones(shape=(50, 50, 3)) * self.u**0.2, dtype=np.uint8)
        r2 = 255 * np.ones(shape=(50, 50, 3)) * self.v**0.3

        self.assertTrue(np.allclose(r[0], a, rtol=0, atol=2))
        self.assertTrue(np.allclose(r[1], r1, rtol=0, atol=2))
        self.assertTrue(np.allclose(r[2], r2, rtol=0, atol=2))


class TestTransformImageThreeDim(unittest.TestCase):
    u = np.array([0.1, 0.2, 0.3])
    v = np.array([1, 0.3, 0.8])
    t = np.array([0.7, 0.8, 0.1])
    basis = [u, v, t]

    @staticmethod
    def white_matrix(x, y):
        return 255 * np.ones(shape=(x, y, 3))

    def test_transform_image__1(self):
        """Test for a uniformly colored image, 50x50"""
        a = self.white_matrix(50, 50) * self.u**0.2 * self.v**0.3 * self.t**0.4
        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, 3, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1 = self.white_matrix(50, 50) * self.u**0.2
        r2 = self.white_matrix(50, 50) * self.v**0.3
        r3 = self.white_matrix(50, 50) * self.t**0.4

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[3], r3, rtol=5e-03, atol=1))

    def test_transform_image__2(self):
        """Test for a uniformly colored image, 100x50"""
        a = self.white_matrix(100, 50) * self.u**0.2 * self.v**0.3 * self.t**0.4
        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, 3, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1 = self.white_matrix(100, 50) * self.u**0.2
        r2 = self.white_matrix(100, 50) * self.v**0.3
        r3 = self.white_matrix(100, 50) * self.t**0.4

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[3], r3, rtol=5e-03, atol=1))

    def test_transform_image__3(self):
        """Test for a non-uniformly colored image"""
        a_1 = self.white_matrix(50, 50) * self.u**0.2 * self.v**0.3 * self.t**0.4
        a_2 = self.white_matrix(50, 50) * self.u**0.1 * self.v**0.5 * self.t**0.6
        a = np.concatenate((a_1, a_2))

        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        for ri in r:
            self.assertEqual(ri.dtype, np.uint8)
            self.assertEqual(ri.shape, a.shape)

        r1_1 = self.white_matrix(50, 50) * self.u**0.2
        r1_2 = self.white_matrix(50, 50) * self.u**0.1
        r1 = np.concatenate((r1_1, r1_2))

        r2_1 = self.white_matrix(50, 50) * self.v**0.3
        r2_2 = self.white_matrix(50, 50) * self.v**0.5
        r2 = np.concatenate((r2_1, r2_2))

        r3_1 = self.white_matrix(50, 50) * self.t**0.4
        r3_2 = self.white_matrix(50, 50) * self.t**0.6
        r3 = np.concatenate((r3_1, r3_2))

        self.assertTrue(np.allclose(r[0], a, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[1], r1, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[2], r2, rtol=5e-03, atol=1))
        self.assertTrue(np.allclose(r[3], r3, rtol=5e-03, atol=1))

    def test_bad_image(self):
        """Test for an image of shape (X,Y,4)"""

        a = 255 * np.ones(shape=(50, 50, 4))
        pix_ops = px.PixelOperations(basis=self.basis)
        with self.assertRaises(ValueError):
            pix_ops.transform_image(a)

    def test_noisy_image(self):
        """Test for an image with noise added"""
        a = 255 * np.ones(shape=(50, 50, 3))
        a *= (self.u**0.2) * (self.v**0.3)

        a += np.random.rand(50, 50, 3)

        b = np.array(a, dtype=np.uint8)

        pix_ops = px.PixelOperations(basis=self.basis)
        r = pix_ops.transform_image(b, mode=[0, 1, 2, -1])

        r1 = np.array(255 * np.ones(shape=(50, 50, 3)) * self.u**0.2, dtype=np.uint8)
        r2 = 255 * np.ones(shape=(50, 50, 3)) * self.v**0.3

        self.assertTrue(np.allclose(r[0], a, rtol=0, atol=2))
        self.assertTrue(np.allclose(r[1], r1, rtol=0, atol=2))
        self.assertTrue(np.allclose(r[2], r2, rtol=0, atol=2))
if __name__ == '__main__':
    unittest.main()
