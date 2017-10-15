import unittest
import numpy as np
import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import deconvolution.auxlib as aux


class TestSimpleFunctions(unittest.TestCase):
    def test_to_colour_1(self):
        """Test if numbers are thresholded properly to range [0,1]"""

        for x in [-1.2, -100, 0., -0.001]:
            with self.subTest(mode="to 0", x=x):
                self.assertEqual(aux.to_colour_1(x), 0.)

        for x in [1, 1.2, 100, 1000, 1.00001]:
            with self.subTest(mode="to 1", x=x):
                self.assertEqual(aux.to_colour_1(x), 1.)

        for x in [0., 1., 0.001, 0.3, 0.2, 0.7, 0.9, 0.99999]:
            with self.subTest(mode="id", x=x):
                self.assertEqual(aux.to_colour_1(x), x)

    def test_to_colour_255(self):
        """Test if numbers are thresholded properly to range [0,255]"""

        for x in [-1.2, -10, 0., -0.001, -2]:
            with self.subTest(mode="to 0", x=x):
                self.assertEqual(aux.to_colour_255(x), 0.)

        for x in [255, 257, 1000, 1000.0, 3124, 299.9]:
            with self.subTest(mode="to 255", x=x):
                self.assertEqual(aux.to_colour_255(x), 255)

        for x in [0., 255, 255.0, 5, 10., 20., 40., 120, 210.1]:
            with self.subTest(mode="id", x=x):
                self.assertEqual(aux.to_colour_255(x), int(x))

        for x in [-1., 0, 1, 255, 1., 1.2, 3.44, 300., 300.1, -10, -10.]:
            with self.subTest(mode="type", x=x):
                self.assertIsInstance(aux.to_colour_255(x), int)

    def test_positive(self):
        """Test if negative numbers are turned into positive"""
        for x in [0.1, 0.000001, 1E-9, 2, 2.11]:
            with self.subTest(x=x):
                self.assertEqual(aux.positive(x), x)
        for x in [-1E-9, -3, 0, 0.0, -0.0001, -1.1, -2]:
            with self.subTest(x=x, mode="positivity"):
                self.assertGreater(aux.positive(x), 0)
            with self.subTest(x=x, mode="< 0.01"):
                self.assertLess(aux.positive(x), 0.01)

        for x in [1, 2, 3, -1, 0, 1., 0.2, -0.2, 0.]:
            with self.subTest(x=x, mode="type"):
                self.assertIsInstance(aux.positive(x), float)

    def test_nonpositive(self):
        """Test if numbers are made non-positive"""
        for x in [-0.1, -0.000001, -1E-9, -2, -2.11]:
            with self.subTest(x=x):
                self.assertEqual(aux.negative(x), x)
        for x in [1E-9, 3, 0, 0.0, 0.0001, 1.1, 2]:
            with self.subTest(x=x):
                self.assertEqual(aux.negative(x), 0)

        for x in [1, 2, 3, -1, 0, 1., 0.2, -0.2, 0.]:
            with self.subTest(x=x, mode="type"):
                self.assertIsInstance(aux.negative(x), float)

    def test_check_positivity(self):
        """Test if check_positivity works properly"""
        c = {1, 2, 0.001, 0.1}

        # dictionary with positive values
        with self.subTest(container=c):
            self.assertTrue(aux.check_positivity(c))
        d = c | {0}
        # dictionary with 0
        with self.subTest(container=d):
            self.assertFalse(aux.check_positivity(d))
        d = c | {-0.001}
        # dictionary with negative value
        with self.subTest(conatiner=d):
            self.assertFalse(aux.check_positivity(d))
        # list with positive values
        d = list(c)
        with self.subTest(container=d):
            self.assertTrue(aux.check_positivity(d))
        # numpy array with positive values
        d = np.array(d)
        with self.subTest(container=d):
            self.assertTrue(aux.check_positivity(d))


class TestComplexFunctions(unittest.TestCase):
    def test_find_vals(self):
        """Test if we find correct density coefficients from log basis matix and pixel vals"""

        a = np.array([[1., 0.],
                      [0., 2.],
                      [0., 0.]])

        for r in [[1., 2., 5.], [1., 2., -5.], [1., 2., 0.]]:
            with self.subTest(mode="Simple projection", x=[a, r]):
                self.assertListEqual(list(aux.find_vals(a, r)), [-1., -1.])

        a = np.array([[1., 0.],
                      [0., 0.],
                      [0., 2.]])

        for r in [[1., 2., -5.], [1., -1., -5.]]:
            with self.subTest(mode="Simple projection", x=[a, r]):
                self.assertListEqual(list(aux.find_vals(a, r)), [-1., 2.5])

        a = np.array([[2., 0.],
                      [2., 0.],
                      [0., 1.]])

        for r in [[2., 2., 2.], [4., 0., 2.]]:
            with self.subTest(mode="Harder projection", x=[a, r]):
                self.assertListEqual(list(aux.find_vals(a, r)), [-1., -2.])

    def test_get_physical_normal(self):
        vl = list(
            map(lambda v: v/np.linalg.norm(v), np.random.randn(50, 3))
        )

        for v in vl:
            with self.subTest(x=v):
                n = aux.get_physical_normal(v)
                self.assertAlmostEqual(
                    np.linalg.norm(n), 1., 4
                )
                one_zero = (n[0] == 0 and n[1] * n[2] != 0) or\
                           (n[1] == 0 and n[2] * n[0] != 0) or\
                           (n[2] == 0 and n[0] * n[1] != 0)

                self.assertFalse(aux.check_positivity(+n))
                self.assertFalse(aux.check_positivity(-n))
                self.assertFalse(one_zero)

    def test_get_basis_from_normal(self):
        vl = list(
            map(lambda v: v/np.linalg.norm(v), np.random.randn(50, 3))
        )

        for v in vl:
            n = aux.get_physical_normal(v)
            with self.subTest(x=n):
                x, y = aux.get_basis_from_normal(n)
                self.assertAlmostEqual(np.dot(x, n), 0., 5)
                self.assertAlmostEqual(np.dot(y, n), 0., 5)
                self.assertTrue(np.alltrue(x > -np.full(3, aux._epsilon)))
                self.assertTrue(np.alltrue(y > -np.full(3, aux._epsilon)))

    def test_orthonormal_rotation(self):
        vl = list(
            map(lambda v: v/np.linalg.norm(v), np.random.randn(50, 3))
        )

        for v in vl:
            a = aux.orthonormal_rotation(v)
            k = np.array([1, 0, 0])
            with self.subTest(x=v):
                self.assertListEqual(list(np.matmul(a, k)), list(v))

    def test_find_vector(self):
        m = [[1., 0., 0.],
             [0., 2., 0.],
             [0., 0., 1.]]
        m = np.array(m)

        with self.subTest(x=m):
            v = aux.find_vector(m)
            self.assertListEqual(list(v), [0., 0., 1.])

        m = [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 2.]]
        m = np.array(m)

        with self.subTest(x=m):
            v = aux.find_vector(m)
            self.assertListEqual(list(v), [0., 1., 0.])

        m = [[1., 0.,  0.],
             [0., 2., -2.],
             [0., 2., -2.]]
        m = np.array(m)

        with self.subTest(x=m):
            v = aux.find_vector(m)
            self.assertTrue(np.linalg.norm(v - np.real(v)) < 0.001)
            v = np.real(v)
            self.assertAlmostEqual(v[0], 0., 4)
            self.assertAlmostEqual(v[1], np.sqrt(0.5), 4)
            self.assertAlmostEqual(v[2], np.sqrt(0.5), 4)

if __name__ == '__main__':
    unittest.main()
