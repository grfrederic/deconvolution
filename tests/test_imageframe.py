import unittest
import numpy as np
from PIL import Image
import deconvolution.imageframe as ifr
import deconvolution.pixeloperations as px
import deconvolution.exceptions as ex


# Prepare images
img_arr = np.array(
    [[[100, 255, 255], [255, 255, 255]],
     [[100, 100, 255], [255, 100, 255]]], dtype=np.uint8)
img_mask_R_G = Image.fromarray(img_arr)

img_arr = np.array(
    [[[100, 255, 255], [255, 255, 255]],
     [[100, 255, 100], [255, 255, 100]]], dtype=np.uint8)
img_mask_R_B = Image.fromarray(img_arr)

img_arr = np.array(
    [[[100, 255, 255], [255, 255, 255]],
     [[100, 100, 100], [255, 100, 100]]], dtype=np.uint8)
img_mask_R_GB = Image.fromarray(img_arr)

del img_arr


class TestImageFrame(unittest.TestCase):
    noneArray = np.array([None])

    def test_ImageFrame_class(self):
        """Check if ImageFrame's are properly constructed"""
        image_frame = ifr.ImageFrame()

        self.assertFalse(image_frame.get_verbose())
        self.assertEqual(image_frame.get_threads(), 1)
        self.assertEqual(image_frame.get_image(), self.noneArray)

        image_frame = ifr.ImageFrame(image=img_mask_R_B,
                                     threads=4,
                                     verbose=True)

        self.assertTrue(image_frame.get_verbose())
        self.assertEqual(image_frame.get_threads(), 4)

        image_frame = ifr.ImageFrame()
        image_frame.set_verbose(True)
        self.assertTrue(image_frame.get_verbose())

        with self.assertRaises(ValueError):
            image_frame.set_verbose(2.32)

        with self.assertRaises(ValueError):
            image_frame.set_image({1:42})

    def test_matrix_inertia(self):
        pixel_operations = px.PixelOperations()
        image_frame = ifr.ImageFrame()
        image_frame.set_verbose(True)

        # no image yet
        with self.assertRaises(ex.ImageException):
            image_frame.sample_source(pixel_operations)
 
        # no image yet
        with self.assertRaises(ex.ImageException):
            image_frame.get_inertia_matrix()       

        image_frame = ifr.ImageFrame(image=img_mask_R_G, threads=4, sample_density=2)

        # not sampled
        with self.assertRaises(ex.ImageException):
            image_frame.get_inertia_matrix()

        image_frame.sample_source(pixel_operations)
        a = np.array(image_frame.get_inertia_matrix())

        self.assertTrue(
            np.linalg.norm(np.dot(a, [0, 0, 1])) <
            np.linalg.norm(np.dot(a, [1, 0, 0]))
        )

        self.assertTrue(
            np.linalg.norm(np.dot(a, [0, 0, 1])) <
            np.linalg.norm(np.dot(a, [0, 1, 0]))
        )

        self.assertTrue(
            np.linalg.norm(np.dot(a, [0, 0, 1])) <
            np.linalg.norm(np.dot(a, [0.7071, 0.7071, 0]))
        )


class TestBasisPickingAndCompletion(unittest.TestCase):
    @staticmethod
    def white_matrix(x, y):
        return 255 * np.ones(shape=(x, y, 3))

    def test_inner_mistakes(self):
        pixel_operations = px.PixelOperations(basis=[])
        image_frame = ifr.ImageFrame(image=img_mask_R_G, threads=2, verbose=True)

        with self.assertRaises(ex.BasisException):
            image_frame._find_substance_one(pixel_operations)

        with self.assertRaises(ex.BasisException):
            image_frame.resolve_dependencies(pixel_operations)

    def test_missing_data(self):
        pixel_operations = px.PixelOperations(basis=[[0.5, 0.9, 0.9], [0.9, 0.5, 0.9]])
        image_frame = ifr.ImageFrame(threads=2, verbose=True)
        with self.assertRaises(ex.ImageException):
            image_frame.complete_basis(pixel_operations)

        image_frame.set_image(img_mask_R_G)
        image_frame.complete_basis(pixel_operations)

    def test_complete_one(self):
        u = np.array([0.5, 0.9, 0.9])
        v = np.array([0.9, 0.5, 0.9])

        a = self.white_matrix(50, 50) * u**0.2 * v**0.3
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = [u]
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3, verbose=True)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=3.))

    def test_dont_complete_2(self):
        u = np.array([0.5, 0.9, 0.9])
        v = np.array([0.9, 0.5, 0.9])

        a = self.white_matrix(50, 50) * u**0.2 * v**0.3
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = [u, v]
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3)
        image_frame.set_verbose(True)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=0))

    def test_complete_two(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256)
                a[i][j] *= v**((j+1)/256)
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3, sample_density=8)
        image_frame.set_verbose(True)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=3.))

    def test_out_scalars(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])

        a = self.white_matrix(256, 256)
        uf = np.zeros([256, 256])
        vf = np.zeros([256, 256])

        for i in range(256):
            for j in range(256):
                uf[i][j] = ((i+1)/256)
                vf[i][j] = ((j+1)/256)
                a[i][j] *= u**uf[i][j]
                a[i][j] *= v**vf[i][j]
        img = Image.fromarray(np.array(a, dtype=np.uint8))
        image_frame = ifr.ImageFrame(image=img, threads=3)

        basis = [u, v]
        pixel_operations = px.PixelOperations(basis=basis)

        uf_out, vf_out = image_frame.out_scalars(pixel_operations)
        
        self.assertTrue(np.allclose(uf, uf_out, rtol=0.01, atol=3.))
        self.assertTrue(np.allclose(vf, vf_out, rtol=0.01, atol=3.))

    def test_complete_two_and_resolve(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256)
                a[i][j] *= v**((j+1)/256)
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3, sample_density=8)
        image_frame.set_verbose(True)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)
        image_frame.resolve_dependencies(pixel_operations, belligerency=0.001)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=3.))

    def test_resolve(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256)
                a[i][j] *= v**((256-i)/256)
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(threads=3, sample_density=8)

        with self.assertRaises(ex.ImageException):
            image_frame.out_images(pixel_operations)

        image_frame.set_image(img)
        image_frame.set_verbose(True)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)
        image_frame.resolve_dependencies(pixel_operations, belligerency=0.001)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        un, vn = pixel_operations.get_basis()
        self.assertTrue(
            np.allclose(
                np.cross(np.log(u), np.log(un)),
                0, 
                rtol=0.00, atol=0.01
            ) or np.allclose(
                np.cross(np.log(u), np.log(vn)),
                0, 
                rtol=0.00, atol=0.01
            )
        )

    def test_resolve_pathologies(self):
        img_arr = np.array(
            [[[255, 100, 100], [255, 100, 255]],
             [[255, 255, 100], [255, 255, 255]]], dtype=np.uint8)
        img = Image.fromarray(img_arr)

        image_frame = ifr.ImageFrame(image=img, threads=3, verbose=True)
        basis = [
            [0.5, 0.5, 1.],
            [0.5, 1., 0.5]
        ]
        pixel_operations = px.PixelOperations(basis=basis)

        with self.assertRaises(ValueError):
            image_frame.resolve_dependencies(pixel_operations,
                                             belligerency = -1)

        image_frame.resolve_dependencies(pixel_operations,
                                         belligerency = 100)

        self.assertTrue(
            np.allclose(basis,
                        pixel_operations.get_basis(),
                        rtol=0.01, atol=3.
            )
        )

    def test_sample_density_1(self):
        for wrong_val in [-1, 0, 1, 9, 10, 100]:
            with self.assertRaises(ValueError):
                ifr.ImageFrame(sample_density=wrong_val)

    def test_sample_density_2(self):
        for wrong_val in ["abc", 2.6, 3.0, {1, 2}]:
            with self.assertRaises(TypeError):
                ifr.ImageFrame(sample_density=wrong_val)

if __name__ == '__main__':
    unittest.main()
