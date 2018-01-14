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

        # no image yet
        with self.assertRaises(ex.ImageException):
            image_frame.sample_source(pixel_operations)
 
        # no image yet
        with self.assertRaises(ex.ImageException):
            image_frame.get_inertia_matrix()       

        image_frame = ifr.ImageFrame(image=img_mask_R_G, threads=4)

        # not sampled
        with self.assertRaises(ex.ImageException):
            image_frame.get_inertia_matrix()

        image_frame.sample_source(pixel_operations, sample_density=2)
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

    def test_complete_one(self):
        u = np.array([0.5, 0.9, 0.9])
        v = np.array([0.9, 0.5, 0.9])

        a = self.white_matrix(50, 50) * u**0.2 * v**0.3
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = [u]
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3)

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

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=0))

    def test_complete_two(self):
        u = np.array([0.5, 0.9, 0.9])
        v = np.array([0.9, 0.5, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256) * v**((i+1)/256) 
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=3.))

    def test_complete_two_and_resolve(self):
        # TODO fails??
        u = np.array([0.5, 0.9, 0.9])
        v = np.array([0.9, 0.5, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256) * v**((j+1)/256) 
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img, threads=3)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)
        #image_frame.resolve_dependencies(pixel_operations)

        out = np.asarray(
                image_frame.out_images(pixel_operations, mode=[0])[0]
        )
        
        self.assertTrue(np.allclose(a, out, rtol=0.01, atol=10.))

    def test_resolve(self):
        return #TODO
        u = np.array([0.5, 1.0, 1.0])
        v = np.array([1.0, 0.5, 1.0])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256) * v**((j+1)/256) 

        img = Image.fromarray(np.array(a, dtype=np.uint8))

        basis = []
        pixel_operations = px.PixelOperations(basis=basis)
        image_frame = ifr.ImageFrame(image=img)

        image_frame.sample_source(pixel_operations)
        image_frame.complete_basis(pixel_operations)

        print(pixel_operations.get_basis(), [u, v])
        image_frame.resolve_dependencies(pixel_operations)
        print(pixel_operations.get_basis(), [u, v])
#        self.assertTrue(np.allclose(
#            pixel_operations.get_basis,
#            [u, v], 
#            rtol=0.01, atol=3.)
#        )


if __name__ == '__main__':
    unittest.main()
