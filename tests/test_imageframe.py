import unittest
import numpy as np
from PIL import Image
import deconvolution.imageframe as ifr
import deconvolution.pixeloperations as px


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
        self.assertEqual(image_frame.get_inertia_matrix(), self.noneArray)

        image_frame = ifr.ImageFrame(image=img_mask_R_B,
                                     threads=4,
                                     verbose=True)

        self.assertTrue(image_frame.get_verbose())
        self.assertEqual(image_frame.get_threads(), 4)
        self.assertEqual(image_frame.get_inertia_matrix(), self.noneArray)

        image_frame = ifr.ImageFrame()
        image_frame.set_verbose(True)
        self.assertTrue(image_frame.get_verbose())

    def test_sample_source(self):
        pixel_operations = px.PixelOperations()
        image_frame = ifr.ImageFrame(image=img_mask_R_B, threads=4)

        image_frame.sample_source(pixel_operations, sample_density=8)
        a = np.array(image_frame.get_inertia_matrix())

        self.assertTrue(a[0, 0] > 20. * a[1, 0])
        self.assertTrue(a[0, 0] > 20. * a[0, 1])
        self.assertTrue(a[0, 0] > 20. * a[1, 1])
        self.assertTrue(a[0, 0] > 20. * a[2, 1])
        self.assertTrue(a[0, 0] > 20. * a[1, 2])

        self.assertTrue(a[2, 2] > 20. * a[1, 0])
        self.assertTrue(a[2, 2] > 20. * a[0, 1])
        self.assertTrue(a[2, 2] > 20. * a[1, 1])
        self.assertTrue(a[2, 2] > 20. * a[2, 1])
        self.assertTrue(a[2, 2] > 20. * a[1, 2])

        self.assertTrue(a[0, 2] > 20. * a[1, 0])
        self.assertTrue(a[0, 2] > 20. * a[0, 1])
        self.assertTrue(a[0, 2] > 20. * a[1, 1])
        self.assertTrue(a[0, 2] > 20. * a[2, 1])
        self.assertTrue(a[0, 2] > 20. * a[1, 2])

    def test_complete_basis(self):
        pixel_operations = px.PixelOperations()
        image_frame = ifr.ImageFrame(image=img_mask_R_B, threads=4)
        image_frame.sample_source(pixel_operations, sample_density=8)

        image_frame.complete_basis(pixel_operations)
        basis = pixel_operations.get_basis()

        # print("\n", basis)


if __name__ == '__main__':
    unittest.main()
