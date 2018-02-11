import unittest
import numpy as np
from PIL import Image
import deconvolution as dc
import deconvolution.imageframe as ifr
import deconvolution.pixeloperations as px
import deconvolution.exceptions as ex


class TestBasisPickingAndCompletion(unittest.TestCase):
    @staticmethod
    def white_matrix(x, y):
        return 255 * np.ones(shape=(x, y, 3))

    def test_simple_usage(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])
        b = np.array([0.8, 0.7, 0.6])
        w = np.array([255, 255, 255])

        img_arr = np.array(
            [[u * v * w, u * w],
             [    v * w,     w]], dtype=np.uint8)
        img_arr_tainted = np.array(
            [[u * v * b * w, u * b * w],
             [    v * b * w,     b * w]], dtype=np.uint8)
        img = Image.fromarray(img_arr_tainted)

        dec = dc.Deconvolution()
        with self.assertRaises(ValueError):
            dec.set_verbose(None)
        dec.set_verbose(True)
        dec.set_source(img)
        dec.set_background(b)
        dec.set_basis([u, v])
        out_img_arr = np.asarray(dec.out_images(mode=[0])[0])
        uf, vf = dec.out_scalars()
        print("NOW")
        print(uf)
        print(vf)
        
        self.assertTrue(
            np.allclose(img_arr,
                        out_img_arr,
                        rtol=0.01, atol=3.
            )
        )
        self.assertTrue(
            np.allclose(uf,
                        [[1, 1], [0, 0]],
                        rtol=0., atol=0.01
            )
        )
        self.assertTrue(
            np.allclose(vf,
                        [[1, 0], [1, 0]],
                        rtol=0., atol=0.01
            )
        )

    def test_basis_completion(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])
        w = np.array([255, 255, 255])

        img_arr = np.array(
            [[u * v * w, u * w],
             [    v * w,     w]], dtype=np.uint8)
        img = Image.fromarray(img_arr)

        dec = dc.Deconvolution(image=img)
        out_img_arr = np.asarray(dec.out_images(mode=[0])[0])
        self.assertTrue(
            np.allclose(img_arr,
                        out_img_arr,
                        rtol=0.01, atol=5.
            )
        )

        dec = dc.Deconvolution(image=img)
        dec.complete_basis()
        dec.resolve_dependencies(belligerency=0.0001)
        uf, vf = dec.out_scalars()
        t = uf * vf
        self.assertTrue(
            np.allclose([t[0][1], t[1][1], t[1][0]],
                        0,
                        rtol=0., atol=0.01
            )
        )

    def test_out_scalars_directly(self):
        u = np.array([0.3, 0.9, 0.9])
        v = np.array([0.9, 0.3, 0.9])

        a = self.white_matrix(256, 256)
        for i in range(256):
            for j in range(256):
                a[i][j] *= u**((i+1)/256)
                a[i][j] *= v**((j+1)/256)
        img = Image.fromarray(np.array(a, dtype=np.uint8))

        dec = dc.Deconvolution(image=img)
        uf, vf = dec.out_scalars()
        self.assertTrue(
            np.all(uf > 0)
        )
        self.assertTrue(
            np.all(vf > 0)
        )

if __name__ == '__main__':
    unittest.main()
