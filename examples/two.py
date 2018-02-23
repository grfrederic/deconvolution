"""
 In this example, we have an image and a known basis with two vectors. We want to get the density layers.

 In this example we use a cropped image [1], released under CC licence.

 [1] https://en.wikipedia.org/wiki/Chronic_lymphocytic_leukemia#/media/File:Chronic_lymphocytic_leukemia_-_high_mag.jpg
"""
from deconvolution import Deconvolution
from PIL import Image


def join_horizontally(*args):
    """Joins many PIL images of the same dimensions horizontally"""
    w, h = args[0].size
    n = len(args)
    joined = Image.new("RGB", (n*w, h))
    for x_off, img in zip(range(0, n*w, w), args):
        joined.paste(img, (x_off, 0))
    return joined


if __name__ == "__main__":
    # Load an image
    original = Image.open("cropped.jpg")

    # Create a deconvolution object with the image
    dec = Deconvolution(image=original, basis=[[0.91, 0.38, 0.71], [0.39, 0.47, 0.85]])

    # Produce density matrices for both colors. Be aware, as Beer's law do not always hold.
    first_density, second_density = dec.out_scalars()
    print(first_density.shape, second_density.shape)

    # Produce reconstructed image, first layer, second layer and rest
    out_images = dec.out_images(mode=[0, 1, 2, -1])
    # Original image, reconstructed, layers and rest
    join_horizontally(original, *out_images).show()

