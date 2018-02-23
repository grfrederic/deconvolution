"""
 In this example, we have an image and an insufficient number of color vectors - we will use Deconvolution
 complete_basis method to find other vectors.
 Then we try to make them more independent applying resolve_dependencies with different belligerency parameter.
 We show all the bases found.

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


def join_vertically(*args):
    """Joins many PIL images of the same dimensions vertically"""
    w, h = args[0].size
    n = len(args)
    joined = Image.new("RGB", (w, n*h))
    for y_off, img in zip(range(0, n*h, h), args):
        joined.paste(img, (0, y_off))
    return joined

if __name__ == "__main__":
    # Load an image
    original = Image.open("cropped.jpg")

    # Create a deconvolution object with the image
    dec = Deconvolution(image=original, sample_density=6)
    # Complete basis - as we did not provide 2 or 3 vectors, it needs to be found
    dec.complete_basis()
    # We can get the basis found
    print("Basis before resolve:\n{}\n".format(dec.pixel_operations.get_basis()))

    # Produce reconstructed image, first layer, second layer and rest
    out_images1 = dec.out_images(mode=[0, 1, 2, -1])
    # Original image, reconstructed, layers and rest
    before_resolve = join_horizontally(original, *out_images1)

    # Resolve dependencies - make the vectors more independent
    dec.resolve_dependencies(belligerency=0.1)
    # We can get the basis found
    print("Basis after resolve:\n{}\n".format(dec.pixel_operations.get_basis()))
    # And produce reconstructed image, first layer, second layer and rest
    out_images2 = dec.out_images(mode=[0, 1, 2, -1])
    # Construct a joined image from original image, reconstructed, layers and rest
    after_resolve = join_horizontally(original, *out_images2)

    # Resolve dependencies with huge belligerency - make the vectors very independent
    dec.resolve_dependencies(belligerency=1)
    print("Basis after aggressive resolve:\n{}\n".format(dec.pixel_operations.get_basis()))
    # Produce reconstructed image, first layer, second layer and rest
    out_images3 = dec.out_images(mode=[0, 1, 2, -1])
    # Show original image, reconstructed, layers and rest
    after_huge_resolve = join_horizontally(original, *out_images3)

    # Show all images for visual comparision
    join_vertically(before_resolve, after_resolve, after_huge_resolve).show()

