"""
 In this example, we have an image and an insufficient number of color vectors - we will use Deconvolution
 complete_basis method to find other vectors.

 In this example we use a cropped image [1], released under CC licence.

 [1] https://en.wikipedia.org/wiki/Chronic_lymphocytic_leukemia#/media/File:Chronic_lymphocytic_leukemia_-_high_mag.jpg
"""
import sys
sys.path.append("..")

from deconvolution import Deconvolution
from PIL import Image

if __name__ == "__main__":
    img = Image.open("cropped.jpg")
    dc = Deconvolution(image=img)


