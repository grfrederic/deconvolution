import sys
sys.path.append("..")
from deconvolution import Deconvolution
from PIL import Image

if __name__ == "__main__":
    img = Image.open("cropped.jpg")
    dc = Deconvolution(image=img)

