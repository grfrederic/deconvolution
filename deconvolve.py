#!/usr/bin/python3
from deconvolution import Deconvolution
from PIL import Image
import argparse
import json
from os.path import basename


if __name__ == "__main__":
    # input to operate on
    source = []

    parser = argparse.ArgumentParser(description="Deconvolve an image.")
    parser.add_argument("--sample_density", help="set sampling density. Range 2-8", type=int, default=5)
    parser.add_argument("--basis", help="set basis", default="[]", type=str)
    parser.add_argument("--belligerency", help="set resolve aggressiveness", default=0.3)
    parser.add_argument("--mod", help="set output mode", type=list, default=[-1, 0, 1, 2])
    parser.add_argument("-r", dest="r", action="store_const", const=True, default=False,
                        help="resolve dependencies")
    parser.add_argument("-v", dest="v", action="store_const", const=True, default=False,
                        help="resolve dependencies")
    parser.add_argument("images", metavar="IMAGE", type=str, nargs="+", help="paths to an image")

    args = parser.parse_args()
    args.basis = json.loads(args.basis)
    if args.sample_density not in range(2, 9):
        raise ValueError("--sample_density must be between 2 and 8")

    for imgpth in args.images:
        img = Image.open(imgpth)
        dec = Deconvolution(image=img, verbose=args.v, basis=args.basis, sample_density=args.sample_density)

        dec.complete_basis()

        if args.r:
            dec.resolve_dependencies()

        mode_to_name = {
            -1: "rest_",
            0: "reconstructed_",
            1: "first_substance_",
            2: "second_substance_",
            3: "third_substance_"
        }

        bsimgpth = basename(imgpth)

        for m, img in zip(args.mod, dec.out_images(mode=args.mod)):
            img.save(mode_to_name[m]+bsimgpth)
