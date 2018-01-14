*************
deconvolution
*************
A Python module providing Deconvolution class that implements and generalises Ruifrok-Johnston color deconvolution algorithm [RJ]_. It allows one to split an image into distinct color layers

.. code:: python

  from deconvolution import Deconvolution
  from PIL import Image

  img = Image.open("image.jpg")
  
  # Declare an instance of Deconvolution, with image loaded and with color basis defining what layers are interesting
  decimg = Deconvolution(image=img, basis=[[1, 0.1, 0.2], [0, 0.1, 0.8]])
  
  # Constructs new PIL Images, with different color layers
  layer1, layer2 = decimg.out_images(mode=[1, 2])


Installation
------------
TODO

Testing
-------
.. code:: bash
  
  # For Python 3 users
  make test
  
  # For Python 2 users
  make comp

  # Check the code coverage
  make coverage

  # Check the coverage interactively, using a web browser
  make html

Contributors
------------
Method developed by Frederic Grabowski generalising Ruifrok-Johnston algorithm [RJ]_. and implemented by Frederic Grabowski [FG]_ and Paweł Czyż [PC]_.
Special thanks to prof. Daniel Wójcik and dr Piotr Majka [N1]_, [N2]_ who supervised the project.

References
----------
.. [RJ] https://www.researchgate.net/publication/11815294_Ruifrok_AC_Johnston_DA_Quantification_of_histochemical_staining_by_color_deconvolution_Anal_Quant_Cytol_Histol_23_291-299
.. [N1] http://en.nencki.gov.pl/laboratory-of-neuroinformatics
.. [N2] https://github.com/Neuroinflab/
.. [FG] https://github.com/grfrederic
.. [PC] https://github.com/pawel-czyz
