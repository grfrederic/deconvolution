========
Examples
========
Ruifrok-Johnston deconvolution
------------------------------
Assume that we have an image and we know three stains. We can deconvolve image and get density layers: 

.. include:: ../../examples/three.py
  :code: python

Two stain deconvolution
-----------------------
Alternatively, we can use two stains. If we know them, we can just mimic the procedure for two stains:

.. include:: ../../examples/two.py
  :code: python

But if we do not know at least one of them, we should first find appropriate vectors:

.. include:: ../../examples/zero_or_one.py
  :code: python
