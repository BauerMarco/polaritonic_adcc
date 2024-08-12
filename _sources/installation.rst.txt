.. _installation:

Installation
============

In order to run polaritonic_adcc, you first need to install the
dependencies required, which consist of adcc and an SCF provider.


Installing adcc
---------------

How to install adcc in general can be found 
`here <https://adc-connect.org/v0.15.13/installation.html>`_ ,
but until a required pull request gets accepted, you have to setup
adcc in a slightly different manner. 

After installing the required dependencies for adcc, clone the following
adcc fork on the given branch

.. code-block:: shell

   git clone https://github.com/BauerMarco/adcc/tree/increase_inheritable_objects

and then install by switching into the new directory and type

.. code-block:: shell

   python setup.py install


Installing an SCF provider
--------------------------

Many SCF providers are already supported, for instance VeloxChem, psi4 and pyscf.
However, since you want to conduct a polaritonic electronic structure
calculation, you probably also want to use a polaritonic SCF provider.
More on this can be found in the :ref:`theory` section, but just as a short
answer you should know, that polaritonic_adcc supports usage of both
standard (non-polaritonic) and polaritonic SCF providers, which do **not**
yield the same result.


Installing a standard SCF provider
..................................

How to install the standard SCF providers can
be found `here <https://adc-connect.org/v0.15.13/installation.html>`_ ,
and on the corresponding homepages of the program packages.


Installing a polaritonic SCF provider
.....................................

For polaritonic SCF providers one out of two things can occur.

1. An extension to one of the supported SCF providers yields the same
   object a standard calculation would yield. In this case the
   polaritonic SCF result can be parsed from polaritonic_adcc out of
   the box.


2. An extension to one of the supported SCF providers yields a different
   object than the standard calculation, or the SCF provider is not supported
   in the first place. In this case a new functionality needs to be integrated
   into adcc and/or polaritonic_adcc.

Besides polaritonic SCF providers falling under the first category, the
`hilbert package <https://github.com/edeprince3/hilbert>`_ is currently 
supported, which is based on the psi4 program package.


.. _tests:

Testing
=======

The implemented test routines are based on pytest and can be executed as follows

.. code-block:: shell

   pytest  # if you don't want to run the full test routine, use the -k flag

.. attention::
   Note, that the current test routine requires psi4 as a dependency.


