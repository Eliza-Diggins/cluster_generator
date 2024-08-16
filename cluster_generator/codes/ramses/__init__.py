"""RAMSES (magneto)hydrodynamics software support module.

.. _ramses:
RAMSES Overview
--------------

|support-level| |docs| |type|

`RAMSES <https://bitbucket.org/rteyssie/ramses/src/master/>`_ Ramses is an open source code to model astrophysical systems, featuring self-gravitating, magnetised, compressible, radiative fluid flows.
It utilizes an AMR scheme to compute fluid flows with a poisson solver for collisionless components.


Code Configuration
------------------

Downloading RAMSES
+++++++++++++++++++

You can obtain RAMSES from the bitbucket url: https://bitbucket.org/rteyssie/ramses/src/master/. The first thing to do is clone the repository
into your machine:

.. code-block:: shell

    $ git clone https://bitbucket.org/rteyssie/ramses/src/master/


Compiling RAMSES
++++++++++++++++

.. hint::

    For the most part, RAMSES is a simple(r) code to compile. You shouldn't need to change many parameters
    to get things running.

During the compilation stage, RAMSES uses a number of "compile-time flags" which play a critical role in the behavior of the end software.
These compile-time flags become the CTP's that the user can set at initialization in the :py:class:`ramses.Ramses` class. Our recommendation for
these parameters is as follows; however, this may differ from yours depending on the science goal. The compile-time flags are contained in the
``/bin`` directory of the repo and look like the following:

.. code-block:: make

    #############################################################################
    # If you have problems with this makefile, contact Romain.Teyssier@gmail.com
    #############################################################################
    # Compilation time parameters

    # Do we want a debug build? 1=Yes, 0=No
    DEBUG = 0
    # Compiler flavor: GNU or INTEL
    COMPILER = GNU
    # Size of vector cache
    NVECTOR = 32
    # Number of dimensions
    NDIM = 3 #! ---> CLUSTER_GENERATOR REQUIRES 3 DIMS
    # Float precision size
    NPRE = 8
    # hydro/mhd/rhd solver
    SOLVER = hydro # -----> CLUSTER_GENERATOR REQUIRES HYDRO (for now)
    # Patch
    PATCH =
    # Use RT? 1=Yes, 0=No
    RT = 0
    # Use turbulence? 1=Yes, 0=No (requires fftw3)
    USE_TURB = 0
    # Use MPI? 1=Yes, 0=No
    MPI = 0
    MPIF90 = mpif90
    # Root name of executable
    EXEC = ramses
    # Number of additional energies
    NENER = 0
    # Use Grackle cooling? 1=Yes, 0=No
    GRACKLE = 0
    # Number of metal species
    NMETALS = 0
    # Use ATON? 1=Yes, 0=No
    ATON = 0
    # Number of ions for RT
    # use 1 for HI+HII, +1 for added H2, +2 for added HeI, HeII, HeIII
    NIONS = 0
    # Number of photon groups for RT
    NGROUPS = 0
    # Number of passive scalars
    NPSCAL = 0
    # Light MPI communicator structure (for > 10k MPI processes)
    LIGHT_MPI_COMM = 0


Once completed, you can compile RAMSES using the ``make`` command.

Initial Conditions
------------------

RAMSES is a fairly typical AMR software; however, its support for standard formats is somewhat antiquated. As it stands, there are a
few options for loading your simulation into RAMSES. The choice of which to use will largely be dependent on your experience
with the software and your particular scientific goals:

- **DICE Patch**: RAMSES provides support for DICE, an IC software for generating disk galaxies. This allows the user to load
  particle ICs into RAMSES.



.. |support-level| image:: https://img.shields.io/badge/Support-80%25-green?logo=gnometerminal
.. |docs| image:: https://img.shields.io/badge/Documentation-Moderate-blue?logo=readthedocs
   :target: https://arepo-code.org/
.. |type| image:: https://img.shields.io/badge/Code_Type-Moving_Mesh-black
"""
from .ramses import Ramses
