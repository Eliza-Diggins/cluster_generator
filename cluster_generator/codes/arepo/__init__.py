"""AREPO (magneto)hydrodynamics software support module.

.. _arepo:
AREPO Overview
--------------

|support-level| |docs| |type|

`Arepo <https://arepo-code.org/>`_ is a massively parallel gravity and magnetohydrodynamics code for astrophysics, designed for problems
of large dynamic range. It employs a finite-volume approach to discretize the equations of hydrodynamics
on a moving Voronoi mesh, and a tree-particle-mesh method for gravitational interactions. Arepo is originally
optimized for cosmological simulations of structure formation, but has also been used in many other applications
in astrophysics.

.. raw:: html

    <iframe src="https://drive.google.com/file/d/1pMEIExuvBK0CGtHFQi8FZO4FMm781EkO/preview" width="100%" height="480" allow="autoplay"></iframe>


Code Configuration
------------------

Downloading AREPO
+++++++++++++++++

You can obtain AREPO from the gitlab url: https://gitlab.mpcdf.mpg.de/vrs/arepo. The first thing to do is clone the repository
into your machine:

.. code-block:: shell

    $ git clone https://gitlab.mpcdf.mpg.de/vrs/arepo

.. note::

    In order to use AREPO with Cluster Generator, you will actually need two separate compiled versions of the AREPO code. You can
    choose to either recompile as needed or setup two different installation. We will do the later in this example.

Configuring your System
+++++++++++++++++++++++

.. hint::

    **Big Idea**: AREPO basically needs a blue print of where to access the relevant packages it needs at compile time. This
    part of the compilation process will necessarily be system specific, but should be similar between most systems. If you're working
    on a supercomputing system with dedicated support staff, they should be able to answer any questions you might have about this
    step.

In ``Makefile``, you'll find a section with the following schematic:

.. code-block:: make

    ifeq ($(SYSTYPE),"Darwin")

    # Set a bunch of environmental variables.

    ifeq ($(SYSTYPE),"Ubuntu")

    # Set a bunch of environmental variables.

    endif

These directives provide the ``Makefile`` with the locations to access all of the various pre-requisite dependencies (accessible `here <https://arepo-code.org/wp-content/userguide/running.html#library-requirements>`_).
If necessary, add a branch to the if statement for your ``SYSTYPE``.

.. hint::

    For example, the following is a sufficient set of specifications for the University of Utah's CHPC clusters

    .. code-block:: make

        ifeq ($(SYSTYPE),"CHPC")
        CC        =  mpicc   # sets the C-compiler
        OPTIMIZE  =  -std=c11 -ggdb -O3 -Wall -Wno-format-security -Wno-unknown-pragmas -Wno-unused-function        # Standard linux.

        MPICH_INCL =                                                                                                # ML automatically adds MPI to -I
        MPICH_LIB  = -lmpi
        GSL_INCL   =                                                                                                # ML automatically adds GSL to -I
        GSL_LIB    = -lgsl -lgslcblas
        HWLOC_LIB = -lhwloc

        # optional libraries
        FFTW_INCL =                                                                                                  # ML automatically adds...
        FFTW_LIBS =
        HDF5_INCL = -DH5_USE_16_API
        HDF5_LIB  = -lhdf5 -lz
        HWLOC_INCL=

        endif

If you need to add a custom ``SYSTYPE`` as described, you will also need to edit the ``Makefile.systype`` to reflect the
correct ``SYSTYPE`` for your system.

Compiling AREPO
+++++++++++++++

During the compilation stage, AREPO uses a number of "compile-time flags" which play a critical role in the behavior of the end software.
These compile-time flags become the CTP's that the user can set at initialization in the :py:class:`arepo.Arepo` class. Our recommendation for
these parameters is as follows; however, this may differ from yours depending on the science goal. First, copy the ``Template-Config.sh`` file to
``Config.sh``, then alter the flags in ``Config.sh`` to meet your needs.

.. code-block:: shell

    #!/bin/bash            # this line only there to enable syntax highlighting in this file

    ##################################################
    #  Enable/Disable compile-time options as needed #
    ##################################################

    #--------------------------------------- Mesh motion and regularization; default: moving mesh
    REGULARIZE_MESH_CM_DRIFT      # Mesh regularization; Move mesh generating point towards center of mass to make cells rounder.
    REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED  # Limit mesh regularization speed by local sound speed
    REGULARIZE_MESH_FACE_ANGLE    # Use maximum face angle as roundness criterion in mesh regularization

    #--------------------------------------- Refinement and derefinement; default: no refinement/derefinement; criterion: target mass
    REFINEMENT_SPLIT_CELLS        # Refinement
    REFINEMENT_MERGE_CELLS        # Derefinement
    REFINEMENT_VOLUME_LIMIT       # Limit the volume of cells and the maximum volume difference between neighboring cels
    NODEREFINE_BACKGROUND_GRID    # Do not de-refine low-res gas cells in zoom simulations


    #--------------------------------------- Gravity treatment; default: no gravity
    SELFGRAVITY                   # gravitational intraction between simulation particles/cells
    HIERARCHICAL_GRAVITY          # use hierarchical splitting of the time integration of the gravity
    CELL_CENTER_GRAVITY           # uses geometric centers to calculate gravity of cells, only possible with HIERARCHICAL_GRAVITY
    GRAVITY_NOT_PERIODIC          # gravity is not treated periodically
    ALLOW_DIRECT_SUMMATION        # Performed direct summation instead of tree-based gravity if number of active particles < DIRECT_SUMMATION_THRESHOLD (= 3000 unless specified differently here)
    DIRECT_SUMMATION_THRESHOLD=500  # Overrides maximum number of active particles for which direct summation is performed instead of tree based calculation
    EVALPOTENTIAL                 # computes gravitational potential
    ENFORCE_JEANS_STABILITY_OF_CELLS

    #--------------------------------------- Gravity softening
    NSOFTTYPES=2                  # Number of different softening values to which particle types can be mapped.
    MULTIPLE_NODE_SOFTENING       # If a tree node is to be used which is softened, this is done with the softenings of its different mass components
    INDIVIDUAL_GRAVITY_SOFTENING=32  # bitmask with particle types where the softenig type should be chosen with that of parttype 1 as a reference type
    ADAPTIVE_HYDRO_SOFTENING      # Adaptive softening of gas cells depending on their size




    #--------------------------------------- Time integration options
    TREE_BASED_TIMESTEPS          # non-local timestep criterion (take 'signal speed' into account)


    #--------------------------------------- Single/Double Precision
    DOUBLEPRECISION=1             # Mode of double precision: not defined: single; 1: full double precision 2: mixed, 3: mixed, fewer single precisions; unless short of memory, use 1.
    NGB_TREE_DOUBLEPRECISION      # if this is enabled, double precision is used for the neighbor node extension


    #--------------------------------------- output options
    PROCESS_TIMES_OF_OUTPUTLIST   # goes through times of output list prior to starting the simulaiton to ensure that outputs are written as close to the desired time as possible (as opposed to at next possible time if this flag is not active)

    #--------------------------------------- Testing and Debugging options
    DEBUG                         # enables core-dumps

Once completed, you can compile AREPO using the ``make`` command.

Initial Conditions
------------------

Because AREPO uses a moving-mesh methodology, there is some conversion necessary from the ``cluster_generator`` output initial conditions (which are
essentially SPH ICs) and those compatible with AREPO. Most significantly, AREPO needs to have a fixed mesh inserted behind the ICs that come out
of cluster generator. Arepo comes with a tool for this, but it does require having two copies of the code on disk... First, copy the
first AREPO directory to a new directory (don't copy the build, just the code). Everything should match your previous installation except that
the only necessary parameters are

.. code-block:: shell

    #!/bin/bash         # this line only there to enable syntax highlighting in this file

    ##########################################################################
    #  Enable/Disable compile-time options as needed                         #
    #  examples/galaxy_merger_star_formation_3d/Config_ADDBACKGROUNDGRID.sh  #
    ##########################################################################


    #--------------------------------------- Basic operation mode of code
    ADDBACKGROUNDGRID=16                     # Re-grid hydrodynamics quantities on a Oct-tree AMR grid. This does not perform a simulation.


    #--------------------------------------- Mesh motion and regularization
    REGULARIZE_MESH_FACE_ANGLE               # Use maximum face angle as roundness criterion in mesh regularization


    #--------------------------------------- Gravity softening
    NSOFTTYPES=2                             # Number of different softening values to which particle types can be mapped.


    #--------------------------------------- output options
    HAVE_HDF5                                # needed when HDF5 I/O support is desired (recommended)


Now, recompile the code.

Generating Initial Conditions
-----------------------------


.. |support-level| image:: https://img.shields.io/badge/Support-80%25-green?logo=gnometerminal
.. |docs| image:: https://img.shields.io/badge/Documentation-Moderate-blue?logo=readthedocs
   :target: https://arepo-code.org/
.. |type| image:: https://img.shields.io/badge/Code_Type-Moving_Mesh-black
"""
from .arepo import Arepo
