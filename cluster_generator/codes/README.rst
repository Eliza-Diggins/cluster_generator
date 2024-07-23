Cluster Generator External Simulation Codes
============================================

Overview
--------

Cluster Generator is intended to be intuitive, easy to use, and widely applicable. One critical aspect of meeting these goals is
ensuring that initial conditions generated in CG are easily exported for use in external hydrodynamics simulation software. Doing this
is much easier said than done for a few reasons:

1. Hydrodynamics simulation codes are complicated. Really complicated.

   This means that there is a lot that can go wrong with the user's setup of the code, cluster generator's interfacing with said code, etc.
   that can lead to people getting stuck. Furthermore, errors arising from the simulation codes are usually originating from low level code
   which can make debugging difficult for scientists who don't have an intimate understanding of the code's inner workings.

2. There are a lot of parameters

   Because of their complexity, simulation codes have a number of compile-time and runtime parameters which can effect how they function.
   In order to use cluster generator ICs in a given code, it may be necessary to specify more than just the correct format for the ICs, but also
   to provide certain parameters at runtime.

To deal with these complications, the ``codes`` module of ``cluster_generator`` is developed in a very particular way to maximize the user's
ability to achieve their science goal from the use of our software and the simulation code they wish to use.


This document is intended as a guide to developers and users regarding how the ``code`` module is structured, how to work with it, and what
pitfalls to be aware of.


Simulation Code Classes
-----------------------

Every hydrodynamics code has its own class in the ``codes`` module; typically following the convention ``codes.<code_name>.<CodeName>``. All
of these are subclasses of the class ``codes.abc.SimulationCode``. Every ``SimulationCode`` subclass has the following basic structure:

- **Compile-time Parameters**: These are the settings you specify to the simulation code when you install it. ``cluster_generator`` needs to
  know these so that we can ensure the IC's provided will actually work.

  The CTPs of a given code are just **attributes** of the class (``class variables``). They typically have defaults, but can be specified as ``kwargs`` when
  instantiating the class.

  In some codes, it's also possible to have the ``SimulationCode`` class read the CTPs from disk, but this isn't implemented for all codes.

  .. hint::

    See the section on :ref:`compile-time parameters <ctp>` for a more comprehensive explanation of how these are implemented and
    for examples of them in action.

- **Runtime Parameters**: Unlike CTPs, RTPs are specified by the user *before every simulation run*. They typically tell the code
  where the ICs are stored, how long to run the code, where to store the output, etc. Codes can have hundreds of RTPs.

  Cluster Generator isn't responsible for idiot-proofing the use of simulation codes. Doing so would be impossible. As such, the runtime parameters
  are generally left up to the user to determine; however, **cluster generator related RTPs are provided**.

  Every code has a special ``codes.abc.RuntimeParameters`` class (subclass) which   tells ``cluster_generator`` about the RTPs for a particular code.
  Whenever you   ask the ``SimulationCode`` class to generate ICs for your simulation, it will also fill in the blanks about what different RTP values should be.
  It will then provide a "skeleton" of the necessary RTPs for your run. These should never be assumed to be correct,
  but should be a good guide for getting things running.

  .. hint::

    See the section on :ref:`runtime parameters <rtp>` for a more comprehensive explanation of how these are implemented and
    for examples of them in action.

- **IC Generator**: Finally, every ``SimulationCode`` class implements a ``generate_ics`` method which takes a ``ClusterICs`` instance
  from the user and converts it into the necessary format for use with the corresponding simulation code.

.. _ctp:

Compile-time Parameters
+++++++++++++++++++++++

.. _rtp:

Runtime Parameters
++++++++++++++++++

Runtime parameters are
