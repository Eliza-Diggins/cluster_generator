"""Frontends for converting ``cluster_generator`` systems into formats used by other
software.

Overview
--------

Simulation codes (regardless of their inner workings) all follow a similar implementation style in ``cluster_generator``. Each code has a module
in :py:mod:`codes` which shares its name. Within that module, there is a core submodule (with the same name + ``.py``) containing the "code-class" for that simulation
code. For example, the code-class for the AREPO code may be accessed as

.. code-block:: python

    >>> from cluster_generator.codes.arepo import Arepo

The code-class for each simulation code is the only part of the code that the user needs to interact with and is designed to make
converting initial conditions made in cluster generator to formats recognized by the code as easy as possible. All code-classes implement
a few familiar methods that users should be familiar with and they are all subclassed from the :py:class:`~codes.abc.SimulationCode`:

1. The :py:meth:`~codes.abc.SimulationCode.generate_ics` method takes an instance of :py:class:`ics.ClusterICs` and converts them into
   the correct format for use with the simulation software.

2. The :py:meth:`~codes.abc.SimulationCode.determine_runtime_params` uses the user's inputs (created when initializing the class) to
   determine what parameters need to be passed to the simulation software to get the ICs setup and running.

3. The :py:meth:`~codes.abc.SimulationCode.generate_rtp_template` will use the runtime parameters it deduces and write a "parameter file" or
   similar (equivalent) template for you to use when getting the simulation started.

   .. note::

        Simulation codes like these often have hundreds of different runtime parameters. We cannot account for every use-case; instead,
        we provide a template with reasonable defaults for most or all of the runtime parameters and with correct values for all of the runtime
        parameters that are involved in loading the ICs. From there, it's the user's responsibility to make sure that their science
        goals are met by their runtime setup of the code.

Code Structure
++++++++++++++

Hydrodynamics codes range considerably in their design, implementation, methodology, etc.; however, ``cluster_generator``
takes a general approach to implementing support for these codes so that the user experience is similar between codes (at least as
far as using ``cluster_generator`` is concerned). In ``cluster_generator``, simulation codes are composed of 3 components:

- **Compile-Time Parameters** (CTPs): are the parameters used when compiling the code. These are also parameters of the :py:class:`~codes.abc.SimulationCode`
  class for the code; however, they almost always have default values and so don't need to be set by the user unless the user has
  a non-standard setup.

  In some cases, codes may have a :py:meth:`~codes.abc.SimulationCode.from_install_directory`; which can be used to automatically determine
  the CTPs that were used when compiling the code.

- **Runtime Parameters** (RTPs): are the parameters the user feeds to the simulation software to set up each simulation.

  For any given code, there may be hundreds of RTPs, not all of which are relevant for correctly loading initial conditions
  into the simulation code. Nonetheless, cluster generator does need to manage the RTPs which are relevant to the ICs. To do so,
  ``cluster_generator`` keeps a copy of the RTPs for each code in ``.yaml`` files. These default values can be accessed via
  the :py:attr:`~codes.abc.SimulationCode.rtp`. Once instantiated, the :py:attr:`~codes.abc.SimulationCode.rtp` attribute of the
  class instance takes on the default values but can be changed by the user at will.

  Once the user is ready to set up their ICs, the :py:meth:`~codes.abc.SimulationCode.determine_runtime_params` uses the parameters the
  user provided when instantiating the :py:class:`~codes.abc.SimulationCode` to determine the correct values for RTPs.

- **IC Format**: The specific format the code expects for the ICs.

  The :py:class:`~codes.abc.SimulationCode` implements a method: :py:meth:`~codes.abc.SimulationCode.generate_ics`, which
  takes a :py:class:`ics.ClusterICs` instance and converts it into the correct format for the simulation.


Examples
--------

.. admonition:: In Progress

    This section is still being written. It will be completed soon.
"""
from .arepo.arepo import Arepo

__all__ = ["arepo", "athena", "flash", "gamer", "ramses", "gadget"]
