"""Frontends for converting ``cluster_generator`` systems into formats used by other
software.

Overview
--------

`cluster_generator` provides a unified framework for converting initial conditions (ICs) into formats compatible with various simulation codes.
Each simulation code supported by `cluster_generator` has a dedicated module in :py:mod:`codes`, named after the code itself. Within these modules,
a core submodule contains the "code-class" for the respective simulation code. For instance, the AREPO code's class can be accessed as follows:

.. code-block:: python

    >>> from cluster_generator.codes.arepo import Arepo

The code-class for each simulation code is the primary interface for users, facilitating the conversion of ICs generated in `cluster_generator`
to formats compatible with the simulation code. All code-classes inherit from the :py:class:`~codes.abc.SimulationCode` abstract base class (ABC)
and implement a set of standardized methods:

1. **:py:meth:`~codes.abc.SimulationCode.generate_ics`**: Converts an instance of :py:class:`ics.ClusterICs` into the format required by the simulation software.

2. **:py:meth:`~codes.abc.SimulationCode.determine_runtime_params`**: Uses user inputs (provided during class instantiation) to determine the necessary parameters
   for initializing the ICs in the simulation software.

3. **:py:meth:`~codes.abc.SimulationCode.generate_rtp_template`**: Generates a template "parameter file" or equivalent for the simulation, incorporating the runtime
   parameters deduced from the user's input.

   .. note::

       Simulation codes may have numerous runtime parameters, often hundreds. While `cluster_generator` provides templates with reasonable defaults for many parameters,
       the user must ensure that these defaults align with their scientific goals. Users may need to adjust these settings based on their specific requirements.

Code Structure
--------------

Hydrodynamics codes can vary significantly in their design and implementation. However, `cluster_generator` aims to provide a consistent user experience
across different simulation codes by adhering to a standardized approach. The structure of each simulation code within `cluster_generator` consists of three main components:

1. **Compile-Time Parameters (CTPs)**: Parameters used during the compilation of the simulation code. These parameters are also attributes of the
   :py:class:`~codes.abc.SimulationCode` class. Typically, CTPs have default values and do not require user modification unless using a non-standard setup.

   Some codes offer a method, :py:meth:`~codes.abc.SimulationCode.from_install_directory`, to automatically determine the CTPs used during compilation.

2. **Runtime Parameters (RTPs)**: Parameters required by the simulation software to initialize and run simulations. While there are many RTPs for each simulation
   code, `cluster_generator` focuses on managing those essential for loading ICs. Default RTPs are stored in `.yaml` files within `cluster_generator` and can be accessed
   via the :py:attr:`~codes.abc.SimulationCode.rtp` attribute. Once a code-class instance is created, its `rtp` attribute adopts the default values, which users can modify as needed.

   The :py:meth:`~codes.abc.SimulationCode.determine_runtime_params` method uses user-provided parameters from the class instantiation to set the correct RTPs for the simulation.

3. **IC Format**: The specific data format expected by the simulation code for the initial conditions. The :py:meth:`~codes.abc.SimulationCode.generate_ics` method takes an
   :py:class:`ics.ClusterICs` instance and converts it into the required format for the simulation software.

Examples
--------

.. admonition:: In Progress

    Examples demonstrating the implementation of new frontends are currently under development and will be added soon.
"""
