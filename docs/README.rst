.. cluster_generator documentation master file, created by
   sphinx-quickstart on Mon Jul 27 14:41:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



cluster_generator
=================
|yt-project| |docs| |testing| |Github Page| |Pylint| |coverage|

.. raw:: html

   <hr style="height:10px;background-color:black">

``cluster_generator`` is a cross-platform galaxy cluster initializer for N-body / hydrodynamics codes. ``cluster_generator`` supports
a variety of different possible configurations for the initialized galaxy clusters, including a variety of profiles, different construction
assumptions, and non-Newtonian gravity options.


.. raw:: html

   <hr style="color:black">

Features
========

.. raw:: html

   <table width="100%" table-layout="fixed">
   <tr>
      <td width="500" style="vertical-align: middle;">
      <h3 style="text-align: center;"> Gravities </h3> </td>
      <td width="500" style="vertical-align: middle;">
      <h3 style="text-align: center;"> Profiles </h3> </td>
   </tr>
   <tr>
      <td>
      <ul>
      <li> Newtonian Gravity </li>
      <li> <a href="https://en.wikipedia.org/wiki/AQUAL">AQUAL</a> (<a href="https://en.wikipedia.org/wiki/Modified_Newtonian_dynamics">MOND</a>ian) </li>
      <li> <a href="https://watermark.silverchair.com/mnras0403-0886.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2UwggNhBgkqhkiG9w0BBwagggNSMIIDTgIBADCCA0cGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMGb-fef5Ctx7fV5WJAgEQgIIDGKepu4GZqp7A-i3x1gJbehOyxm4vG9kx4eohWE2ipnUGBf_25ORxOWVF3RG5-wVYger-KaprllV2wY4GHZ0wgwHvb21RfjhDFLkQH7iVLLR2PJTIIEXVfrdU1djeQtRmtcc-NbRF_iAAxoE6q3RDr3hhTndEaYnR_ElwbUhCctE9UcZHCqiD4-3MbwCfKmQm1NJRsI38vjiti9EoHbuz0VVT4-vyOrMIySssTS6A_qGUnW_r2Ar0yDBrtqbjJk5QkOhxG6ZtJQtLFWAJZ6rh5j66ifwBdmPpIaBlsPUM0FcctpFVi8BuvdhaQkE06WzsAvCm-etmIkzV83sNw0bT1G2l-YkZYMJ6IqqX8oqN4kzKxlwYp58CfHg4RNbIXtGwkwmw-FYIXRgbTlinbwlxa9pQO3XxtCySEjDbwFKGzQy-FtqNVDSWpAa4F87y1ie2XzU5pDZri7Fzw4Tw2W0izjptcb6hG1TPFFmQ_X-eXC48yToIXTaoVcdZrAiX3CtLWDLoXM2PbeaSs3ARJszpgZKavP3Et-kPnkhskV589iZSLKVGR4eR8uhCGXWu07sNFCixOMPA6KGkUOBrvukhhdcT0tjbX93SsPB_UH1MOyVowaKjJwkVGGUFEcb3LfYTsqBbs8PZWcu3Jomr6yd7zo5s6hExmHACfz_h_ic8kUZWSnAr3P2TlGNQgyX9DX9O6pghWMtkuhomWu4r9f6Mv2xMjVJ1A_ZCwGZIPm7SeBc70s1TaT4daMzLG6UDEQevzv8M3W7jkd4gYOWqBojvWz2JyR2SO7YWC_LHb4JD6VgrvsvAwZcrEoHyIGb_O25ULxEtgz2d8hd_cbmsO8XgE_VrTp2gz6Twp3c2J46_TpOJitrkKR7MUhr91MNHR5XypthZfQ5zxYR13fQ78TvE-RDe9enShgqlIYU0QQGmfSqocSx8LHFq8B1HPcQiCEMQl5-8tz39dANME-Hvmxn0a9XGblHeGeO5R6Dfgb-AyWW3oZJYJUmNHMpY2P-lS2Bpy8Fmhb_LthPyZqyqj7w2INBr6mWv2TkTpA">QUMOND</a> (<a href="https://en.wikipedia.org/wiki/Modified_Newtonian_dynamics">MOND</a>ian) </li>
      </ul>
      </td>
      <td>
      <ul>
      <li> [T/S]-NFW  </li>
      <li> Hernquist [cored / uncored] </li>
      <li> Einasto </li>
      <li> Power Law </li>
      <li> Beta Model </li>
      <li> am06 </li>
      <li> Vikhlinin (density) </li>
      <li> Vikhlinin (temperature) </li>
      <li> Entropy [baseline, broken, walker] </li>
      </ul>
      </td>
   </tr>
   </table>


Installing ``cluster_generator``
--------------------------------
In this guide are instructions on the installation of the ``cluster_generator`` library. If you encounter any issues during
the installation process, please visit the **ISSUES** page of the github and report the issue. We will attempt to provide
support as quickly as possible.

.. raw:: html

   <hr style="height:10px;background-color:black">

Getting the Package
===================
From PyPI
+++++++++
.. attention::

    This package is not yet published on PyPI

From Source
+++++++++++
To gather the necessary code from source, simple navigate to a directory in which you'd like to store the local copy
of the package and execute

.. code-block:: bash

    git clone https://github.com/jzuhone/cluster_generator

If you want a specific branch of the project, use the ``-b`` flag in the command and provide the name of the branch.

Once the git clone has finished, there should be a directory ``./cluster_generator`` in your current working directory.

.. raw:: html

   <hr style="height:3px;background-color:black">


Dependencies
============

``cluster_generator`` is compatible with Python 3.8+, and requires the following
Python packages:

- `unyt <http://unyt.readthedocs.org>`_ [Units and quantity manipulations]
- `numpy <http://www.numpy.org>`_ [Numerical operations]
- `scipy <http://www.scipy.org>`_ [Interpolation and curve fitting]
- `h5py <http://www.h5py.org>`_ [h5 file interaction]
- `tqdm <https://tqdm.github.io>`_ [Progress bars]
- `ruamel.yaml <https://yaml.readthedocs.io>`_ [yaml support]

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.

.. admonition:: Recommended

    Though not required, it may be useful to install `yt <https://yt-project.org>`_
    for creation of in-memory datasets from ``cluster_generator`` and/or analysis of
    simulations which are created using initial conditions from
    ``cluster_generator``.

Installation
============

``cluster_generator`` can be installed in a few different ways. The simplest way
is via the conda package if you have the
`Anaconda Python Distribution <https://store.continuum.io/cshop/anaconda/>`_:

.. code-block:: bash

    [~]$ conda install -c jzuhone cluster_generator

This will install all of the necessary dependencies.

The second way to install ``cluster_generator`` is via pip. pip will attempt to
download the dependencies and install them, if they are not already installed
in your Python distribution:

.. code-block:: bash

    [~]$ pip install cluster_generator

Alternatively, to install into your Python distribution from
`source <http://github.com/jzuhone/cluster_generator>`_:

.. code-block:: bash

    [~]$ git clone https://github.com/jzuhone/cluster_generator
    [~]$ cd cluster_generator
    [~]$ python -m pip install .




Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://eliza-diggins.github.io/cluster_generator/build/html/index.html

.. |testing| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg
.. |Pylint| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg
.. |Github Page| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg
.. |coverage| image:: https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=MOND
   :target: https://coveralls.io/github/Eliza-Diggins/cluster_generator?branch=MOND