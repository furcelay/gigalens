GIGA-Lens-Clusters: Dual Source Plane (DSP)
========================

.. image:: https://img.shields.io/pypi/v/gigalens.svg
    :target: https://pypi.python.org/pypi/gigalens
    :alt: Latest PyPI version

This is an alternative branch of Giga-Lens intended for group and cluster lens modeling, this stills under intense
development. This version is intended only for DSP systems where the first source is also a lens.
See Collett & Auger 2014 for more details.

Gradient Informed, GPU Accelerated Lens modelling (GIGA-Lens) is a package for fast Bayesian inference on strong
gravitational lenses. For details, please see `our paper <https://arxiv.org/abs/2202.07663>`__. See
`here <https://giga-lens.github.io/gigalens/>`__ for our documentation.

Usage
-----

Installation
------------
Install via pip from the github repo ::

    pip install --no-deps --upgrade git+https://github.com/furcelay/gigalens.git@cluster-lens#egg=gigalens


Requirements
^^^^^^^^^^^^
The following packages are requirements for GIGA-Lens. However, ``!pip install gigalens`` is all you need to do. In fact,
separately installing other packages can cause issues with subpackage dependencies. Some users may find it necessary
to install PyYAML.

::

    tensorflow>=2.6.0
    tensorflow-probability>=0.15.0
    lenstronomy==1.9.3
    scikit-image==0.18.2
    tqdm==4.62.0

The following dependencies are required by ``lenstronomy``:

::

    cosmohammer==0.6.1
    schwimmbad==0.3.2
    dynesty==1.1
    corner==2.2.1
    mpmath==1.2.1



Authors
-------

`GIGALens` was written by `Andi Gu <andi.gu@berkeley.edu>`_.
