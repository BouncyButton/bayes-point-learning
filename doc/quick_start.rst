#####################################
Quick Start with the bpllib package
#####################################

This package provides an implementation of a Bayes Point Classifier.

Installation
===================================================


1. Install from the repository
-------------------------------------

To install this package, run the following command::

    $ pip install git+https://github.com/BouncyButton/bayes-point-learning.git

1. Clone and install from the repository
-------------------------------------

To run this package, you need to clone the ``bpllib`` repository::

    $ git clone https://github.com/BouncyButton/bayes-point-learning.git

To install this package, run the following command, after navigating to the ``bpllib`` directory::

    $ pip install -e .

This should automatically install all the dependencies. If needed, you can install them manually::

    $ pip install -r requirements.txt


2. Development
-------------------------------------------

.. _check_estimator: http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator
.. _`Contributor's Guide`: http://scikit-learn.org/stable/developers/
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
.. _NumPyDoc: https://github.com/numpy/numpydoc
.. _doctests: https://docs.python.org/3/library/doctest.html

We follow the scikit-learn development workflow, as found in the skltemplate repository.
Hence, estimators need to pass the check_estimator_ test to be scikit-learn compatible. You can
refer to the :ref:`User Guide <user_guide>` to help you create a compatible
scikit-learn estimator.

In any case, developers should endeavor to adhere to scikit-learn's
`Contributor's Guide`_ which promotes the use of:

* algorithm-specific unit tests, in addition to ``check_estimator``'s common
  tests;
* PEP8_-compliant code;
* a clearly documented API using NumpyDoc_ and PEP257_-compliant docstrings;
* references to relevant scientific literature in standard citation formats;
* doctests_ to provide succinct usage examples;
* standalone examples to illustrate the usage, model visualisation, and
  benefits/benchmarks of particular algorithms;
* efficient code when the need for optimization is supported by benchmarks.

3. Edit the documentation
-------------------------

.. _Sphinx: http://www.sphinx-doc.org/en/stable/

The documentation is created using Sphinx_. In addition, the examples are
created using ``sphinx-gallery``. Therefore, to generate locally the
documentation, you are required to install the following packages::

    $ pip install sphinx sphinx-gallery sphinx_rtd_theme matplotlib numpydoc pillow

The documentation is made of:

* a home page, ``doc/index.rst``;
* an API documentation, ``doc/api.rst`` in which you should add all public
  objects for which the docstring should be exposed publicly.
* a User Guide documentation, ``doc/user_guide.rst``, containing the narrative
  documentation of your package, to give as much intuition as possible to your
  users.
* examples which are created in the `examples/` folder. Each example
  illustrates some usage of the package. the example file name should start by
  `plot_*.py`.

The documentation is built with the following commands::

    $ cd doc
    $ make html

4. Setup the continuous integration
-----------------------------------

The project template already contains configuration files of the continuous
integration system. Basically, the following systems are set:

* Travis CI is used to test the package in Linux. You need to activate Travis
  CI for your own repository. Refer to the Travis CI documentation.
* AppVeyor is used to test the package in Windows. You need to activate
  AppVeyor for your own repository. Refer to the AppVeyor documentation.
* Circle CI is used to check if the documentation is generated properly. You
  need to activate Circle CI for your own repository. Refer to the Circle CI
  documentation.
* ReadTheDocs is used to build and host the documentation. You need to activate
  ReadTheDocs for your own repository. Refer to the ReadTheDocs documentation.
* CodeCov for tracking the code coverage of the package. You need to activate
  CodeCov for you own repository.
* PEP8Speaks for automatically checking the PEP8 compliance of your project for
  each Pull Request.

Publish your package
====================

.. _PyPi: https://packaging.python.org/tutorials/packaging-projects/
.. _conda-foge: https://conda-forge.org/

You can make your package available through PyPi_ and conda-forge_. Refer to
the associated documentation to be able to upload your packages such that
it will be installable with ``pip`` and ``conda``. Once published, it will
be possible to install your package with the following commands::

    $ pip install your-scikit-learn-contribution
    $ conda install -c conda-forge your-scikit-learn-contribution
