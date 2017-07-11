The CosmoTransitions package is a set of python modules for calculating properties of effective potentials with one or more scalar fields. Most importantly, it can be used to find the instanton solutions which interpolate between different vacua in a given theory, allowing one to determine the probability for a vacuum transition.

For more info, please read the documentation_. The current version is available via github_.

 .. _documentation: http://clwainwright.github.io/CosmoTransitions
 .. _github: https://github.com/clwainwright/CosmoTransitions


Attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use CosmoTransitions in scholarly work, please cite `Comput. Phys. Commun. 183 (2012)`_ [`arXiv:1109.4189`_].

 .. _`arXiv:1109.4189`: http://arxiv.org/abs/1109.4189
 .. _`Comput. Phys. Commun. 183 (2012)`: http://dx.doi.org/10.1016/j.cpc.2012.04.004


Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CosmoTransitions can be installed via ``pip`` using ``pip install cosmoTransitions`` for both Python 2 and Python 3. You can also install the package manually by downloading the source and putting the 'cosmoTransitions' folder somewhere in your path.
Then from a python prompt you should be able to ``import cosmoTransitions`` and run the code. For examples, look in the ``examples`` folder.

CosmoTransitions makes extensive use of numpy_ and scipy_, and the plotting functions use matplotlib_. It is recommended that users also install IPython_ for easier interactive use (IPython also contains an excellent html-based python notebook, which is very handy for organizing computational and scientific work). These packages can be installed separately (using e.g. easy_install_), or as part of a bundle (see the Anaconda_ distribution or the `Enthought Canopy`_ distribution).

CosmoTransitions was built and tested with Python v2.7.6, numpy v1.8.0, scipy v0.11.0, and matplotlib v1.2.0.

.. _numpy: http://www.numpy.org
.. _scipy: http://www.scipy.org
.. _matplotlib: http://matplotlib.org
.. _IPython: http://ipython.org
.. _easy_install: http://pythonhosted.org/setuptools/easy_install.html
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
