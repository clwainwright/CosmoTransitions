Overview
----------------------------------------

The CosmoTransitions package is a set of python modules for calculating properties of effective potentials with one or more scalar fields. Most importantly, it can be used to find the instanton solutions which interpolate between different vacua in a given theory, allowing one to determine the probability for a vacuum transition.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install CosmoTransitions is to download it from the `Python Package Index`_ using ``pip install cosmoTransitions`` (``pip`` comes standard with Python 3 and many Python 2 distributions. It can also generally be installed with ``easy_install pip``.). If you want to install it manually, just download source code and put the 'cosmoTransitions' folder somewhere in your path. Then from a python prompt one should be able to ``import cosmoTransitions`` and run the code. For examples, look in the ``examples`` folder.

CosmoTransitions makes extensive use of numpy_ and scipy_, and the plotting functions use matplotlib_. It is recommended that users also install IPython_ for easier interactive use (IPython also contains an excellent html-based python notebook, which is very handy for organizing computational and scientific work). These packages can be installed separately (using e.g. easy_install_), or as part of a bundle (see the Anaconda_ distribution or the `Enthought Canopy`_ distribution).

CosmoTransitions was built and tested with Python v2.7.6, numpy v1.8.0, scipy v0.11.0, and matplotlib v1.2.0.

.. _Python Package Index: https://pypi.python.org/pypi
.. _numpy: http://www.numpy.org
.. _scipy: http://www.scipy.org
.. _matplotlib: http://matplotlib.org
.. _IPython: http://ipython.org
.. _easy_install: http://pythonhosted.org/setuptools/easy_install.html
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _`Enthought Canopy`: https://www.enthought.com/products/canopy/

Module overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: cosmoTransitions

There are four core modules in the package, with each subsequent module relying upon the ones that preceed it. However, the reverse is not true: lower-level modules do not rely on higher-level modules. For example, :mod:`.tunneling1D` can be used independently of all the functions which calculate instantons in multiple field dimensions and the functions which determine finite-temperature effects. The core modules are:

  - :mod:`.tunneling1D`: Calculates instantons in a single field dimension using the overshoot / undershoot method.
  - :mod:`.pathDeformation`: Calculates instantons in multiple field dimensions by first guessing a tunneling path, solving for the one-dimensional instanton constrained to that path, and then iteratively deforming the path so that it satisfies the multi-dimensional instanton equations of motion. The algorithm used here was the core insight around which CosmoTransitions took shape.
  - :mod:`.transitionFinder`: Calculates the phase structure of a potential at finite temperature (that is, the position of the minima as a function of temperature), and finds the bubble nucleation temperature(s) to transition between phases.
  - :mod:`.generic_potential`: Contains an abstract class :class:`.generic_potential.generic_potential` which can be easily subclassed to model potentials in physically interesting theories. Also contains convenience functions for interacting with :mod:`.transitionFinder` and for plotting the potential and its phases.

In addition, there a few auxiliary modules which do not rely upon any of the core modules:

  - :mod:`.helper_functions`: A collection of (mostly) stand-alone functions which are used by the core modules. This module is new.
  - :mod:`.finiteT`: Contains functions for calculates the finite-temperature contributions exactly and in different approximation regimes. Used by :mod:`.generic_potential`.
  - :mod:`.multi_field_plotting`: Contains a class to ease visualization of potentials with three or more fields. This module is new.


Change log
~~~~~~~~~~~~~~~~~~~~

Version 2.0.2
=========================

Version 2.0.2 updates CosmoTransitions to work with Python 3. It also marks the first version that's been uploaded to the `Python Package Index`_, so installation can now be as easy as ``pip install cosmoTransitions``. Numerous formatting changes were made to bring the code into closer compliance with `PEP 8`_ (the official python style guide), although all of the original un-pythonic naming conventions remain as they were.

.. _Python Package Index: https://pypi.python.org/pypi
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/


Version 2.0
=========================

CosmoTransitions version 2 is a major update from version 1.0.2. The basic structure and computational strategy of the program remain the same, but many of the functions have changed to be more modular and extensible, and new functions have been added. Therefore, version 2 is not strictly backwards-compatible with version 1, and scripts that were written to use version 1 may need some minor revision to work with version 2.

The overall changes are:

  - Much better documentation. Almost all functions now have detailed docstrings which describe their use. These can be examined interactively by running ``help(function)`` at the python prompt, or by simply looking them up on this website. This website is built using the `sphinx <http://sphinx-doc.org/>`_ documentation tool, so any future changes to the code should be automatically updated here.
  - More transparent return types. Unless otherwise noted, any function with multiple named return values returns a named tuple. This should make interactive use easier and scripts clearer to read. For example, the :meth:`~tunneling1D.SingleFieldInstanton.findProfile` method returns a *Profile1D* named tuple, so the field values along the profile can be retrieved using ``profile = instanton.findProfile(); field_vals = profile.Phi``. 
  - More rational nested calling structure. Because of it's onion-like structure, CosmoTransitions often calls functions which call functions which call functions which might have some parameter that the user wants to tweak. Previously this was handled by passing extra keyword arguments to the top level function (like this: ``foo(**kwargs)``), which often meant that the top-level function needed to know about the arguments in the bottom-level function. This is now generally handled by passing in whole dictionaries to the top-level function (without the two asterisks). For example, if when calling :meth:`~generic_potential.generic_potential.findAllTransitions`, one wishes to change the accuracy in the field *phi* used to calculate the instantons, one can call 

    >>> model.findAllTransitions(tunnelFromPhase_args=dict(
    ...                          fullTunneling_params=dict(
    ...                          tunneling_findProfile_params=dict(
    ...                          phitol=new_phitol_value))))

    This is verbose, granted, but relatively unambiguous.
  - More pythonic use of exceptions. Exceptions are much better than error codes for both debugging and general code readability. When the code encounters an unexpected value (for example, when the metastable minimum is lower than the supposedly stable minimum), it should now report an error immediately rather than producing a seemingly unrelated error later on.
  - Syntax style changes. All tabs have been converted to spaces in keeping with the official python style guide, and lines have for the most part been shortened to 80 or fewer characters.

What follows are some of the more notable specific changes, organized by module:

  - :mod:`.tunneling1D`

    - The *bubbleProfile* class has been renamed :class:`~tunneling1D.SingleFieldInstanton`, and a new class :class:`~tunneling1D.WallWithConstFriction` has been added.
    - The radial scale is now set in its own function, and is set by the frequency of oscillations about the barrier's maximum.
    - Derivatives are now calculated to fourth order in ``phi_eps``.
    - The initial guess for the overshoot / undershoot method now defaults to the bottom of the potential barrier, rather than half way between the minima (this can be important for thick-walled bubbles).
    - The :meth:`~tunneling1D.SingleFieldInstanton.exactSolution` method now finds the exact solution about the point of interest (considering both *dV* and *d2V*), rather than always about the stable minimum.

  - :mod:`.pathDeformation`

    - The *Deformation* class has been renamed :class:`~pathDeformation.Deformation_Spline`, and a new :class:`~pathDeformation.Deformation_Points` class has been added. The latter does not use a spline to approximate the path, and may be faster in certain circumstances (but slower in others). It is a simpler implementation, in any case.
    - Fixed a bug in the :meth:`~pathDeformation.Deformation_Points.step` method which caused errors for thick-walled bubbles.
    - Added a :class:`~pathDeformation.SplinePath` class which encapsulates information about the tunneling path between deformation steps. This is used to describe the potential along the path for use in :mod:`.tunneling1D`.
    - :func:`~pathDeformation.fullTunneling` is now a function, not a class.
    - The *criticalTunneling* and *secondOrderTransition* classes have been removed. The functionality of both are now in :mod:`.transitionFinder`.

  - :mod:`.transitionFinder`

    - Added a :class:`~transitionFinder.Phase` class which encapsulates information about a single temperature-dependent phase.
    - The :func:`~transitionFinder.traceMultiMin` function now returns a dictionary of phases, with each phase defined by a unique key.
    - The *findTransitionRegions* function has been removed.
    - The class *fullTransitions* has been replaced with the function :func:`~transitionFinder.findAllTransitions`, which has a somewhat more streamlined algorithm and interface.

  - :mod:`.generic_potential`

    - Derivatives now default to fourth-order error in ``x_eps``, and are calculated using classes :class:`~helper_functions.gradientFunction` and :class:`~helper_functions.hessianFunction`.
    - The temperature scale is now set solely by ``self.Tmax``. This avoids errors when there is a tree-level barrier and ``self.T0 == 0``.
    - :meth:`~generic_potential.generic_potential.forbidPhaseCrit` is now a proper class method rather than a lambda function.
    - :meth:`~generic_potential.generic_potential.findAllTransitions` has somewhat different output matching the changes in :mod:`.transitionFinder`.


To-do list
~~~~~~~~~~~~~~~~~~

I still need to do a better job of testing the whole package, particularly the :mod:`.transitionFinder` code. I haven't tested any edge cases there yet.

Additionally:

.. todolist::
