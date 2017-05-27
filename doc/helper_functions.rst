helper_functions.py
----------------------------------------

.. automodule:: cosmoTransitions.helper_functions
    

Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: setDefaultArgs
.. autofunction:: monotonicIndices
.. autofunction:: clampVal

Numerical integration
~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: IntegrationError
.. autofunction:: rkqs
.. autofunction:: rkqs2

Numerical derivatives
~~~~~~~~~~~~~~~~~~~~~

The *derivij()* functions accept arrays as input and return arrays as output.
In contrast, :class:`gradientFunction` and :class:hessianFunction` accept
functions as input and return callable class instances (essentially functions)
as output. The returned functions can then be used to find derivatives.

.. autofunction:: deriv14
.. autofunction:: deriv14_const_dx
.. autofunction:: deriv1n
.. autofunction:: deriv23
.. autofunction:: deriv23_const_dx

.. autoclass:: gradientFunction
    :members:
    :special-members:

.. autoclass:: hessianFunction
    :members:
    :special-members:

Two-point interpolation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: makeInterpFuncs
.. autoclass:: cubicInterpFunction


Spline interpolation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: Nbspl
.. autofunction:: Nbspld1
.. autofunction:: Nbspld2

