pathDeformation.py
----------------------------------------

.. automodule:: pathDeformation

Deformation classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: DeformationError

.. autoclass:: Deformation_Spline
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: Deformation_Points
    :members:
    :undoc-members:
    :show-inheritance:    

SplinePath
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SplinePath
    :members:
    :undoc-members:
    :show-inheritance:

fullTunneling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fullTunneling(path_pts, V, dV, maxiter=20, fixEndCutoff=.03, save_all_steps=False, verbose=False, V_spline_samples=100, tunneling_class=tunneling1D.SingleFieldInstanton, tunneling_init_params={}, tunneling_findProfile_params={}, deformation_class=Deformation_Spline, deformation_init_params={}, deformation_deform_params={})

 