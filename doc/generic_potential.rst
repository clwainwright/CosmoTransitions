generic_potential.py
----------------------------------------

.. automodule:: generic_potential
    :members:
    :undoc-members:
    :show-inheritance:

Example subclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows typical usage for subclassing :class:`generic_potential`:

.. literalinclude:: ../test/testModel1.py

Running model1 should produce following output:

 .. literalinclude:: ../test/testModel1_output.rst

There is a second-order transition at *T=222.9* (no units are given, but this would probably be in GeV) in which the high-temperature phase disappears and a mid-temperature phase starts in the fourth quadrant. At *T=117.2*, the low-temperature phase appears in the first quadrant, and by *T=109.4* the two phases are degenerate (they have equal pressure). There is considerable super-cooling, and by thermal tunneling does not occur until *T=80.0*. The mid-temperature phase then disappears at *T=77.6*.

These next few plots can be output by ``testModel1.makePlots()``. The first plot shows the different phases as they change with temperature:

.. image:: ../test/model1_phases.png

This allows one to see at a glance what the overlap between the different phases is, and how big a jump there is between the phases. This plot comes from :meth:`generic_potential.plotPhasesPhi`.

The next figure shows contour levels at *T=0* and at the two transition temperatures. The black line in the middle plot is the tunneling direction. Each contour is produced with :meth:`generic_potential.plot2d`.

.. image:: ../test/model1_contours.png

The final figure shows the bubble wall profile during the first-order phase transition. It is extremely thick-walled, so the center of the bubble is far away from the absolute minimum.

.. image:: ../test/model1_profile.png