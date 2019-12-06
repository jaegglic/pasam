================================
User Guide of the PaSam Package
================================
Let us provide some examples and (hopefully) some helpful guidance for the
usage of the sub-packages :ref:`lattice_reference`, **TODO**

.. _lattice_reference:

Lattice
-------
The following description is dedicated to uniform lattices in two- and three-
dimensions and the real valued maps that can be associated with. A *grid* (or
a *lattice*) is defined to be a tensor product of spacial distributed points
.. math::

   (x_1, \dots, x_n), (y_1, \dots, y_m)

in two dimensions and
.. math::

   (x_1, \dots, x_n), (y_1, \dots, y_m), (z_1, \dots, z_r)

in three dimensions. We assume :math:`x_i < x_{i+1}`, :math:`y_j < x_{j+1}`,
and :math:`z_k < z_{k+1}` for all :math:`i=1,\dots,n-1`, :math:`j=1,\dots,m-1`,
and :math:`k=1,\dots,r-1`.

