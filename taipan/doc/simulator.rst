Simulator API
=============

The simulator code is primarily designed to run simulations of the
Taipan survey to inform target selection and prioritization algorithms. Elements
of the code will be re-purposed to form the scheduler for live survey
operations.

Due to this, the simulator API closely interfaces with, and depends upon,
the :any:`TaipanDB` module for managing simulator/survey information.
It is strongly
recommended that the reader examines the documentation for both packages in
tandem.

.. toctree::
   :maxdepth: 2

   simulator_highlevel
   simulator_lowlevel

