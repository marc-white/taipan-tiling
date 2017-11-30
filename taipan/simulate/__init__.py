"""
:any:`taipan.simulate` contains the code for running Taipan survey simulations.

This module is structured as follows:

- Top-level submodules beginning with ``fullsurvey`` are the code that
  actually runs the simulations. :any:`fullsurvey` contains the majority of
  the simulator logic; other simulator run modules
  (e.g. :any:`fullsurvey_baseline`) will typically override the specific
  behaviours they need to, and inherit everything else from :any:`fullsurvey`.
- The submodules :any:`simulate.simulate` and :any:`simulate.utils` contain
  helper functions used during simulator runs. These should have enough
  argument-based options that they do not need modification between
  different simulation types.
"""