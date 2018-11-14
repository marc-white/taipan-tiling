Preparing ``taipan`` for use
============================

There are two main steps required to prepare ``taipan`` to be run:

1. Ingest catalogues into the database (and related computations);
2. Make an initial tiling of the sky.

Ingesting catalogues
--------------------

The initial ingest of data into the database is handled by a script found in
``taipandb/resources``. There is one script per code version; you should
run the most recent script, which is `v_0_0_1.py`.

Before running this script, all catalogues should be placed into a single
directory. This directory can be passed to the ingest script as a command-line
argument. For further details of what is required, see the documentation
for :any:`taipandb.resources.v_0_0_1.execute`.

.. warning::
    There is currently no standardized format for Taipan catalogues. Currently,
    each catalogue type has a specific format requirement based on how that
    catalogue was created. You should review the code within the modules of
    ``taipandb.resources.stable.ingest`` to see what is required of each
    catalogue type.

Running an initial tiling
-------------------------
