Preparing ``taipan`` for use
============================

There are two main steps required to prepare ``taipan`` to be run:

1. Ingest catalogues into the database (and related computations);
2. Make an initial tiling of the sky.

Ingesting catalogues
--------------------

The initial ingest of data into the database is handled by a script found in
``taipandb/resources``. There is one script per code version; you should
run the most recent script, which is symlinked as ``stable_load.py``.

Before running this script, all catalogues should be placed into a single
directory. This directory can be passed to the ingest script as a command-line
argument. For further details of what is required, see the documentation of
the ``update`` function within ``stable_load.py``.

.. warning::
    There is currently no standardized format for Taipan catalogues. Currently,
    each catalogue type has a specific format requirement based on how that
    catalogue was created. You should review the code within the modules of
    ``taipandb.resources.stable.ingest`` to see what is required of each
    catalogue type.

.. warning::
    The execution of ``stable_load.py`` takes 24-36 hours. It is *strongly*
    recommended that the script is run from the command line, and you schedule
    the job using the ``cron`` or ``at`` Unix utilities.

The name of the science target catalogue, and the date range required of
the setup, can be set as command-line arguments when running ``stable_load.py``.
The names of all other catalogues are fixed in code, and will need to be
modified there if required. All catalogues should be placed in the same
directory for ``stable_load.py`` to read from.

The most time-consuming part of ingesting catalogues is the computation of
the necessary :any:`Almanac` objects. If you have already computed these objects
and saved them to disk, using the :any:`Almanac.save` function, the Almanacs
can be ingested to the database from disk instead. This will save
significant time. To do this:

- Make sure that the Almanacs *exactly* match the Almanac parameters within
  ``stable_load.py`` (i.e. sky position, time resolution), otherwise the
  preparation script will re-compute the Almanacs.
- All the Almanacs need to be stored in the same directory, which can be
  passed to ``stable_load.py`` as a command-line argument.

Preparing an initial tiling
---------------------------

Initial tilings are handled by :any:`taipan.ops.prepare`. After ingesting the
catalogues, :any:`taipan.ops.prepare.do_initial_tile` should be run *once*
to generate an initial tiling. At the successful completion of this function,
your setup is ready to start live tiling!
