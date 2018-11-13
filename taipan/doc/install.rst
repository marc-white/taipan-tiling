Installation & Setup
====================

Code Installation
-----------------

``taipan`` and ``TaipanDB`` are not provided as installable Python packages;
rather, the source code is directly available. Simply place the
``tiling-code/taipan`` and
``TaipanDB/taipandb`` directories somewhere accessible to the ``$PYTHONPATH``
variable in your computing environment.

Database Setup
--------------

``taipan`` and ``TaipanDB`` require a back-end PostgreSQL database in order to
function. ``TaipanDB`` will setup database tables and relationships for you.
However, you need to have stood up a PSQL database yourself in the first
instance, and make sure it's accessible to ``TaipanDB``. Instructions for this
can be found in the documentation for ``TaipanDB``.

The minimum required version of PostgreSQL is 9.6. This is necessary to
allow for table partitioning and effective multi-threading.

To prepare your
system to use ``TaipanDB``:

#. Make sure PostgreSQL is installed on your system. If not, it can be
   downloaded from the `PostgreSQL website <https://www.postgresql.org/>`_.
#. Make the cusotmisation tweaks recommended in Database Customisation.
#. Create a database within PSQL for the use of ``TaipanDB``. This may require
   you to set up a PSQL user as well; see the
   `PSQL documentation <https://www.postgresql.org/docs/9.0/static/app-createdb.html>`_
   for details.
   Be sure to note the host address (normally ``localhost`` if on your own
   machine), database name, associated user name, and password (if
   any).
#. Place the database connection details into a file called ``config.json``, in
   the same location as the ``taipan`` and ``taipandb`` directories. The JSON
   object should look like this::

        {  # Change all values as necessary
          "host": "localhost",
          "port": 5432,
          "user": "uname",
          "password": "pwd",
          "database": "taipandb"
        }

Your ``taipan`` and ``taipandb`` modules should now be able to connect with
and manipulate the PSQL database.

Database Customisation
^^^^^^^^^^^^^^^^^^^^^^

The following database settings have been found to be necessary to get
the system to work properly:

- ``max_stack_depth=7680kB``
- ``shared_buffers=500MB``
- ``work_mem=1GB``
- ``max_locks_per_transaction=512``
- ``constraint_exclusion=partition``

These settings enable the large queries, table partitioning, and multi-threading
that are built in to ``taipandb``.
