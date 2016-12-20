Installation & Setup
====================

Code Installation
-----------------

``taipan`` and ``TaipanDB`` are not provided as installable Python packages;
rather, the source code is directly available. Simply place the
``tiling-code/taipan`` and
``TaipanDB/src`` directories somewhere accessible to the ``$PYTHONPATH``
variable in your computing environment.

Database Setup
--------------

``taipan`` and ``TaipanDB`` require a back-end PostgreSQL database in order to
function. ``TaipanDB`` will setup database tables and relationships for you.
However, you need to have stood up a PSQL database yourself in the first
instance, and make sure it's accessible to ``TaipanDB``. To prepare your
system to use ``TaipanDB``:

#. Make sure PostgreSQL is installed on your system. If not, it can be
   downloaded from the `PostgreSQL website <https://www.postgresql.org/>`_.
#. Create a database within PSQL for the use of ``TaipanDB``. This may require
   you to set up a PSQL user as well; see the
   `PSQL documentation <https://www.postgresql.org/docs/9.0/static/app-createdb.html>`_
   for details.
   Be sure to note the host address (normally ``localhost`` if on your own
   machine), database name, associated user name, and password (if
   any).
#. Reconfigure