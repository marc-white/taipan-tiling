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

For details on how to setup your database, see the :any:`taipandb`
documentation.

