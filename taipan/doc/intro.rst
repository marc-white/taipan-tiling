Introduction
============

This documentation describes two related Python modules for use by the
Taipan Galaxy Survey:

* ``taipan`` contains class definitions for generating, selecting and
  manipulating tile and target objects in memory, as well as the capability
  to simulate live survey operations.
* ``TaipanDB`` provides functionality for storing the inputs and outputs of
  the ``taipan`` module into a PostgreSQL database.

The two modules are intertwined; one cannot be used without the other.