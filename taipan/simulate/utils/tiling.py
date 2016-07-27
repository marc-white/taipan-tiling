# Utility functions for the simulator to ease tiling operations

import taipan.core as tp
import taipan.tiling as tl


def retile_fields(field_list, tiles_per_field=1):
    """
    Re-tile the fields passed.

    Parameters
    ----------
    field_list:
        A list of fields to be re-tiled. Should be a list of field IDs.
        ValueError will be thrown if any of the passed IDs don't exist in the
        database.
    tiles_per_field:
        Optional int, denoting how many tiles to generate per field. Defaults
        to 1.

    Returns
    -------
    Nil. Tiles are generated, pushed back to the database, and then eliminated
    from memory.
    """
    pass