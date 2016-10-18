# Generate reports on completed simulations


def generate_report(cursor):
    """
    Generate a report on the simulation from the database results.

    Parameters
    ----------
    cursor:
        psycopg2 cursor for interacting with the database.

    Returns
    -------
    Nil. Output is displayed to terminal (and can be piped to a file if this
    module is run as a script).
    """