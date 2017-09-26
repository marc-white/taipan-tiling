"""
The reference for the input and output of routing is:

https://dev.aao.gov.au/forum/forum.php?thread_id=7&forum_id=4&group_id=18

Key parameters are xMicrons and yMicrons

should have an:
import taipan.core as constants, and refer to e.g.
constants.BUGPOS_MM_ORIG

No classes so we can use MPI in the future.

Questions for Nuria:
1) Are xMicrons and yMicrons in the "translation" type actually delta-x and delta-y?
we think no - but someone has to do the math.

"""

ALGORITHM_FUNCS={"nuria_favourite":route_dtheta_dy_1, "fastest_impossible":dummy_function}

def read_obsdef():
    """Read in the observing definition file, and store xy positions of all fibers
    in a simple numpy array"""
    return None
    
def write_path(ticks, initpos):
    """
    Parameters
    ----------
    ticks: list of {"xMicrons":numpy array, 
        "yMicrons":numpy array, "thetaDeg":numpy array}
        If thetaDeg is given, this assumes a rotation, otherwise it assumes 
        a translation.
    
    initpos:
        Initial positions in the same format as read_obsdef
    """
    return None
    
def ticks_dtheta_dy(dtheta,dy):
    """
    Make a ticks array for write_path based on number dbugs x nticks//2 dtheta
    and dy arrays
    
    Or - both in a cube
    """
    return None
    

    