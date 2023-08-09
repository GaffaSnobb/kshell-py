from functools import cache
from scipy.special import comb

@cache
def n_choose_k(n, k):
    """
    NOTE: This can be replaced by a lookup table to increase speed
    further.
    """
    return comb(n, k, exact=True)