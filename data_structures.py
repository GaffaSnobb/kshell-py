from typing import NamedTuple
import numpy as np

class ModelSpace(NamedTuple):
    """
    For containing model space orbital information.

    Example
    -------
    3   3     8   8
        1     0   2   3  -1  !   1 = p 0d_ 3/2
        2     0   2   5  -1  !   2 = p 0d_ 5/2
        3     1   0   1  -1  !   3 = p 1s_ 1/2
        4     0   2   3   1  !   4 = n 0d_ 3/2
        5     0   2   5   1  !   5 = n 0d_ 5/2
        6     1   0   1   1  !   6 = n 1s_ 1/2

    Attributes
    ----------
    n : np.ndarray
        The lj index of the orbital. Counts the number of orbitals with
        each lj combinations.

    l : np.ndarray
        The orbital angular momentum (l) of the orbital.

    j : np.ndarray
        The total angular momentum (j) of the orbital, multiplied by 2
        to circumvent fractions.
    
    isospin : np.ndarray
        The isospin of the orbital. -1 for protons, +1 for neutrons.

    parity : np.ndarray
        The parity of the orbital.
    """
    n: np.ndarray
    l: np.ndarray
    j: np.ndarray
    isospin: np.ndarray
    parity: np.ndarray

class Interaction(NamedTuple):
    """
    For containing one-body and two-body matrix elements of the
    interaction.

    Example
    -------
    ! ...
    ! interaction
    ! num, method=,  hbar_omega
    !  i  j     <i|H(1b)|j>
    6   0
    1   1      1.97980000                   <---- one-body
    2   2     -3.94360000                   <----
    3   3     -3.06120000                   <----
    4   4      1.97980000                   <----
    5   5     -3.94360000                   <----
    6   6     -3.06120000                   <----
    ! TBME
            158   1  18 -0.300000
    1   1   1   1    0       -1.50500000    <---- two-body
    1   1   1   1    2       -0.15700000    <----
    1   1   1   2    2        0.52080000    <----
    1   1   1   3    2        0.13680000    <----
    ...                                     ...
    """
    one_body: np.ndarray    # TODO: This does not need to be a matrix since only diagonals will be non-zero.
    two_body: np.ndarray