from typing import Union
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class ModelSpace:
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

    jz : np.ndarray
        Possible jz values per j value.
    
    isospin : np.ndarray
        The isospin of the orbital. -1 for protons, +1 for neutrons.

    parity : np.ndarray
        The parity of the orbital.

    n_proton_orbitals : int
        The number of proton orbitals in the model space.

    n_neutron_orbitals : int
        The number of neutron orbitals in the model space.

    n_orbitals : int
        The total number of orbitals in the model space.

    n_orbital_degeneracy : np.ndarray
        The number of possible jz projections per orbital.

    max_proton_j_couple : Union[int, None]
        The maximum j value two protons in the highest angular momentum
        orbital can couple to. Remember that identical nucleons must
        have different jz values, thus the maximum j_couple value for
        two protons in a 3/2 orbital is not 3/2 + 3/2, but 3/2 + 1/2.

    max_neutron_j_couple : Union[int, None]
        The maximum j value two neutrons in the highest angular
        momentum orbital can couple to. Remember that identical nucleons
        must have different jz values, thus the maximum j_couple value
        for two neutronss in a 3/2 orbital is not 3/2 + 3/2, but 3/2 +
        1/2.

    max_j_couple : Union[int, None]
        The maximum j value two nucleons (non-identical nucleons if the
        model space supports it) in the highest angular momentum orbital
        can couple to. Remember that non-identical nucleons can have the
        same jz value, thus the maximum j_couple value for two
        non-identical nucleons in a 3/2 orbital is 3/2 + 3/2.

    """
    n: np.ndarray
    l: np.ndarray
    j: np.ndarray
    jz: np.ndarray
    isospin: np.ndarray
    parity: np.ndarray
    
    n_orbitals: int
    n_orbital_degeneracy: np.ndarray
    
    max_j_couple: Union[int, None] = None
    max_proton_j_couple: Union[int, None] = None
    max_neutron_j_couple: Union[int, None] = None
    
    n_proton_orbitals: Union[int, None] = None
    n_neutron_orbitals: Union[int, None] = None

@dataclass(slots=True)
class OneBody:
    orbital_0: np.ndarray   # Might remove these since they simply are 0, 1, 2, ...
    orbital_1: np.ndarray   # Might remove these since they simply are 0, 1, 2, ...
    reduced_matrix_element: np.ndarray
    method: int
    n_elements: int

@dataclass(slots=True)
class TwoBody:
    orbital_0: np.ndarray
    orbital_1: np.ndarray
    orbital_2: np.ndarray
    orbital_3: np.ndarray
    parity: np.ndarray
    jj: np.ndarray
    reduced_matrix_element: np.ndarray
    method: int
    im0: int
    pwr: float
    n_elements: int

@dataclass(slots=True)
class Interaction:
    """
    For containing one-body and two-body matrix elements of the
    interaction, as well as the model space parameters.

    Example
    -------
    ! ...
    ! interaction
    ! num, method=,  hbar_omega
    !  i  j     <i|H(1b)|j>
    (n) (method)
     6      0
    (orbital 0) (orbital 1) (mat. element)
        1           1        1.97980000        <---- one-body
        2           2       -3.94360000        <----
        3           3       -3.06120000        <----
        4           4        1.97980000        <----
        5           5       -3.94360000        <----
        6           6       -3.06120000        <----
    ! TBME
    (n) (method) (im0) (pwr)
    158    1      18   -0.300000
    (orbital 0) (orbital 1) (orbital 2) (orbital 3) (jj) (mat. element)
        1           1           1           1        0    -1.50500000    <---- two-body
        1           1           1           1        2    -0.15700000    <----
        1           1           1           2        2     0.52080000    <----
        1           1           1           3        2     0.13680000    <----
    ...
    """    
    one_body: OneBody
    two_body: TwoBody
    model_space: ModelSpace
    proton_model_space: ModelSpace
    neutron_model_space: ModelSpace

    n_core_protons: int
    n_core_neutrons: int
    n_core_nucleons: int

@dataclass(slots=True)
class OperatorJ:
    """
    Save the treated matrix elements. The raw matrix elements are
    stored in the Interaction dataclass (.one_body and .two_body).

    Attributes
    ----------
    two_body : np.ndarray
        3D array of indices [j_couple, parity, proton_neutron_idx]. Each
        entry in the array is a 2D array with reduced matrix elements.
    """
    one_body_reduced_matrix_element: np.ndarray # 1D array.
    two_body_reduced_matrix_element: np.ndarray # 3D array of indices [j_couple, parity, proton_neutron_idx].

@dataclass(slots=True)
class CouplingIndices:
    """
    Description copy-paste from type index_jcouple in
    operator_jscheme.f90:
    ! 
    ! index for the coupling of [a+_i*a+_j]^J for given J, iprty (parity) 
    ! and ipn (p-p, n-n or p-n)
    !
    ! n : the number of (i,j)
    ! idx(2,n) : index to i and j
    ! idxrev : reverse index (i,j) to n
    !

    Attributes
    ----------
    n_couplings : Union[int, None]
        The number of couplings for a given [jcpl, parity_idx,
        proton_neutron_idx].
    
    idx : Union[np.ndarray, None]
        The indices of the orbitals that couple to each other.

    idxrev : Union[np.ndarray, None]
        idx_reverse is organised like this for j = 0, parity =
        +1 (using usda.snt as an example):

            O1  O2  O3
        O1  1   0   0
        O2  0   2   0
        O3  0   0   3

        meaning that the zero angular momentum coupling only
        happens within the same orbital and that the coupling
        between orbital 1 and orbital 1 was counted first,
        orbital 2 to orbital 2 was counted second, etc.
    """
    n_couplings: Union[int, None] = None
    idx: Union[np.ndarray, None] = None         # 2D integer array.
    idx_reverse: Union[np.ndarray, None] = None # 2D integer array.

@dataclass(slots=True)
class Debug:
    """
    Values which are stored only for unit testing and debug.
    """
    ij_01: Union[np.ndarray, None] = None
    ij_23: Union[np.ndarray, None] = None
    proton_neutron_idx: Union[np.ndarray, None] = None

@dataclass(slots=True)
class Flags:
    """
    Package-wide flags.
    """
    debug: bool = False