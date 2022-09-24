from typing import NamedTuple
import math
import numpy as np
from ast import literal_eval

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

def main():
    n_orbitals: np.ndarray[int, int] = np.zeros(2, dtype=int)
    n_core_nucleons: np.ndarray[int, int] = np.zeros(2, dtype=int)  # Protons, neutrons.

    with open("usda.snt", "r") as infile:
        for line in infile:
            """
            Read the number of j orbitals and the number of nucleons.

            ! ...
            ! model space 
            3   3     8   8                             <---- This line!
                1     0   2   3  -1  !   1 = p 0d_ 3/2
                2     0   2   5  -1  !   2 = p 0d_ 5/2
                3     1   0   1  -1  !   3 = p 1s_ 1/2
                4     0   2   3   1  !   4 = n 0d_ 3/2
                5     0   2   5   1  !   5 = n 0d_ 5/2
                6     1   0   1   1  !   6 = n 1s_ 1/2
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue

            line_split = [literal_eval(elem) for elem in line.split()]

            if not all(isinstance(elem, int) for elem in line_split):
                msg = "The number of j orbitals and core nucleons should only be integers!"
                msg += " Might be due to unexpected snt file structure."
                raise TypeError(msg)

            n_orbitals[:] = line_split[:2]
            n_core_nucleons[:] = line_split[2:]
            break
        
        model_space = ModelSpace(
            n = np.zeros(n_orbitals[0] + n_orbitals[1], dtype=int),
            l = np.zeros(n_orbitals[0] + n_orbitals[1], dtype=int),
            j = np.zeros(n_orbitals[0] + n_orbitals[1], dtype=int),
            isospin = np.zeros(n_orbitals[0] + n_orbitals[1], dtype=int),
            parity = np.zeros(n_orbitals[0] + n_orbitals[1], dtype=int)
        )
        
        for i in range(n_orbitals[0] + n_orbitals[1]):
            """
            All orbitals. Read four elements from each line and
            convert them into ints.

            ! ...
            ! model space 
            3   3     8   8
                1     0   2   3  -1  !   1 = p 0d_ 3/2  <---- These lines!
                2     0   2   5  -1  !   2 = p 0d_ 5/2  <----
                3     1   0   1  -1  !   3 = p 1s_ 1/2  <----
                4     0   2   3   1  !   4 = n 0d_ 3/2  <----
                5     0   2   5   1  !   5 = n 0d_ 5/2  <----
                6     1   0   1   1  !   6 = n 1s_ 1/2  <----
            """
            line_split = [literal_eval(elem) for elem in infile.readline().split()[1:5]]
            
            if not all(isinstance(elem, int) for elem in line_split):
                msg = "Orbital parameters should only be integers!"
                msg += " Might be due to unexpected snt file structure."
                raise TypeError(msg)

            model_space.n[i] = line_split[0]
            model_space.l[i] = line_split[1]
            model_space.j[i] = line_split[2]
            model_space.isospin[i] = line_split[3]
            model_space.parity[i] = (-1)**model_space.l[i]

        for line in infile:
            """
            Read the number of one-body interactions and method(?).
            NOTE: Figure out what method means.
            
            ! ...
            ! num, method=,  hbar_omega
            !  i  j     <i|H(1b)|j>
            6   0                       <---- This line!
            1   1      1.97980000
            2   2     -3.94360000
            3   3     -3.06120000
            4   4      1.97980000
            5   5     -3.94360000
            6   6     -3.06120000
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue
            
            n_onebody_interactions, method_onebody = [literal_eval(elem) for elem in line.split()]
            break

        if method_onebody == 0:
            dep_mass1 = 1
        else:
            msg = f"dep_mass1 only implemented for method_onebody 0. Got {method_onebody}."
            raise NotImplementedError(msg)

        one_body = np.zeros((n_onebody_interactions, n_onebody_interactions), dtype=np.float64)    # TODO: This does not need to be a matrix since only diagonals will be non-zero.
        
        # for i in range(n_onebody_interactions):
        for i in range(n_orbitals[0] + n_orbitals[1]):
            """
            Example
            -------
            ! ...
            ! num, method=,  hbar_omega
            !  i  j     <i|H(1b)|j>
            6   0
            1   1      1.97980000   <---- These lines!
            2   2     -3.94360000   <---- 
            3   3     -3.06120000   <---- 
            4   4      1.97980000   <----
            5   5     -3.94360000   <----
            6   6     -3.06120000   <----
            """
            row, col, reduced_matrix_element = [literal_eval(elem) for elem in infile.readline().split()]
            row -= 1    # Indices start at 1 in the snt files.
            col -= 1
            factor = math.sqrt(model_space.j[i] + 1)                 # NOTE: Figure out the origin of this factor.
            one_body[row, col] = reduced_matrix_element*dep_mass1*factor    # NOTE: Figure out dep_mass1.

        for line in infile:
            """
            Read the number of two-body interactions, method(?), im0(?)
            and pwr(?).
            NOTE: Figure out what im0 and pwr means.

            Example
            -------
            ! ...
            ! TBME
                    158   1  18 -0.300000           <---- This line!
            1   1   1   1    0       -1.50500000
            1   1   1   1    2       -0.15700000
            1   1   1   2    2        0.52080000
            1   1   1   3    2        0.13680000
            1   1   2   2    0       -3.56930000
            ...
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue
            
            try:
                n_twobody_interactions, method_twobody, im0, pwr = [literal_eval(elem) for elem in line.split()]
            except ValueError:
                msg = "Only method_twobody = 1 is implemented!"
                raise NotImplementedError(msg)
            break

        tmp = np.zeros(n_twobody_interactions)

        for i in range(n_twobody_interactions):
            """
            I think jj is the coupled total angular momentum of two
            nucleons in the two-body interaction.
            """
            k1, k2, k3, k4, jj, reduced_matrix_element = [literal_eval(elem) for elem in infile.readline().split()]
            k1 -= 1     # snt indices start from 1.
            k2 -= 1
            k3 -= 1
            k4 -= 1
            parity_twobody_12 = model_space.parity[k1]*model_space.parity[k2]
            parity_twobody_34 = model_space.parity[k3]*model_space.parity[k4]

            tmp[i] = jj

        # print(model_space.parity)
        # print(f"{np.unique(tmp) = }")
        # print(one_body)
        # print(f"{type(one_body) = }")
        # print(f"{type(one_body[0, 0]) = }")
        # print(f"{one_body[1, 1] = }")

            # if not all(isinstance(elem, int) for elem in line_split):
            #     msg = "Orbital parameters should only be integers!"
            #     msg += " Might be due to unexpected snt file structure."
            #     raise TypeError(msg)

            # n_onebody_interactions = 1
        # dep = (dble(mass)/dble(im0))**pwr

if __name__ == "__main__":
    main()