import math, sys, time, ast
import numpy as np
from data_structures import ModelSpace, Interaction, OneBody, TwoBody
from parameters import flags
flags.debug = True

def read_interaction_file(
    path: str
):  
    """
    Translated to Python from operator_jscheme.f90 (subroutine
    read_intfile).

    Changes
    -------
    The two integer arrays kfin and kini are removed in favor of
    directly using the values they contain.
    kfin = (no. proton orbitals, no. proton + neutron orbitals)
    kini = (1, no. proton orbitals + 1)

    norb, jorb, etc. are saved as a named tuple (ModelSpace) of arrays
    instead of being individual arrays.

    Parameters
    ----------
    path : str
        Path to interaction file (.snt file).
    """
    # n_core_nucleons: np.ndarray[int, int] = np.zeros(2, dtype=int)  # Protons, neutrons.
    read_interaction_time = time.perf_counter()

    with open(path, "r") as infile:
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

            line_split = [ast.literal_eval(elem) for elem in line.split()]

            if not all(isinstance(elem, int) for elem in line_split):
                msg = "The number of j orbitals and core nucleons should only be integers!"
                msg += " Might be due to unexpected snt file structure."
                raise TypeError(msg)

            n_proton_orbitals, n_neutron_orbitals = line_split[:2]
            n_orbitals = n_proton_orbitals + n_neutron_orbitals

            # n_core_nucleons[:] = line_split[2:]
            n_core_protons, n_core_neutrons = line_split[2:]
            
            model_space = ModelSpace(
                n = np.zeros(n_orbitals, dtype=int),
                l = np.zeros(n_orbitals, dtype=int),
                j = np.zeros(n_orbitals, dtype=int),
                isospin = np.zeros(n_orbitals, dtype=int),
                parity = np.zeros(n_orbitals, dtype=int),
                n_proton_orbitals = n_proton_orbitals,
                n_neutron_orbitals = n_neutron_orbitals,
                n_orbitals = n_orbitals
            )
            break

        for i in range(model_space.n_orbitals):
            """
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
            line_split = [ast.literal_eval(elem) for elem in infile.readline().split()[1:5]]
            
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
            
            n_onebody_elements, method_onebody = [ast.literal_eval(elem) for elem in line.split()]
            break

        # if method_onebody == 0:
        #     dep_mass1 = 1
        # else:
        #     msg = f"dep_mass1 only implemented for method_onebody 0. Got {method_onebody}."
        #     raise NotImplementedError(msg)

        one_body = OneBody(
            orbital_0 = np.zeros(n_onebody_elements, dtype=int),
            orbital_1 = np.zeros(n_onebody_elements, dtype=int),
            reduced_matrix_element = np.zeros(n_onebody_elements, dtype=float),
            method = method_onebody,
            n_elements = n_onebody_elements
        )
        
        for i in range(one_body.n_elements):
            """
            Read the one-body orbitals and reduced matrix elements.
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
            one_body.orbital_0[i], one_body.orbital_1[i], one_body.reduced_matrix_element[i] = \
                [ast.literal_eval(elem) for elem in infile.readline().split()]
            one_body.orbital_0[i] -= 1  # Indices start at 1 in the snt files.
            one_body.orbital_1[i] -= 1

            # factor = math.sqrt(model_space.j[i] + 1)                 # NOTE: Figure out the origin of this factor.
            # one_body[orbital_0, orbital_1] = reduced_matrix_element*dep_mass1*factor    # NOTE: Figure out dep_mass1.

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
                n_twobody_elements, method_twobody, im0, pwr = [ast.literal_eval(elem) for elem in line.split()]
            except ValueError:
                msg = "Only method_twobody = 1 is implemented! For snt files"
                msg += " of method_twobody != 1, im0 and / or pwr might not"
                msg += " be present in the file, hence error when unpacking"
                msg += " the list comprehension."
                raise NotImplementedError(msg)
            break

        two_body = TwoBody(
            orbital_0 = np.zeros(n_twobody_elements, dtype=int),
            orbital_1 = np.zeros(n_twobody_elements, dtype=int),
            orbital_2 = np.zeros(n_twobody_elements, dtype=int),
            orbital_3 = np.zeros(n_twobody_elements, dtype=int),
            parity = np.zeros(n_twobody_elements, dtype=int),
            jj = np.zeros(n_twobody_elements, dtype=float),
            reduced_matrix_element = np.zeros(n_twobody_elements, dtype=float),
            method = method_twobody,
            im0 = im0,
            pwr = pwr,
            n_elements = n_twobody_elements
        )

        for i in range(two_body.n_elements):
            """
            I think jj is the coupled total angular momentum of two
            nucleons in the two-body interaction. However, consider the
            line
            (orb 0)  (orb 1)  (orb 2)  (orb 3)   (jj)      (mat elem)
               1        1        1        1       2       -0.15700000

            from usda.snt. Here jj = 2, but orbital 1 is 0d3/2. 3/2 +
            3/2 equals 3, not 2. Recall that d orbitals have l = 2.
            """
            two_body.orbital_0[i], two_body.orbital_1[i], two_body.orbital_2[i], \
            two_body.orbital_3[i], two_body.jj[i], two_body.reduced_matrix_element[i] = \
                [ast.literal_eval(elem) for elem in infile.readline().split()]
            
            two_body.orbital_0[i] -= 1     # snt indices start from 1.
            two_body.orbital_1[i] -= 1
            two_body.orbital_2[i] -= 1
            two_body.orbital_3[i] -= 1
            parity_twobody_01 = model_space.parity[two_body.orbital_0[i]]*model_space.parity[two_body.orbital_1[i]]
            parity_twobody_23 = model_space.parity[two_body.orbital_2[i]]*model_space.parity[two_body.orbital_3[i]]

            if parity_twobody_01 != parity_twobody_23:
                msg = "parity_twobody_01 != parity_twobody_23."
                msg += " This raises an error in the KSHELL Fortran code."
                msg += " I dont know why yet."
                raise RuntimeError(msg)

            two_body.parity[i] = parity_twobody_01  # _01 arbitrary choice, can just as well use _23.

        interaction = Interaction(
            one_body = one_body,
            two_body = two_body,
            model_space = model_space,
            n_core_protons = n_core_protons,
            n_core_neutrons = n_core_neutrons,
            n_core_nucleons = n_core_protons + n_core_neutrons,
        )

        print(interaction.model_space.isospin)

            # isign_01 = 1    # NOTE: Dont know what these are yet.
            # isign_23 = 1

            # if orbital_0 > orbital_1:
            #     """
            #     NOTE: Don't know why this has to happen (copied from
            #     operator_jscheme.f90). gs8.snt, jun45.snt and usda.snt
            #     do not fulfill this condition.
            #     """
            #     print("orbital_0 > orbital_1")
            #     orbital_0, orbital_1 = orbital_1, orbital_0
            #     isign_01 = (-1)**((model_space.j(orbital_0) + model_space.j(orbital_1))/2 - jj + 1)

            # if orbital_2 > orbital_3:
            #     """
            #     NOTE: Don't know why this has to happen (copied from
            #     operator_jscheme.f90). gs8.snt, jun45.snt and usda.snt
            #     do not fulfill this condition.
            #     """
            #     print("orbital_2 > orbital_3")
            #     orbital_2, orbital_3 = orbital_3, orbital_2
            #     isign_23 = (-1)**((model_space.j(orbital_2) + model_space.j(orbital_3))/2 - jj + 1)

            # reduced_matrix_element *= isign_01*isign_23
            
            # if (
            #     (orbital_1 <= model_space.n_proton_orbitals) and
            #     (orbital_3 <= model_space.n_proton_orbitals)
            # ):
            #     """
            #     If orbital 1 and 3 are smaller or equal to the number of
            #     proton orbitals, then ... (?)
            #     """
            #     ipn = 0 # Index proton neutron?

            # elif (
            #     (orbital_0 > model_space.n_proton_orbitals) and
            #     (orbital_2 > model_space.n_proton_orbitals)
            # ):
            #     """
            #     If orbital 0 and 2 are greater than the number of proton
            #     orbitals, then ... (?)
            #     """
            #     ipn = 1

            # elif (
            #     (orbital_0 <= model_space.n_proton_orbitals) and
            #     (orbital_2 <= model_space.n_proton_orbitals) and
            #     (orbital_1 >  model_space.n_proton_orbitals) and
            #     (orbital_3 >  model_space.n_proton_orbitals)
            # ):
            #     ipn = 2
            
            # else:
            #     """
            #     NOTE: I dont know the consequences of this yet.
            #     """
            #     msg = "Error in orbital indices from interaction file: "
            #     msg += f"{path}."
            #     raise RuntimeError(msg)

            # ij_01 = jcouple(jj, parity, ipn).idxrev(orbital_0, orbital_1)
            # ij_23 = jcouple(jj, parity, ipn).idxrev(orbital_2, orbital_3)

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

        read_interaction_time = time.perf_counter() - read_interaction_time
        if flags.debug:
            print("--------")
            print(f"{read_interaction_time = :.4f} s")
            print("--------")

if __name__ == "__main__":
    read_interaction_file(path="usda.snt")