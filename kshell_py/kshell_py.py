import math, sys, time, ast
import numpy as np
from parameters import flags
from data_structures import (
    ModelSpace, Interaction, OneBody, TwoBody, OperatorJ, CouplingIndices
)

def read_interaction_file(
    path: str
) -> Interaction:
    """
    Read an interaction file (.snt) and store all relevant parameters
    as an Interaction named tuple. Part of Python translation of
    operator_jscheme.f90 (subroutine read_intfile).

    Changes
    -------
    The two integer arrays kfin and kini are removed in favor of
    directly using the values they contain.
    kfin = (no. proton orbitals, no. proton + neutron orbitals)
    kini = (1, no. proton orbitals + 1)

    norb, jorb, etc. are saved as a dataclass (ModelSpace) of arrays
    instead of being individual arrays.

    Parameters
    ----------
    path : str
        Path to interaction file (.snt file).
    """
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
            n_core_protons, n_core_neutrons = line_split[2:]
            
            model_space = ModelSpace(
                n = np.zeros(n_orbitals, dtype=int),
                l = np.zeros(n_orbitals, dtype=int),
                j = np.zeros(n_orbitals, dtype=int),
                jz = np.zeros(n_orbitals, dtype=np.ndarray),
                isospin = np.zeros(n_orbitals, dtype=int),
                parity = np.zeros(n_orbitals, dtype=int),
                n_proton_orbitals = n_proton_orbitals,
                n_neutron_orbitals = n_neutron_orbitals,
                n_orbitals = n_orbitals,
                n_orbital_degeneracy = np.zeros(n_orbitals, dtype=int)
            )
            break
        
        jz_max = -np.inf
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

            model_space.n_orbital_degeneracy[i] = model_space.j[i] + 1  # Degeneracy is 2*j + 1, but the j values are stored as 2*j already.
            model_space.jz[i] = np.arange(-model_space.j[i], model_space.j[i] + 1, 2, dtype=int)    # From -j to j in integer steps.
            # model_space.jz_max = max(jz_max, max(model_space.jz[i]))

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

            if one_body.orbital_0[i] != one_body.orbital_1[i]:
                msg = "One-body indices are not identical!"
                msg += f" Got {one_body.orbital_0[i] = }, {one_body.orbital_1[i] = }"
                msg += f" on line {i}. (Don't know how to treat this yet!)"
                raise NotImplementedError(msg)

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

        proton_model_space = ModelSpace(
            n = model_space.n[:model_space.n_proton_orbitals],
            l = model_space.l[:model_space.n_proton_orbitals],
            j = model_space.j[:model_space.n_proton_orbitals],
            jz = model_space.jz[:model_space.n_proton_orbitals],
            isospin = model_space.isospin[:model_space.n_proton_orbitals], 
            parity = model_space.parity[:model_space.n_proton_orbitals],
            n_orbitals = model_space.n_proton_orbitals,
            n_orbital_degeneracy = model_space.n_orbital_degeneracy[:model_space.n_proton_orbitals]
        )

        neutron_model_space = ModelSpace(
            n = model_space.n[model_space.n_proton_orbitals:],
            l = model_space.l[model_space.n_proton_orbitals:],
            j = model_space.j[model_space.n_proton_orbitals:],
            jz = model_space.jz[model_space.n_proton_orbitals:],
            isospin = model_space.isospin[model_space.n_proton_orbitals:],
            parity = model_space.parity[model_space.n_proton_orbitals:],
            n_orbitals = model_space.n_neutron_orbitals,
            n_orbital_degeneracy = model_space.n_orbital_degeneracy[model_space.n_proton_orbitals:]
        )

        interaction = Interaction(
            one_body = one_body,
            two_body = two_body,
            model_space = model_space,
            n_core_protons = n_core_protons,
            n_core_neutrons = n_core_neutrons,
            n_core_nucleons = n_core_protons + n_core_neutrons,
            proton_model_space = proton_model_space,
            neutron_model_space = neutron_model_space
        )

        read_interaction_time = time.perf_counter() - read_interaction_time
        if flags.debug:
            print(f"{read_interaction_time = :.4f} s")

        return interaction

def initialise_operator_j_scheme(
    interaction: Interaction
) -> CouplingIndices:
    """
    Initialise the j-scheme operator.
    """
    n_parities: int = 2
    n_proton_neutron: int = 3   # Proton, neutron, both.
    
    proton_ms: ModelSpace = interaction.proton_model_space
    neutron_ms: ModelSpace = interaction.neutron_model_space
    
    max_proton_j_couplings = np.max(np.concatenate(proton_ms.jz))
    max_neutron_j_couplings = np.max(np.concatenate(neutron_ms.jz))
    max_j_couplings = max(max_proton_j_couplings, max_neutron_j_couplings) + 1
    j_couple: np.ndarray = np.zeros(
        shape = (max_j_couplings, n_parities, n_proton_neutron),
        dtype = CouplingIndices
    )

    # Initialise the proton coupling indices.
    for i in range(max_j_couplings):
        for j in range(n_parities):
            j_couple[i, j, 0]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((proton_ms.n_orbitals, proton_ms.n_orbitals), dtype=int)
            )

    # Initialise the neutron coupling indices.
    for i in range(max_j_couplings):
        for j in range(n_parities):
            j_couple[i, j, 1]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((neutron_ms.n_orbitals, neutron_ms.n_orbitals), dtype=int)
            )

    # Initialise the proton-neutron coupling indices.
    for i in range(max_j_couplings):
        for j in range(n_parities):
            j_couple[i, j, 2]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((proton_ms.n_orbitals, neutron_ms.n_orbitals), dtype=int)
            )

    calculate_couplings(
        model_space_1 = proton_ms,
        model_space_2 = proton_ms,
        j_couple = j_couple,
        proton_neutron_idx = 0
    )

    calculate_couplings(
        model_space_1 = neutron_ms,
        model_space_2 = neutron_ms,
        j_couple = j_couple,
        proton_neutron_idx = 1
    )

    calculate_couplings(
        model_space_1 = proton_ms,
        model_space_2 = neutron_ms,
        j_couple = j_couple,
        proton_neutron_idx = 2
    )

    # for i in range(jz_max):
    #     # for j in range(n_parities):
    #     print(j_couple[i, 1, 0].n_couplings)
    #     print(j_couple[i, 1, 0].idx)
    #     print(j_couple[i, 1, 0].idx_reverse)

    return j_couple
    
def calculate_couplings(
    model_space_1: ModelSpace,
    model_space_2: ModelSpace,
    j_couple: np.ndarray,
    proton_neutron_idx: int
):
    """
    Calculate the possible couplings of proton-proton, neutron-neutron,
    and proton-neutron orbitals. model_space_1 and model_space_2 are
    not edited in this function, but the coupling indices are added to
    the j_couple array.

    Parameters
    ----------
    model_space_1 : ModelSpace
        Either proton or neutron model space.

    model_space_2 : ModelSpace
        Either proton or neutron model space.

    j_couple : np.ndarray
        Array of coupling indices.
    """
    for orbital_1 in range(model_space_1.n_orbitals):
        
        if proton_neutron_idx == 2:
            """
            In proton-neutron coupling, the couplings (i, j) and (j, i)
            are different couplings and both must be counted. The first
            case is proton orbital i and neutron orbital j, and the
            second case is neutron orbital i and proton orbital j.
            """
            range_start = 0
        else:
            """
            In proton-proton and neutron-neutron coupling, the couplings
            (i, j) and (j, i) are the same coupling and only one must be
            counted.
            """
            range_start = orbital_1
            
        for orbital_2 in range(range_start, model_space_2.n_orbitals):
            
            if model_space_1.parity[orbital_1] == model_space_2.parity[orbital_2]:
                """
                The total parity is always +1 if each orbital have the
                same parity. parity_idx 0 means positive parity.
                """
                parity_idx = 0
            else:
                """
                The total parity is always -1 if each orbital have
                different parities. parity_idx 1 means negative parity.
                """
                parity_idx = 1

            if (orbital_1 == orbital_2) and (proton_neutron_idx != 2):
                """
                Two identical nucleons in the same orbital can only have
                even-numbered couplings.
                """
                j_step = 2
            else:
                """
                Two nucleons in different orbitals can have even- and
                odd-numbered couplings.
                """
                j_step = 1

            possible_couplings: np.ndarray = np.arange( # NOTE: model_space.j values are stored as 2*j to avoid fractions, but the couplings are never fractions and are stored as-is!
                start = abs(model_space_1.j[orbital_1] - model_space_2.j[orbital_2])/2,
                stop = (model_space_1.j[orbital_1] + model_space_2.j[orbital_2] + j_step)/2,    # + j_step to include the end point.
                step = j_step,
                dtype = int
            )

            for j in possible_couplings:
                """
                Count each possible coupling (n_couplings). Save the
                indices of the coupled orbitals (idx) as column vectors
                in a 2D array.

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
                j_couple[j, parity_idx, proton_neutron_idx].n_couplings += 1  # Count the coupling.
                
                j_couple[j, parity_idx, proton_neutron_idx].idx = np.concatenate( # Append the indices.
                    (j_couple[j, parity_idx, proton_neutron_idx].idx, np.array([orbital_1, orbital_2], dtype=int).reshape(-1, 1)),
                    axis = 1
                )
                
                j_couple[j, parity_idx, proton_neutron_idx].idx_reverse[orbital_1, orbital_2] = \
                    np.max(j_couple[j, parity_idx, proton_neutron_idx].idx_reverse) + 1   # Save the order of which the coupling was counted (se docstring for structure).

def operator_j_scheme(
    path: str,
    nucleus_mass: int
):
    """
    In this function, the "raw" reduced matrix elements from the
    interaction file are multiplied by a few different factors and
    stored in an OperatorJ object. NOTE: At the moment I do not know
    what the purpose of changing the raw matrix elements is.

    Parameters
    ----------
    path : str
        Path to interaction file (.snt file).

    nucleus_mass : int
        Mass number of the nucleus. E.g. 20 for 20Ne.
    """
    interaction = read_interaction_file(path=path)
    initialise_operator_j_scheme(interaction=interaction)
    sys.exit()
    
    operator_j_scheme_time = time.perf_counter()

    operator_j = OperatorJ(
        one_body = OneBody(
            orbital_0 = np.copy(interaction.one_body.orbital_0),
            orbital_1 = np.copy(interaction.one_body.orbital_1),
            reduced_matrix_element = np.zeros_like(interaction.one_body.reduced_matrix_element),
            method = interaction.one_body.method,
            n_elements = interaction.one_body.n_elements,
        ),
        two_body = TwoBody(
            orbital_0 = np.copy(interaction.two_body.orbital_0),
            orbital_1 = np.copy(interaction.two_body.orbital_1),
            orbital_2 = np.copy(interaction.two_body.orbital_2),
            orbital_3 = np.copy(interaction.two_body.orbital_3),
            parity = np.copy(interaction.two_body.parity),
            jj = np.copy(interaction.two_body.jj),
            reduced_matrix_element = np.zeros_like(interaction.two_body.reduced_matrix_element),
            method = interaction.two_body.method,
            im0 = interaction.two_body.im0,
            pwr = interaction.two_body.pwr,
            n_elements = interaction.two_body.n_elements,
        )
    )
    # ONE-BODY
    if interaction.one_body.method == 0:
        one_body_mass_dependence = 1    # NOTE: Whats this?
    else:
        msg = f"one_body_mass_dependence only implemented for one-body method 0. Got {interaction.one_body.method}."
        raise NotImplementedError(msg)

    for i in range(interaction.one_body.n_elements):
        """
        One-body loop.

        NOTE: operator_jscheme.f90 stores the reduced matrix elements as
        2D matrices and stores an entry for [orbital_0, orbital_1] and
        [orbital_1, orbital_0]. Swapping the indices changes nothing
        since they are equal (at least for usda.snt, jun45.snt,
        and gs8.snt) so I don't do that here. read_interaction_file also
        raises an error if they are not equal, so they will always be
        equal here.
        """
        factor = math.sqrt(interaction.model_space.j[i] + 1) # NOTE: Figure out the origin of this factor.
        operator_j.one_body.reduced_matrix_element[i] = \
            interaction.one_body.reduced_matrix_element[i]*one_body_mass_dependence*factor

    # TWO-BODY
    if interaction.two_body.method == 1:
        two_body_mass_dependence = (nucleus_mass/operator_j.two_body.im0)**operator_j.two_body.pwr
    else:
        msg = "Only method 1 is implemented for two-body."
        msg += f" Got: {interaction.two_body.method}."
        raise NotImplementedError(msg)

    for i in range(interaction.two_body.n_elements):
        """
        Two-body loop.
        """
        operator_j.two_body.reduced_matrix_element[i] = \
            interaction.two_body.reduced_matrix_element[i]*two_body_mass_dependence

        isign_01 = 1    # NOTE: Dont know what these are yet.
        isign_23 = 1

        if (
            (interaction.two_body.orbital_0[i] > interaction.two_body.orbital_1[i]) or
            (interaction.two_body.orbital_2[i] > interaction.two_body.orbital_3[i])
        ):
            """
            NOTE: Don't know why this has to happen (copied from
            operator_jscheme.f90). gs8.snt, jun45.snt and usda.snt
            do not fulfill this condition. For now this raises an error.
            
            if orbital_0 > orbital_1:
                orbital_0, orbital_1 = orbital_1, orbital_0
                isign_01 = (-1)**((model_space.j(orbital_0) + model_space.j(orbital_1))/2 - jj + 1)
            
            if orbital_2 > orbital_3:
                orbital_2, orbital_3 = orbital_3, orbital_2
                isign_23 = (-1)**((model_space.j(orbital_2) + model_space.j(orbital_3))/2 - jj + 1)
            """
            msg = "orbital_0 > orbital_1 or orbital_2 > orbital_3."
            msg += " Don't know why this has to be taken care of."
            raise NotImplementedError(msg)

        operator_j.two_body.reduced_matrix_element[i] *= isign_01*isign_23  # Pointless as of now, since the previous if statement is not implemented yet.
        
        if (
            (interaction.two_body.orbital_1[i] <= (interaction.model_space.n_proton_orbitals - 1)) and
            (interaction.two_body.orbital_3[i] <= (interaction.model_space.n_proton_orbitals - 1))
        ):
            """
            If orbital 1 and 3 are smaller or equal to the number of
            proton orbitals, then ... (?)

            Subtracting 1 to correct for the subtraction of the orbital
            indices.
            """
            ipn = 0 # Index proton neutron?

        elif (
            (interaction.two_body.orbital_0[i] > (interaction.model_space.n_proton_orbitals - 1)) and
            (interaction.two_body.orbital_2[i] > (interaction.model_space.n_proton_orbitals - 1))
        ):
            """
            If orbital 0 and 2 are greater than the number of proton
            orbitals, then ... (?)
            """
            ipn = 1

        elif (
            (interaction.two_body.orbital_0[i] <= (interaction.model_space.n_proton_orbitals - 1)) and
            (interaction.two_body.orbital_2[i] <= (interaction.model_space.n_proton_orbitals - 1)) and
            (interaction.two_body.orbital_1[i] >  (interaction.model_space.n_proton_orbitals - 1)) and
            (interaction.two_body.orbital_3[i] >  (interaction.model_space.n_proton_orbitals - 1))
        ):
            ipn = 2
        
        else:
            """
            NOTE: I dont know the consequences of this yet.
            """
            msg = f"Error in orbital index iteration {i} from interaction file: "
            msg += f"{path}.\n"
            msg += f"{interaction.two_body.orbital_0[i] = }\n"
            msg += f"{interaction.two_body.orbital_1[i] = }\n"
            msg += f"{interaction.two_body.orbital_2[i] = }\n"
            msg += f"{interaction.two_body.orbital_3[i] = }\n"
            msg += f"{interaction.model_space.n_proton_orbitals = }"
            raise RuntimeError(msg)

        ij_01 = jcouple[
            interaction.two_body.jj, interaction.two_body.parity, ipn
        ].idxrev[interaction.two_body.orbital_0, interaction.two_body.orbital_1]
        
        ij_23 = jcouple[
            interaction.two_body.jj, interaction.two_body.parity, ipn
        ].idxrev[interaction.two_body.orbital_2, interaction.two_body.orbital_3]

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

    operator_j_scheme_time = time.perf_counter() - operator_j_scheme_time
    if flags.debug:
        print(f"{operator_j_scheme_time = :.4f} s")

if __name__ == "__main__":
    flags.debug = True
    # read_interaction_file(path="usda.snt")
    operator_j_scheme(
        path = "../snt/usda.snt",
        nucleus_mass = 20
    )