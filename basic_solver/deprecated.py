import time
import numpy as np
from numpy.linalg import eigh
import kshell_utilities as ksutil
from kshell_utilities.data_structures import (
    Interaction, Partition, OrbitalParameters
)
from kshell_utilities.loaders import load_interaction, load_partition
from tools import n_choose_k
from parameters import flags
from data_structures import timings
from basic_solver_tests import O18_w_manual_hamiltonian

def fill_orbitals(
    orbitals: list[OrbitalParameters],
    orbital_occupations: list[tuple[int]],
    current_orbital_occupation: list[int],
    n_remaining_neutrons: int,
    n_remaining_holes: int,
    current_orbital_idx: int,
):
    """
    Fill all the orbitals in the given model space with all possible
    combinations of occupations. Account for the orbitals' degeneracies.

    To account for a variable number of orbitals, this function will
    call itself recursively and for each call proceeding to the next
    orbital in the model space. For each call, the function loops over
    all possible numbers of occupation, accounting for the current
    orbital degeneracy and the number of remaining nucleons.

    Example
    -------
    Assume a model space of orbitals = [d5/2, d3/2].

    for occupation in [0, ..., d5/2 max allowed occupation]:
        store the current d5/2 occupation
    --- call fill_orbitals, excluding d5/2 in the list 'orbitals'
    |
    |
    --> for occupation in [0, ..., d3/2 max allowed occupation]:
        store the current d3/2 occupation
    --- call fill_orbitals, excluding d3/2 in the list 'orbitals'
    |
    |
    --> The function returns early because there are no orbitals left.
        If there are no nucleons left, 'current_orbital_occupation' is
        saved to 'orbital_occupations'. If there are nucleons left, the
        current orbital occupation is not stored and the occupation
        iteration in the previous recursive call is continued.

    Parameters
    ----------
    orbitals:
        List of the remaining orbitals in the model space.
        OrbitalParameters contains various parameters for the orbitals.

    orbital_occupations:
        A list for storing the valid configurations.

    current_orbital_occupation:
        Storage for the current occupation. Will be copied to
        orbital_occupations if it is valid.

    n_remaining_neutrons:
        The remaining number of nucleons to place into the remaining
        orbitals.
    
    n_remaining_holes:
        The remaining free places to put the remaining nucleons.
    
    current_orbital_idx:
        The index of the current orbital for the
        current_orbital_occupation list. Is incremented +1 for every
        recursive function call.
    """
    timing = time.perf_counter()
    if n_remaining_neutrons == 0:
        """
        No more neutrons to place, aka a complete configuration.
        """
        orbital_occupations.append(tuple(current_orbital_occupation))    # tuple conversion was marginally faster than list.copy().
        return

    if not orbitals:
        """
        No remaining orbitals but there are remaining neutrons, aka
        incomplete configuration.
        """
        return
    
    if n_remaining_neutrons > n_remaining_holes:
        """
        Not enough holes for the remaining neutrons, aka incomplete
        configuration.
        """
        return
    
    current_orbital = orbitals[0]

    for occupation in range(0, min(current_orbital.degeneracy, n_remaining_neutrons) + 1):
        current_orbital_occupation[current_orbital_idx] += occupation
        
        fill_orbitals(
            orbitals = orbitals[1:],
            n_remaining_neutrons = n_remaining_neutrons - occupation,
            n_remaining_holes = n_remaining_holes - current_orbital.degeneracy,
            current_orbital_idx = current_orbital_idx + 1,
            orbital_occupations = orbital_occupations,
            current_orbital_occupation = current_orbital_occupation,
        )
        current_orbital_occupation[current_orbital_idx] -= occupation

    timing = time.perf_counter() - timing
    timings.fill_orbitals = timing  # The initial call of this function will overwrite all the timings from the recursive calls, giving the total time.

def calculate_hamiltonian_orbital_occupation(
    interaction: Interaction,
) -> list[tuple[int]]:
    """
    Given a number of valence nucleons and a model space, calculate the
    different possible orbital occupations. The allowed occupations will
    define the elements of the Hamiltonian matrix.
    
    Parameters
    ----------
    interaction:
        Custom dataclass containing all necessary parameters of the
        given interaction.

    Returns
    -------
    orbital_occupations:
        A list containing each allowed orbital occupation. Example:
            
            [(0, 1, 2), (0, 2, 1), ...]

        where the first allowed occupation tells us that the first
        orbital has zero nucleons, the second orbital has 1 nucleon, and
        the third orbital has 2 nucleons. The order of the orbitals is
        the same as the order they are listed in the interaction file.
    """
    timing = time.perf_counter()
    current_orbital_occupation: list[int] = [0]*interaction.model_space_neutron.n_orbitals
    orbital_occupations: list[tuple[int]] = []

    fill_orbitals(
        orbitals = interaction.model_space_neutron.orbitals,
        n_remaining_neutrons = interaction.model_space_neutron.n_valence_nucleons,
        n_remaining_holes = sum([orb.degeneracy for orb in interaction.model_space_neutron.orbitals]),
        current_orbital_idx = 0,
        orbital_occupations = orbital_occupations,
        current_orbital_occupation = current_orbital_occupation,
    )

    if flags["debug"]:
        """
        Should already be sorted lexicographically from the way the
        orbitals are traversed.
        """
        assert sorted(orbital_occupations) == orbital_occupations
    
    timing = time.perf_counter() - timing
    timings.calculate_hamiltonian_orbital_occupation = timing - timings.fill_orbitals

    return orbital_occupations

def calculate_all_possible_pairs(
    configuration: tuple[int],
) -> list[tuple[int, int]]:
    """
    Calculate all the possible choices of two nucleons from some
    configuration of nucleons in orbitals.

    TODO: @cache this?

    Example
    -------
    ```
    >>> calculate_all_possible_pairs((1, 1, 1))
    [(0, 1), (0, 2), (1, 2)]
    ```
    Parameters
    ----------
    configuration:
        A list containing a possible configuration of the valence
        nucleons in the orbitals of the model space. Represented as
        indices of the orbitals. Ex.:

            (1, 1, 1), (2, 1, 0)

        which means that the first configuration has one nucleon in each
        of the three orbitals of the model space, while the sencond
        configuration has two nucleons in the first orbital, one nucleon
        in the second orbital and no nucleons in the third orbital.
    """
    timing = time.perf_counter()
    configuration_pair_permutation_indices: list[tuple[int, int]] = []
    n_occupations = len(configuration)

    for idx in range(n_occupations):
        occ = configuration[idx]
        if occ == 0: continue

        configuration_pair_permutation_indices.extend(
            [(idx, idx)]*n_choose_k(n=occ, k=2)
        )
    
    for idx_1 in range(n_occupations):
        occ_1 = configuration[idx_1]
        if occ_1 == 0: continue

        for idx_2 in range(idx_1 + 1, n_occupations):
            occ_2 = configuration[idx_2]
            if occ_2 == 0: continue

            if idx_1 == idx_2:
                """
                This if is needed for the cases where the
                configuration occupies only a single orbital.
                """
                # print("equal", [(idx_1, idx_2)]*n_choose_k(n=occ_1, k=2))
                configuration_pair_permutation_indices.extend(
                    [(idx_1, idx_2)]*n_choose_k(n=occ_1, k=2)
                )
            
            else:
                # print("NÆÆT equal", [(idx_2, idx_2)]*n_choose_k(n=occ_2, k=2))
                # print("NÆÆT equal", [(idx_1, idx_2)]*occ_1*occ_2)
                # configuration_pair_permutation_indices.extend(
                #     [(idx_2, idx_2)]*n_choose_k(n=occ_2, k=2)
                # )
                configuration_pair_permutation_indices.extend(
                    [(idx_1, idx_2)]*occ_1*occ_2
                )

    timing = time.perf_counter() - timing
    timings.calculate_all_possible_pairs += timing

    return configuration_pair_permutation_indices

def create_hamiltonian(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
) -> np.ndarray:  
    timing = time.perf_counter()
    timings.calculate_all_possible_pairs = 0.0    # This timing value will be added to several times and must start at 0.
    orbital_occupations = calculate_hamiltonian_orbital_occupation(
        interaction = interaction,
    )
    print(orbital_occupations)
    n_occupations = len(orbital_occupations)
    H = np.zeros((n_occupations, n_occupations))

    for idx_row in range(n_occupations):
        """
        Generate the matrix elements of the hamiltonian.
        """
        orbital_indices_row = orbital_occupations[idx_row]
        tbme_indices_row = calculate_all_possible_pairs(configuration=orbital_indices_row)
        print(tbme_indices_row)
        # return
        
        for idx_col in range(n_occupations):
            matrix_element = 0.0
            orbital_indices_col = orbital_occupations[idx_col]
            tbme_indices_col = calculate_all_possible_pairs(configuration=orbital_indices_col)

            if idx_row == idx_col:
                """
                The single particle energies only show up in the
                diagonal elements.
                """
                for idx_orb in range(interaction.model_space_neutron.n_orbitals):
                    matrix_element += orbital_indices_row[idx_orb]*interaction.spe[idx_orb]

            for tbme_index_row, tbme_index_col in zip(tbme_indices_row, tbme_indices_col):
                """
                Choose all possible pairs of nucleons from each of the
                configurations. In a two nucleon setting we could have:

                    [(0, 0, 2), (0, 1, 1), ...]

                In this case the only way to choose a pair of nucleons
                from the first configuration is two nucleons in the
                third orbital. In the second configuration the only way
                is one nucleon in the second orbital and one nucleon in
                the thirs orbital. However, for a three nucleon setting
                we could have:

                    [(0, 1, 2), (0, 2, 1), ...]

                where in the first configuration we can pick one nucleon
                in orbital 2 and one nucleon in orbital 3, and we can do
                this twice since there are two particles in orbital 3.
                """
                tbme_index_row_col = tbme_index_row + tbme_index_col + (0,)
                matrix_element += interaction.tbme.get(tbme_index_row_col, 0)

            H[idx_row, idx_col] = matrix_element

    timing = time.perf_counter() - timing
    timings.create_hamiltonian = timing - timings.calculate_hamiltonian_orbital_occupation - timings.fill_orbitals

    return H

def main():
    interaction: Interaction = load_interaction(filename_interaction="O19_w/w.snt")
    partition_proton, partition_neutron, partition_combined = \
        load_partition(filename_partition="O19_w/O19_w_p.ptn", interaction=interaction)
    
    timing = time.perf_counter()
    H = create_hamiltonian(
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )
    timing = time.perf_counter() - timing
    timings.main = timing

    # print(np.all(H == O18_w_manual_hamiltonian()))
    # eigvalues, eigfunctions = eigh(H)
    # print(f"{eigvalues = }")

if __name__ == "__main__":
    flags["debug"] = True
    main()
    print(timings)


def load_interaction(
    path: str,
) -> Interaction:

    single_particle_energies: list[float] = []
    twobody_matrix_elements: dict[tuple[int, int, int, int, int], float] = {}

    with open(f"{path}/w_spe.txt", "r") as infile:
        for line in infile:
            tmp = line.split()
            single_particle_energies.append(float(tmp[-1]))

    with open(f"{path}/w_tbme.txt", "r") as infile:
        for line in infile:
            orb_0, orb_1, orb_2, orb_3, j, tbme = line.split()
            orb_0 = int(orb_0)
            orb_1 = int(orb_1)
            orb_2 = int(orb_2)
            orb_3 = int(orb_3)
            j = int(j)*2
            tbme = float(tbme)

            twobody_matrix_elements[(orb_0, orb_1, orb_2, orb_3, j)] = tbme
            twobody_matrix_elements[(orb_2, orb_3, orb_0, orb_1, j)] = tbme

    interaction: Interaction = Interaction(
        spe = single_particle_energies,
        tbme = twobody_matrix_elements,
    )
    return interaction

from math import sqrt
import numpy as np
from parameters import clebsch_gordan
import kshell_utilities as ksutil
from kshell_utilities.data_structures import Interaction
from kshell_utilities.loaders import load_interaction

j = {
    0: 3,
    1: 1,
}
o_to_m_map = {
    (0, -3): 0,
    (0, -1): 1,
    (0, 1): 2,
    (0, 3): 3,
    (1, -1): 4,
    (1, 1): 5,
}
jz = {
    0: (-3, -1, 1, 3),
    1: (-1, 1),
}

def annihilation_operator(
    gamma: int,
    delta: int,
    right_state: tuple[int, ...],
) -> list[tuple[int, list[int]]]:
    new_right_states: list[tuple[int, list[int]]] = []
    norm_annihilation = 1/sqrt(1 + (gamma == delta))
    # res_annihilation: float = 0.0

    for m_gamma in jz[gamma]:
        for m_delta in jz[delta]:
            new_right_state = list(right_state)

            try:
                delta_match_idx = new_right_state.index(o_to_m_map[(delta, m_delta)])
            except ValueError:
                """
                Annihilating an unoccupied state yields zero.
                """
                continue
            
            new_right_state.pop(delta_match_idx)
            annihilation_sign = (-1)**(delta_match_idx)

            try:
                gamma_match_idx = new_right_state.index(o_to_m_map[(gamma, m_gamma)])
            except ValueError:
                """
                Annihilating an unoccupied state yields zero.
                """
                continue

            new_right_state.pop(gamma_match_idx)
            annihilation_sign *= (-1)**(gamma_match_idx)

            cg_annihilation = clebsch_gordan[(
                j[gamma],   # j1
                m_gamma,    # m1
                j[delta],   # j2
                m_delta,    # m2
                0,          # j
                0,          # m
            )]
            if not cg_annihilation: continue    # Might be more performant to just multiply the term by zero (?).
            # res_annihilation = norm_annihilation*cg_annihilation
            new_right_states.append((norm_annihilation*cg_annihilation, new_right_state))

    return new_right_states

def creation_operator(
    alpha: int,
    beta: int,
    # new_right_state: list[int],
    res_annihilation: list[tuple[int, list[int]]],
    tmp_left_state: list[int],
):
    res_creation: float = 0.0
    norm_creation = 1/sqrt(1 + (alpha == beta))

    for annihilation_coefficient, right_state in res_annihilation:

        for m_alpha in jz[alpha]:
            
            if o_to_m_map[((alpha, m_alpha))] in right_state: continue
            
            for m_beta in jz[beta]:
                
                if (m_beta == m_alpha) and (alpha == beta):
                    """
                    Creating the same state yields zero.
                    """
                    continue

                if o_to_m_map[((beta, m_beta))] in right_state: continue

                new_right_state = right_state.copy()

                new_right_state.insert(0, o_to_m_map[((beta, m_beta))])
                new_right_state.insert(0, o_to_m_map[((alpha, m_alpha))])

                if tmp_left_state == new_right_state:
                    creation_sign = +1
                
                elif tmp_left_state == new_right_state[::-1]:
                    creation_sign = -1
                
                else:
                    """
                    Assuming that the basis states are orthonormal.
                    """
                    continue

                cg_creation = clebsch_gordan[(
                    j[alpha],   # j1
                    m_alpha,    # m1
                    j[beta],   # j2
                    m_beta,    # m2
                    0,          # j
                    0,          # m
                )]
                if not cg_creation: continue

                input()
                print(f"{alpha = }")
                print(f"{beta = }")
                print(f"{m_alpha = }")
                print(f"{m_beta = }")
                print(f"{right_state = }")
                print(f"{new_right_state = }")
                print(f"{cg_creation = }")
                print(f"{annihilation_coefficient = }")
                print(f"{norm_creation = }")
                print(f"{creation_sign = }")
                res_creation += norm_creation*cg_creation*creation_sign*annihilation_coefficient
                print(f"{res_creation = }")
    
    print("END")
    # print(F"{res_creation = }")
    return res_creation

def hamiltonian_operator_orbit_loop_O18_no_d5(interaction: Interaction):
    """
    Make the Hamiltonian matrix elements of O18 in the sd model space
    with no d5/2 orbital, only d3/2 and s1/2.

                   4     5        
                -  O  -  O  -             s1/2: 1
                 -1/2   1/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2

    """
    trunc_to_sd_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 10,
        5: 11,
    }
    m_to_o_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
    }
    basis_states = (
        (0, 3),
        (1, 2),
        (1, 5),
        (2, 4),
        (4, 5),
    )

    # annihilation_terms = []
    # creation_terms = []

    H = np.zeros((len(basis_states), len(basis_states)), dtype=np.float64)

    spe = {
        (0, 0): interaction.spe[0],
        (1, 1): interaction.spe[2], # Yes 2, because d5/2 is truncated away so orb idx 1 is s1/2
    }

    for i_left, left_state in enumerate(basis_states):
        tmp_left_state = list(left_state)   # Will not be modified, this is just to be able to compare lists with != and == which does not work with a list and a tuple.
        for i_right, right_state in enumerate(basis_states):

            for alpha in range(2):
                """
                Creation.
                """
                for beta in range(alpha, 2):
                    """
                    Annihilation
                    """
                    tmpres = 0.0
                    for m_alpha in jz[alpha]:
                        for m_beta in jz[beta]:
                            new_right_state = list(right_state)

                            try:
                               """
                               Need the index of the potential match to
                               calculate the correct sign:

                                    c_beta | x, beta, ... >
                                =  -c_beta | beta, x, ... >
                               """
                               match_idx = new_right_state.index(o_to_m_map[(beta, m_beta)])
                            except ValueError:
                                """
                                The state is not occupied, and using an
                                annihilation operator on it yields 0.
                                """
                                continue

                            new_right_state.pop(match_idx)
                            sign = (-1)**match_idx  # Moving the annihilation operator to the correct position before letting it act might pick up a sign.
                            tmp_right_sign = sign
                            
                            if o_to_m_map[(alpha, m_alpha)] in new_right_state:
                                """
                                The state is already occupied. Using a
                                creation operator on it will yield 0.
                                """
                                continue
                            else:
                                """
                                We insert the newly created particle state at
                                the left side of the existing state to keep
                                track of the order of the operators:

                                    | alpha, x, y, ... >
                                """
                                new_right_state.insert(0, o_to_m_map[(alpha, m_alpha)])

                            # print(f"{new_right_state}")
                            # print(f"{sign}")
                            # print()

                            if tmp_left_state == new_right_state:
                                """
                                Example:
                                    | left > = | new_right > = | 0 3 >
                                    < 3 0 | 0 3 >
                                =   < core | c_3 c_0 c_0^\dagger c_3^\dagger | core >
                                =   < core | core >
                                =   1
                                """
                                # sign *= 1
                                tmp_left_sign = +1
                                pass
                            
                            elif tmp_left_state[::-1] == new_right_state:
                                """
                                Because of the reversed order, a minus
                                sign is picked up.
                                """
                                tmp_left_sign = -1
                                sign *= -1
                                pass
                            
                            else:
                                """
                                Assuming that the basis states are
                                orthogonal, the inner product of
                                different states is zero.
                                """
                                continue

                            # H[i_left, i_right] += sign*spe.get(
                            #     (m_to_o_map[new_right_state[0]], m_to_o_map[new_right_state[1]]),
                            #     0
                            # )
                            # tmpres = sign*spe.get(
                            #     (m_to_o_map[new_right_state[0]], m_to_o_map[new_right_state[1]]),
                            #     0
                            # )
                            tmpres = sign*spe.get((alpha, beta), 0)
                            H[i_left, i_right] += tmpres
                    
                            # if tmpres:
                            #     input("lol:")
                            #     print(f"{tmpres = }")
                            #     print(f"H[{i_left}, {i_right}]")
                            #     print(f"c_{o_to_m_map[(alpha, m_alpha)]}+ c_{o_to_m_map[(beta, m_beta)]} | {right_state} > = {'+' if tmp_right_sign == 1 else '-'}| {tuple(new_right_state)} >")
                            #     print(f"< ({tmp_left_state[1]}, {tmp_left_state[0]}) | {tuple(new_right_state)} > = {'-' if tmp_left_sign == -1 else '+'}1")
                            #     print(f"{tmp_left_state  = }")
                            #     print(H)
                            #     print()

    # print(H)
    # print()
    # return

    for i_left, left_state in enumerate(basis_states):
        tmp_left_state = list(left_state)   # Will not be modified, this is just to be able to compare lists with != and == which does not work with a list and a tuple.
        for i_right, right_state in enumerate(basis_states):

            for alpha in range(2):
                """
                Creation.
                """
                for beta in range(alpha, 2):

                    for gamma in range(2):
                        """
                        Annihilation.
                        """
                        for delta in range(gamma, 2):
                            # new_right_state = list(right_state)
                            
                            res_annihilation = annihilation_operator(
                                gamma = gamma,
                                delta = delta,
                                right_state = right_state,
                            )
                            res_creation = creation_operator(
                                alpha = alpha,
                                beta = beta,
                                # new_right_state = new_right_state
                                res_annihilation = res_annihilation,
                                tmp_left_state = tmp_left_state,
                            )

                            tbme = interaction.tbme.get((
                                2 if (alpha == 1) else alpha,   # To get the right orb idx for the s1/2 orb.
                                2 if (beta == 1) else beta,
                                2 if (gamma == 1) else gamma,
                                2 if (delta == 1) else delta,
                                0,
                                ),
                                0
                            )
                            H[i_left, i_right] += res_creation*tbme

    assert np.all(H == H.T)
    print(H)
    print()
    from numpy.linalg import eigh
    eigenvals, eigenvecs = eigh(H)
    print(eigenvals)

    O18 = ksutil.loadtxt(path="O18_w_no_d5/")
    print(O18.levels)

if __name__ == "__main__":
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 2
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    hamiltonian_operator_orbit_loop_O18_no_d5(interaction=interaction)

import math
import numpy as np
from data_structures import Interaction
from loaders import load_interaction
import kshell_utilities.loaders
import kshell_utilities.data_structures
import kshell_utilities as ksutil
from parameters import clebsch_gordan

# jz = [
#     [-3, -1, +1, +3],
#     [-1, +1]
# ]
jz_indices = [
    range(4),
    range(2),
]
om_pair_to_m_idx_map = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (1, 0): 4,
    (1, 1): 5,
}
j_idx_to_j = {
    0: 3,
    1: 1,
}
j_jz_idx_to_jz = {
    (0, 0): -3,
    (0, 1): -1,
    (0, 2): +1,
    (0, 3): +3,
    (1, 0): -1,
    (1, 1): +1,
}

def onebody_hamiltonian_element(
    single_particle_energies: list[float],
    left_state: tuple[int, ...],
    right_state: tuple[int, ...],
):
    """
    Calculate one one-body element of the Hamiltonian matrix, as defined
    in the KSHELL manual by
    ```
    \sum_{\alpha, \beta, m_\alpha, m_\beta} \epsilon_{\alpha \beta} c_{\alpha m_\alpha}^\dagger c_{\beta m_\beta}
    ```

                   4     5        
                -  O  -  O  -             s1/2: 1
                 -1/2   1/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2
    """
    # single_particle_energies = [interaction.spe[0], interaction.spe[2]] # Skipping [1] which is the d5/2 orbital.
    tmp_left_state = list(left_state)
    res = 0.0

    for alpha in range(2):
        """
        Creation operator index alpha and annihilation operator index
        beta. Loops over the orbitals in the model space.

        The full equation also has an identical loop over beta, but
        since the SPEs in my case are only non-zero when alpha = beta,
        theres no real reason to loop over all combinations of alpha and
        beta.
        """
        beta = alpha
        spe = single_particle_energies[alpha]

        for m_alpha in jz_indices[alpha]:
            """
            Creation operator index.
            """
            for m_beta in jz_indices[beta]:
                """
                Annihilation operator index.
                """
                new_right_state = list(right_state)

                try:
                    annihilation_idx = new_right_state.index(om_pair_to_m_idx_map[(beta, m_beta)])
                except ValueError:
                    """
                    In this case, the annihilation operator is trying to
                    annihilate an un-occupied state. This yields zero.
                    """
                    continue
                
                new_right_state.pop(annihilation_idx)
                annihilation_sign = (-1)**annihilation_idx  # If the annihilation operator has to be shifted an odd number of times, a -1 is picked up due to anti-commutation.

                if om_pair_to_m_idx_map[(alpha, m_alpha)] in new_right_state:
                    """
                    If we try to create a particle in an already
                    occupied position, the result is zero.
                    """
                    continue

                new_right_state.insert(0, om_pair_to_m_idx_map[(alpha, m_alpha)])

                if new_right_state[0] > new_right_state[1]:
                    """
                    This assumes that there are only two valence
                    nucleons so this logic has to be re-made later.
                    """
                    new_right_state[0], new_right_state[1] = new_right_state[1], new_right_state[0]
                    creation_sign = -1
                else:
                    creation_sign = +1

                if tmp_left_state != new_right_state:
                    """
                    Assuming that the states are orthonormal,

                    < i | j > = 0
                    < i | i > = 1
                    """
                    continue

                res += annihilation_sign*creation_sign*spe
    return res

def annihilation_operators(
    gamma: int,
    delta: int,
    right_state: tuple[int, ...],
    j: int,
    m: int,
    original_left_state,
    alpha,
    beta,
) -> list[tuple[float, list[int]]]:
    """
    Let the annihilation operators act on the state to the right in the
    inner product.
    """
    annihilation_norm = 1/math.sqrt(1 + (gamma == delta))
    # annihilation_res: list[tuple[float, list[int]]] = []
    annihilation_res = 0.0

    for m_gamma in jz_indices[gamma]:
        for m_delta in jz_indices[delta]:
            new_right_state = list(right_state)

            annihilation_cg = clebsch_gordan[(
                j_idx_to_j[gamma],
                j_jz_idx_to_jz[(gamma, m_gamma)],
                j_idx_to_j[delta],
                j_jz_idx_to_jz[(delta, m_delta)],
                j,
                m,
            )]
            if annihilation_cg == 0: continue

            try:
                annihilation_idx_0 = new_right_state.index(om_pair_to_m_idx_map[(gamma, m_gamma)])
            except ValueError:
                """
                If the state is not occupied, annihilating it will yield
                zero.
                """
                continue
            
            new_right_state.pop(annihilation_idx_0)
            annihilation_sign_0 = (-1)**annihilation_idx_0

            try:
                annihilation_idx_1 = new_right_state.index(om_pair_to_m_idx_map[(delta, m_delta)])
            except ValueError:
                """
                If the state is not occupied, annihilating it will yield
                zero.
                """
                continue
            
            new_right_state.pop(annihilation_idx_1)
            annihilation_sign_1 = (-1)**annihilation_idx_1
            
            assert len(new_right_state) == 0

            annihilation_res += annihilation_norm*annihilation_cg*annihilation_sign_0*annihilation_sign_1
            # annihilation_res.append((annihilation_norm*annihilation_cg*annihilation_sign_0*annihilation_sign_1, new_right_state))

    return annihilation_res

def creation_operators(
    alpha: int,
    beta: int,
    # right_states: list[tuple[float, list[int]]],    # Note that these are new right states from the annihilation operator function, not the original right states.
    right_states: float,
    j: int,
    m: int,
    original_right_state,
    original_left_state,
) -> list[tuple[float, list[int]]]:
    """
    Let the creation operators act on all the new states that were
    produced by the annihilation operators.    
    """
    tmp_left_state = list(original_left_state)
    creation_norm = 1/math.sqrt(1 + (alpha == beta))
    # creation_res: list[tuple[float, list[int]]] = []
    creation_res = 0.0

    new_right_state = []
    for m_alpha in jz_indices[alpha]:
        for m_beta in jz_indices[beta]:
            creation_cg = clebsch_gordan[(
                j_idx_to_j[alpha],
                j_jz_idx_to_jz[(alpha, m_alpha)],
                j_idx_to_j[beta],
                j_jz_idx_to_jz[(beta, m_beta)],
                j,
                m,
            )]
            if creation_cg == 0: continue


            # for coeff, right_state in right_states:
            # new_right_state = right_state.copy()
            # if om_pair_to_m_idx_map[(beta, m_beta)]   in new_right_state: continue
            # if om_pair_to_m_idx_map[(alpha, m_alpha)] in new_right_state: continue

            new_right_state.insert(0, om_pair_to_m_idx_map[(beta, m_beta)]) # Its important that beta is before alpha.
            new_right_state.insert(0, om_pair_to_m_idx_map[(alpha, m_alpha)])

            if new_right_state[0] > new_right_state[1]:
                """
                By convention, we force the order of the creation
                operators to be of increasing index value. If we
                need to swap operators to make this true, we must
                also remember the sign from the anti-commutation
                relation.
                """
                new_right_state[0], new_right_state[1] = new_right_state[1], new_right_state[0]
                creation_sign = -1
            else:
                creation_sign = +1
            
            if tmp_left_state != new_right_state: continue

            creation_res += right_states*creation_sign*creation_norm*creation_cg
            # creation_res.append((coeff*creation_sign*creation_norm*creation_cg, new_right_state))

    return creation_res

def twobody_hamiltonian_element(
    interaction: Interaction,
    right_state: tuple[int, ...],
    left_state:  tuple[int, ...],
):
    res: float = 0.0
    tmp_left_state = list(left_state)
    for alpha in range(2):
        for beta in range(alpha, 2):
            """
            alpha, beta are creation operator indices.
            """
            for gamma in range(2):
                for delta in range(gamma, 2):
                    """
                    gamma, delta are annihilation operator indices.
                    """
                    j_min = max(
                        abs(j_idx_to_j[alpha] - j_idx_to_j[beta]),
                        abs(j_idx_to_j[gamma] - j_idx_to_j[delta]),
                    )
                    j_max = min(
                        j_idx_to_j[alpha] + j_idx_to_j[beta],
                        j_idx_to_j[gamma] + j_idx_to_j[delta],
                    )
                    for j in range(j_min, j_max + 2, 2):
                        if j not in [0, 2, 4, 6]: raise RuntimeError
                        try:
                            tbme = interaction.tbme[(alpha, beta, gamma, delta, j)]
                        except KeyError:
                            """
                            In this case, the interaction between (alpha,
                            beta) and (gamma, delta) is not defined and is
                            set to 0.
                            """
                            continue
                        
                        for m in range(-j, j + 2, 2):

                            new_right_states = annihilation_operators(
                                gamma = gamma,
                                delta = delta,
                                right_state = right_state,
                                j = j,
                                m = m,
                                original_left_state=left_state,
                                alpha=alpha,
                                beta=beta,
                            )
                            if not new_right_states:
                                """
                                The annihilation operators have annihilated the
                                vacuum (core) state, aka 0.
                                """
                                continue
                                
                            # new_right_states = creation_operators(
                            creation_res = creation_operators(
                                alpha = alpha,
                                beta = beta,
                                right_states = new_right_states,
                                j = j,
                                m = m,
                                original_right_state=right_state,
                                original_left_state=left_state
                            )
                            res += creation_res*tbme
                            # for coeff, new_right_state in new_right_states:
                            #     assert new_right_state[0] < new_right_state[1]  # Sanity check.

                            #     if tmp_left_state != new_right_state:
                            #         """
                            #         Orthogonality requirement.
                            #         """
                            #         continue

                            #     res += coeff*tbme

    return res

def hamiltonian():
    """
                   4     5        
                -  O  -  O  -             s1/2: 1
                 -1/2   1/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2
    """
    interaction: Interaction = load_interaction(path="../snt/w/")
    # interaction: kshell_utilities.data_structures.Interaction = kshell_utilities.loaders.load_interaction(filename_interaction="../snt/w.snt")

    # for tbme in interaction.tbme:
    #     print(tbme)
    basis_states = (    # All configurations which produce M = 0.
        (0, 3),
        (1, 2),
        (1, 5),
        (2, 4),
        (4, 5),
    )
    H = np.zeros((len(basis_states), (len(basis_states))))

    for left_idx, left_state in enumerate(basis_states):
        for right_idx, right_state in enumerate(basis_states):
            
            H[left_idx, right_idx] += onebody_hamiltonian_element(
                single_particle_energies = [interaction.spe[0], interaction.spe[2]],
                left_state = left_state,
                right_state = right_state,
            )

    # print(H)

    for left_idx, left_state in enumerate(basis_states):
        for right_idx, right_state in enumerate(basis_states):
            tmpres = twobody_hamiltonian_element(
                interaction = interaction,
                right_state = right_state,
                left_state = left_state,
            )
            if tmpres != 0:
                H[left_idx, right_idx] += tmpres
                # input("press any key to continue")
                # print(f"             {basis_states[0]}       {basis_states[1]}       {basis_states[2]}       {basis_states[3]}       {basis_states[4]}")
                # for i, row in enumerate(H):
                #     print(basis_states[i], end=" ")
                #     for elem in row:
                #         print(f"{elem:13f}", end="")

                #     print()

    # O18 = ksutil.loadtxt(path="O18_w_no_d5_only_spe/", load_and_save_to_file="overwrite")
    O18 = ksutil.loadtxt(path="O18_w_no_d5/", load_and_save_to_file="overwrite")
    # O18 = ksutil.loadtxt(path="O18_w_no_d5_only_tbme/", load_and_save_to_file="overwrite")
    print(O18.levels)
    # assert np.all(H == H.T)
    print()
    print(H)
    print()
    from numpy.linalg import eigh
    eigenvals, eigenvecs = eigh(H)
    print(eigenvals)



if __name__ == "__main__":
    hamiltonian()