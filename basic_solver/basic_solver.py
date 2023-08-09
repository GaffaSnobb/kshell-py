from typing import Any
from functools import cache
import numpy as np
import numpy.linalg as lalg
from scipy.special import comb
import kshell_utilities as ksutil
from kshell_utilities.data_structures import (
    Interaction, Partition, OrbitalParameters
)
from kshell_utilities.loaders import load_interaction, load_partition

@cache
def n_choose_k(n, k):
    """
    NOTE: This can be replaced by a lookup table to increase speed
    further.
    """
    return comb(n, k, exact=True)

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
    # orbital_occupations.sort()   # Sort lexicographically. NOTE: Should already be sorted from the way the orbitals are traversed.
    return orbital_occupations

def create_hamiltonian(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
):  
    orbital_occupations = calculate_hamiltonian_orbital_occupation(
        interaction = interaction,
    )
    print(orbital_occupations)

    n_occupations = len(orbital_occupations)
    H = np.zeros((n_occupations, n_occupations))

    for idx_row in range(n_occupations):
        for idx_col in range(n_occupations):
            matrix_element = 0.0

            if idx_row == idx_col:
                for idx_orb in range(interaction.model_space_neutron.n_orbitals):
                    matrix_element += orbital_occupations[idx_row][idx_orb]*interaction.spe[idx_orb]

            for idx_orb in range(interaction.model_space_neutron.n_orbitals):
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

                where the first configuration can give NOTE: Dont know
                yet if (1, 2) should be counted twice or not.
                """

            H[idx_row, idx_col] = matrix_element

def main():
    # interaction: Interaction = load_interaction(filename_interaction="O18_w/w.snt")
    # partition_proton, partition_neutron, partition_combined = \
    #     load_partition(filename_partition="O18_w/O18_w_p.ptn", interaction=interaction)
    
    # create_hamiltonian(
    #     interaction = interaction,
    #     partition_proton = partition_proton,
    #     partition_neutron = partition_neutron,
    #     partition_combined = partition_combined,
    # )

    configurations = [
        [1, 1, 1], [2, 1, 0], [3, 0, 0]
    ]

    tbme_indices: list[list[tuple[int, int]]] = []

    def calculate_all_possible_pairs(configuration: list[int]) -> list[tuple[int, int]]:
        res: list[tuple[int, int]] = []
        
        for idx_1 in range(len(configuration)):
            c1 = configuration[idx_1]
            if c1 == 0: continue

            for idx_2 in range(idx_1, len(configuration)):
                c2 = configuration[idx_2]
                if c2 == 0: continue

                if idx_1 == idx_2:
                    """
                    This if is needed for the cases where the
                    configuration occupies only a single orbital.
                    """
                    res.extend([(idx_1, idx_2)]*n_choose_k(n=c1, k=2))
                
                else:
                    # res.extend([(idx_1, idx_1)]*n_choose_k(n=c1, k=2))
                    res.extend([(idx_2, idx_2)]*n_choose_k(n=c2, k=2))
                    res.extend([(idx_1, idx_2)]*c1*c2)

        return res
        
    # tbme_indices.append(calculate_all_possible_pairs(configuration=[3, 1, 0]))
    for configuration in configurations:
        tbme_indices.append(calculate_all_possible_pairs(configuration=configuration))

    print(tbme_indices)

if __name__ == "__main__":
    main()