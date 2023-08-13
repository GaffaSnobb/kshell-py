import time
import numpy as np
from numpy.linalg import eigh
import kshell_utilities as ksutil
from kshell_utilities.data_structures import (
    Interaction, Partition, OrbitalParameters
)
from kshell_utilities.loaders import load_interaction, load_partition
from partition import calculate_all_possible_orbital_occupations
from tools import n_choose_k
from parameters import flags
from data_structures import timings
from basic_solver_tests import O18_w_manual_hamiltonian

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
    orbital_occupations = calculate_all_possible_orbital_occupations(
        interaction = interaction,
    )
    print(orbital_occupations)
    return
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
    timings.create_hamiltonian = timing - timings.calculate_all_possible_orbital_occupations - timings.fill_orbitals

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
    # print(timings)