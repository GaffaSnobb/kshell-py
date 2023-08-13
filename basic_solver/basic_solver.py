import time
import numpy as np
from numpy.linalg import eigh
import kshell_utilities as ksutil
from kshell_utilities.loaders import load_interaction, load_partition
from kshell_utilities.data_structures import (
    Interaction, Partition, OrbitalParameters
)
from partition import (
    calculate_all_possible_orbital_occupations, calculate_all_possible_pairs
)
from parameters import flags
from data_structures import timings
from test_basic_solver import O18_w_manual_hamiltonian
from test_partition import test_calculate_all_possible_orbital_occupations

def create_hamiltonian(
    interaction: Interaction,
) -> np.ndarray:  
    timing = time.perf_counter()
    timings.calculate_all_possible_pairs = 0.0    # This timing value will be added to several times and must start at 0.
    orbital_occupations = calculate_all_possible_orbital_occupations(
        interaction = interaction,
    )
    print(orbital_occupations)
    print(calculate_all_possible_pairs(orbital_occupations[0]))
    return
    n_occupations = len(orbital_occupations)
    H = np.zeros((n_occupations, n_occupations))

    for idx_row in range(n_occupations):
        """
        Generate the matrix elements of the hamiltonian.
        """
        orbital_indices_row = orbital_occupations[idx_row]
        tbme_indices_row = calculate_all_possible_pairs(configuration=orbital_indices_row)
        # print(tbme_indices_row)
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
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    
    timing = time.perf_counter()
    H = create_hamiltonian(
        interaction = interaction,
    )
    timing = time.perf_counter() - timing
    timings.main = timing

    # print(np.all(H == O18_w_manual_hamiltonian()))
    # eigvalues, eigfunctions = eigh(H)
    # print(f"{eigvalues = }")

    if flags["debug"]:
        test_calculate_all_possible_orbital_occupations()

if __name__ == "__main__":
    flags["debug"] = True
    main()
    # print(timings)