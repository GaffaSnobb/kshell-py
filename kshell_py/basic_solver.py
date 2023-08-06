from typing import Any
import numpy as np
import numpy.linalg as lalg
import kshell_utilities as ksutil
from kshell_utilities.data_structures import Interaction, Partition
from kshell_utilities.loaders import load_interaction, load_partition

def O18_w_manual_hamiltonian() -> np.ndarray[Any, float]:
    """
    Manual construction of the shell model hamiltonian for 18O (2
    valence protons) with the w interaction.
    """
    O18 = ksutil.loadtxt(path="tmp_delete/")
    interaction: Interaction = load_interaction(filename_interaction="tmp_delete/w.snt")
    spe = interaction.spe

    def tbme(a, b, c, d, j):
        return interaction.tbme.get((a, b, c, d, j), 0)
        return interaction.tbme[(a, b, c, d, j)]

    H = np.array([
        #              d3/2^2                          d5/2^2                           s1/2^2                           d3/2 d5/2                        d3/2 s1/2                        d5/2 s1/2
        [2*spe[0] + tbme(0, 0, 0, 0, 0),            tbme(0, 0, 1, 1, 0),            tbme(0, 0, 2, 2, 0),                   tbme(0, 0, 0, 1, 0),                   tbme(0, 0, 0, 2, 0),                   tbme(0, 0, 1, 2, 0)], # d3/2^2
        [           tbme(1, 1, 0, 0, 0), 2*spe[1] + tbme(1, 1, 1, 1, 0),            tbme(1, 1, 2, 2, 0),                   tbme(1, 1, 0, 1, 0),                   tbme(1, 1, 0, 2, 0),                   tbme(1, 1, 1, 2, 0)], # d5/2^2
        [           tbme(2, 2, 0, 0, 0),            tbme(2, 2, 1, 1, 0), 2*spe[2] + tbme(2, 2, 2, 2, 0),                   tbme(2, 2, 0, 1, 0),                   tbme(2, 2, 0, 2, 0),                   tbme(2, 2, 1, 2, 0)], # s1/2^2
        [           tbme(0, 1, 0, 0, 0),            tbme(0, 1, 1, 1, 0),            tbme(0, 1, 2, 2, 0), spe[0] + spe[1] + tbme(0, 1, 0, 1, 0),                   tbme(0, 1, 0, 2, 0),                   tbme(0, 1, 1, 2, 0)], # d3/2 d5/2
        [           tbme(0, 2, 0, 0, 0),            tbme(0, 2, 1, 1, 0),            tbme(0, 2, 2, 2, 0),                   tbme(0, 2, 0, 1, 0), spe[0] + spe[2] + tbme(0, 2, 0, 2, 0),                   tbme(0, 2, 1, 2, 0)], # d3/2 s1/2
        [           tbme(1, 2, 0, 0, 0),            tbme(1, 2, 1, 1, 0),            tbme(1, 2, 2, 2, 0),                   tbme(1, 2, 0, 1, 0),                   tbme(1, 2, 0, 2, 0), spe[1] + spe[2] + tbme(1, 2, 1, 2, 0)], # d5/2 s1/2
    ])
    
    assert np.all(H == H.T)

    eigenvalues, eigenvectors = lalg.eigh(H)
    # print(H)
    print(eigenvalues)
    print(O18.levels)
    # print(eigenvectors)

    return H

def calculate_hamiltonian_dimension(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
) -> tuple[int, int]:

    print(partition_proton.configurations)
    # for orbital in interaction.model_space.orbitals:
    #     orbital.degeneracy

    return 0, 0

def create_hamiltonian(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
):
    n_rows, n_cols = calculate_hamiltonian_dimension(
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )

def main():
    interaction: Interaction = load_interaction(filename_interaction="tmp_delete/w.snt")
    partition_proton, partition_neutron, partition_combined = \
        load_partition(filename_partition="tmp_delete/O18_w_p.ptn", interaction=interaction)
    
    create_hamiltonian(
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )

if __name__ == "__main__":
    main()