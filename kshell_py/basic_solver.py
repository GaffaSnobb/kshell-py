from typing import Any
import numpy as np
import numpy.linalg as lalg
from kshell_utilities.data_structures import Interaction
from kshell_utilities.loaders import load_interaction, load_partition

def construct_hamiltonian(
    interaction: Interaction
) -> np.ndarray[Any, float]:
    
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

    return H

def main():
    interaction: Interaction = load_interaction(filename_interaction="tmp_delete/w.snt")
    partition_proton, partition_neutron, partition_combined = \
        load_partition(filename_partition="tmp_delete/O18_w_p.ptn", interaction=interaction)
    



    H: np.ndarray = construct_hamiltonian(interaction=interaction)
    print(H)
    # eigenvalues, eigenvectors = lalg.eigh(H)

if __name__ == "__main__":
    main()