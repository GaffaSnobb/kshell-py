from typing import Any
import numpy as np
import numpy.linalg as lalg
import kshell_utilities as ksutil
from kshell_utilities.data_structures import (
    Interaction, Partition, OrbitalParameters
)
from kshell_utilities.loaders import load_interaction, load_partition

def O18_w_manual_hamiltonian() -> np.ndarray[Any, float]:
    """
    Manual construction of the shell model hamiltonian for 18O (2
    valence neutrons) with the w interaction.

    TODO: I made thisn function thinking there were two valence protons
    while it it actually valence neutrons. It still works because the
    interaction and model space treats the protons and neutrons the same
    in this case, but it should be changed so that it is correct.
    """
    O18 = ksutil.loadtxt(path="O18_w/")
    interaction: Interaction = load_interaction(filename_interaction="O18_w/w.snt")
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

def fill_orbitals(
    orbitals: list[OrbitalParameters],
    orbital_occupation: list[tuple[int]],
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
        saved to 'orbital_occupation'. If there are nucleons left, the
        current orbital occupation is not stored and the occupation
        iteration in the previous recursive call is continued.

    Parameters
    ----------
    orbitals:

    """
    if n_remaining_neutrons == 0:
        """
        No more neutrons to place, aka a complete configuration.
        """
        orbital_occupation.append(tuple(current_orbital_occupation))
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
            orbital_occupation = orbital_occupation,
            current_orbital_occupation = current_orbital_occupation,
        )

        current_orbital_occupation[current_orbital_idx] -= occupation

def calculate_hamiltonian_dimension(
    interaction: Interaction,
) -> int:

    print([orb.idx for orb in interaction.model_space_neutron.orbitals])

    current_orbital_occupation: list[int] = [0]*interaction.model_space_neutron.n_valence_nucleons
    orbital_occupation: list[tuple[int]] = []

    fill_orbitals(
        orbitals = interaction.model_space_neutron.orbitals,
        n_remaining_neutrons = interaction.model_space_neutron.n_valence_nucleons,
        n_remaining_holes = sum([orb.degeneracy for orb in interaction.model_space_neutron.orbitals]),
        current_orbital_idx = 0,
        orbital_occupation = orbital_occupation,
        current_orbital_occupation = current_orbital_occupation,
    )
    print(orbital_occupation)
    print(len(orbital_occupation))

    return len(orbital_occupation)

def create_hamiltonian(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
):  
    n_rows_cols = calculate_hamiltonian_dimension(
        interaction = interaction,
    )

def main():
    interaction: Interaction = load_interaction(filename_interaction="O19_w/w.snt")
    partition_proton, partition_neutron, partition_combined = \
        load_partition(filename_partition="O19_w/O19_w_p.ptn", interaction=interaction)
    
    create_hamiltonian(
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )

if __name__ == "__main__":
    main()