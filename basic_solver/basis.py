from itertools import combinations
from kshell_utilities.data_structures import Interaction

def calculate_m_basis_states(
    interaction: Interaction,
    M_target: int,
) -> tuple[tuple[int, ...], ...]:
    """
    Calculate the M-scheme basis states.

    Given a model space and a number of valence nucleons, calculate all
    possible combinations of nucleons whose magnetic substates sum up
    to `M_target`. Consider the following figure which shows the sd
    model space. Note that the ordering of the orbitals is the same
    order as they appear in usda.snt.

                  10    11        
                -  O  -  O  -             s1/2: 2
                 -1/2   1/2

       4     5     6     7     8     9    
    -  O  -  O  -  O  -  O  -  O  -  O    d5/2: 1
     -5/2  -3/2  -1/2   1/2   3/2   5/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2

    This function will iterate over all the possible configurations of
    the valence particles and save the configurations whose m values sum
    up to M_target.

    NOTE: There are good possibilities of parallelising the contents of
    this function, should it be needed.

    Returns
    -------
    index_combinations_filtered : tuple[tuple[int, ...], ...]
        All the possible ways to combine N particles in the given model
        space as to produce M = M_target. Each combination is stored as
        a tuple of N m-substate indices, and each of the tuples are
        nested in a tuple.
    """

    # Generate all combinations of m value indices of length n_valence_nucleons.
    index_combinations = combinations(
        range(len(interaction.model_space_neutron.all_jz_values)),
        interaction.model_space_neutron.n_valence_nucleons,
    )
    
    # Filter combinations so that combinations with m_0 + m_1 + ... == M_target are kept.
    index_combinations_filtered = filter(
        lambda indices: sum(interaction.model_space_neutron.all_jz_values[i] for i in indices) == M_target,
        index_combinations,
    )

    return tuple(index_combinations_filtered)
    # return index_combinations_filtered