import time
from itertools import combinations
from kshell_utilities.data_structures import Interaction
from data_structures import timings

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

                  22    23        
                -  O  -  O  -             neutron s1/2: 5
                 -1/2   1/2

      16    17    18    19    20    21    
    -  O  -  O  -  O  -  O  -  O  -  O    neutron d5/2: 4
     -5/2  -3/2  -1/2   1/2   3/2   5/2

            12    13    14    15
          -  O  -  O  -  O  -  O  -       neutron d3/2: 3
           -3/2  -1/2   1/2   3/2

                  10    11        
                -  O  -  O  -             proton s1/2: 2
                 -1/2   1/2

       4     5     6     7     8     9    
    -  O  -  O  -  O  -  O  -  O  -  O    proton d5/2: 1
     -5/2  -3/2  -1/2   1/2   3/2   5/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       proton d3/2: 0
           -3/2  -1/2   1/2   3/2

    This function will iterate over all the possible configurations of
    the valence particles and save the configurations whose m values sum
    up to M_target.

    NOTE: There are good possibilities of parallelising the contents of
    this function, should it be needed.

    Returns
    -------
    basis_states : tuple[tuple[int, ...], ...]
        All the possible ways to combine N particles in the given model
        space as to produce M = M_target. Each combination is stored as
        a tuple of N m-substate indices, and each of the tuples are
        nested in a tuple.
    """
    timing = time.perf_counter()
    n_proton_m_substates: int = len(interaction.model_space_proton.all_jz_values)
    n_neutron_m_substates: int = len(interaction.model_space_neutron.all_jz_values)

    # Generate all combinations of m substate indices of length n_valence_nucleons.
    proton_index_combinations = combinations(
        iterable = range(n_proton_m_substates),
        r = interaction.model_space_proton.n_valence_nucleons,
    )
    neutron_index_combinations = tuple(combinations(    # One of these generators must be converted to a list / tuple, else they will get exhausted in the first iteration of the combined generator.
        iterable = range(
            n_proton_m_substates,
            n_proton_m_substates + n_neutron_m_substates,
        ),
        r = interaction.model_space_neutron.n_valence_nucleons,
    ))
    combined_index_combinations = ((p + n) for p in proton_index_combinations for n in neutron_index_combinations)
    
    # Filter combinations so that combinations with m_0 + m_1 + ... == M_target are kept.
    basis_states = tuple(filter(
        lambda indices: sum(interaction.model_space.all_jz_values[i] for i in indices) == M_target,
        combined_index_combinations,
    ))
    timing = time.perf_counter() - timing
    timings.calculate_m_basis_states_002 = timing

    return basis_states