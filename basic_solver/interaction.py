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
    all_m_values: tuple[int, ...] = tuple()

    for orbital in interaction.model_space_neutron.orbitals:
        """
        Gather all m (jz) values in a single tuple.
        """
        all_m_values += orbital.jz

    # Generate all combinations of m value indices of length n_valence_nucleons.
    index_combinations = combinations(range(len(all_m_values)), interaction.model_space_neutron.n_valence_nucleons)
    
    # Filter combinations so that combinations with m_0 + m_1 + ... == M_target are kept.
    index_combinations_filtered = filter(
        lambda indices: sum(all_m_values[i] for i in indices) == M_target,
        index_combinations,
    )

    print(f"{all_m_values = }")

    return tuple(index_combinations_filtered)

def hamiltonian_operator(
    interaction: Interaction,
    left_state:  tuple[int, ...],
    right_state: tuple[int, ...],
): 
    """
    A standard shell model Hamiltionian:

        \hat{H}_{\text{TB}} \sum_{a = 0}^{n - 1} \sum_{b = a + 1}^{n - 1} \sum_{c = 0}^{n - 1} \sum_{d = c + 1}^{n - 1} c_a^\dagger c_b^\dagger c_d c_c

    where n is the number of m substates in the model space.

    < left_state | H | right_state >

    Parameters
    ----------
    interaction : Interaction
        The interaction parameters.

    left_state : tuple[int, ...]
        The state in the left position of the inner product. The state
        is represented by its m substate indices organised as a tuple of
        N elements where N is the number of valence nucleons. For a
        system with three valence particles, a tuple might look like

            (0, 3, 7)

        which means that there is a particle in each of the m substates
        0, 3 and 7.

    right_state : tuple[int, ...]
        Same as for left_state but in the right side of the inner
        product.
        
    """
    if len(left_state) != len(right_state):
        msg = (
            "The left state and right state should always be of same length!"
        )
        raise ValueError(msg)

    left_state = left_state[::-1]   # The bra has the reverse ordering of the ket.

    n_m_substates: int = sum([len(orb.jz) for orb in interaction.model_space_neutron.orbitals])
    new_right_state = [0]*len(right_state)

    m_idx_to_orb_idx_mapping = {    # Map the m substate indices to the indices of their orbitals. Hard-coded for sd.
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 2,
        11: 2,
    }
    phase: int = 1  # Can be either +1 or -1 depending on the order of the creation and annihilation operators.

    for a in range(n_m_substates):
        """
        Creation operator: c_a^\dagger.
        """
        for b in range(a + 1, n_m_substates):
            """
            Creation operator: c_b^\dagger.
            """
            for c in range(n_m_substates):
                """
                Annihilation operator: c_c^\dagger.
                """
                if c not in right_state:
                    """
                    The result is zero if the annihilation operator
                    annihilates an already empty magnetic substate. Can
                    prob. calculate the allowed numbers first so this
                    check wont have to be run so often.
                    """
                    continue
                
                for d in range(c + 1, n_m_substates):
                    """
                    Annihilation operator: c_d^\dagger.
                    """
                    if d not in right_state: continue

                    if (a in right_state):
                        """
                        If m substate a already exists, then creating m
                        substate a is going to produce zero unless
                        creation operator c or creation operator d
                        produces the m substate a.
                        """
                        if (a != c) and (a != d):
                            continue

                    if (b in right_state):
                        """
                        Same logic as for a.
                        """
                        if (b != c) and (b != d):
                            continue
                    
                    if (a != c) and (b != d):
                        """
                        Something about that if more than 1 particle
                        changes position, the tbme is 0.
                        TODO: Check this!
                        """
                        continue

                    if (a not in left_state) or (b not in left_state):
                        """
                        The state in the left side of the inner product
                        will bring annihilation operators. If these m
                        substates are not populated, the annihilation
                        operators in the left state will make the inner
                        product 0.

                        NOTE: This is for the 2 valence particles case.
                        if there are more than 2 valence particles, then
                        the right state might originally contain the m
                        substates which the annihilation operators are
                        trying to annihilate, thus everything is a-ok!
                        """
                        continue

                    for m_position, m_idx in enumerate(right_state):
                        """
                        NOTE: It might be a bit confusing, but
                        m_position is the index of the right_state tuple
                        while m_idx is the value at m_position. This is
                        because the elements of the tuple right_state
                        are in fact m substate indices. What we wanna do
                        here is to find the indices of the indices, and
                        I've chosen to call the 'second level' indices
                        for 'positions' so that we can have unique
                        names.

                        TODO: For systems of more than 2 valence
                        particles, the next order has to be checked too.
                        For 2 particles however, we know that if the
                        first annihilation operator has annihilated one
                        of the valence particles successfully, then the
                        there is only one valence particle left and it
                        is therefore in the correct position to be
                        annihilated without producing any change in the
                        phase.
                        
                        To decide the phase, we have to know if the
                        original creation operators in the right state
                        vector have to be swapped so that they are in
                        the correct order to be annihilated. Example:

                                c_x c_y c_x^\dagger c_y^\dagger | core >
                            = - c_x c_y c_y^\dagger c_x^\dagger | core >
                            = - | core >

                        thus, sign = -1.
                        """
                        if c == m_idx:
                            # print(f"{m_position = }")
                            # print(f"{c = }")
                            phase *= (-1)**m_position
                            break

                    for m_position, m_idx in enumerate(left_state):
                        """
                        Same as for the right state.
                        """
                        if b == m_idx:
                            # print(f"{m_position = }")
                            # print(f"{b = }")
                            phase *= (-1)**m_position
                            break

                    o0 = m_idx_to_orb_idx_mapping[a]
                    o1 = m_idx_to_orb_idx_mapping[b]
                    o2 = m_idx_to_orb_idx_mapping[c]
                    o3 = m_idx_to_orb_idx_mapping[d]

                    # print(f"tbme({a}, {b}, {c}, {d})"))
                    print(f"{'+' if phase == +1 else '-'}tbme({o0}, {o1}, {o2}, {o3})")