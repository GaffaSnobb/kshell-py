from math import sqrt
import numpy as np
from kshell_utilities.data_structures import Interaction
from basis import calculate_m_basis_states
from parameters import clebsch_gordan

orbital_idx_to_m_idx_map = [
    range(4),   # d3/2 has 4 m substates
    range(6),
    range(2),
]
orbital_m_pair_to_composite_m_idx_map = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (1, 3): 7,
    (1, 4): 8,
    (1, 5): 9,
    (2, 0): 10,
    (2, 1): 11,
}

j_idx_to_j_map = [3, 5, 1] # 0 is 3/2, 1 is 5/2, 0 is s1/2

m_indices = [
    range(4),   # d3/2 has 4 m substates
    range(6),
    range(2),
]

orbital_m_pair_to_m_idx = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (1, 3): 7,
    (1, 4): 8,
    (1, 5): 9,
    (2, 0): 10,
    (2, 1): 11,
}

m_comp_idx_to_m_map = {
    0:  -3,
    1:  -1,
    2:  +1,
    3:  +3,
    4:  -5,
    5:  -3,
    6:  -1,
    7:  +1,
    8:  +3,
    9:  +5,
    10: -1,
    11: +1
}

def calculate_onebody_matrix_element(
    interaction: Interaction,
    left_state: tuple[int, ...],
    right_state: tuple[int, ...],
) -> float:

    onebody_res: float = 0.0
    left_state_copy = list(left_state)  # I want this as a list so that I can perform list comparisons. Will not be modified.

    # for creation_orb_idx in range(interaction.model_space_neutron.n_orbitals):
    for creation_orb_idx in range(3):
        """
        More generally, there should also be a loop over the same values
        for annihilation_idx, but the SPEs for most, if not all of the
        interaction files, are only defined for when the two indices are
        the same.
        """
        annihilation_orb_idx = creation_orb_idx
    
        for creation_m_idx in orbital_idx_to_m_idx_map[creation_orb_idx]:
            """
            Here is an overview of the indices of the sd model space:

                          10    11        
                        -  O  -  O  -             s1/2: 2
                         -1/2   1/2

               4     5     6     7     8     9    
            -  O  -  O  -  O  -  O  -  O  -  O    d5/2: 1
             -5/2  -3/2  -1/2   1/2   3/2   5/2

                     0     1     2     3
                  -  O  -  O  -  O  -  O  -       d3/2: 0
                   -3/2  -1/2   1/2   3/2
            
            At this point, creation_m_idx is the m index within an
            orbital.

            Example:
                d3/2 gives: 0, 1, 2, 3
                d5/2 gives: 0, 1, 2, 3, 4, 5
                s1/2 gives: 0, 1

            Note that the notation of the basis states uses a composite
            m index which for the sd model space will be 0, 1, ..., 11
            where

                d3/2: 0, 1, 2, 3
                d5/2: 4, 5, 6, 7, 8, 9
                s1/2: 10, 11
            
            This means that creation_m_idx has to be translated to the
            composite m index where both the index of the orbital and
            the index of the orbitals m substate is needed for the
            translation.
            
            Example:
                (0, 0): 0   # Orbital 0 is d3/2, m index 0 is -3/2.
                ...
                (1, 0): 4   # Orbital 1 is d5/2, m index 0 is -5/2.
            """
            creation_comp_m_idx = orbital_m_pair_to_composite_m_idx_map[(creation_orb_idx, creation_m_idx)]
            
            for annihilation_m_idx in orbital_idx_to_m_idx_map[annihilation_orb_idx]:
                """
                Same translation for the annihilation_m_idx as for the
                creation.
                """
                annihilation_comp_m_idx = orbital_m_pair_to_composite_m_idx_map[(annihilation_orb_idx, annihilation_m_idx)]

                new_right_state = list(right_state)
                
                try:
                    """
                    We need the index of the substate which will be
                    annihalated because there might be a phase of -1.
                    This is because we have to make sure that the
                    annihilation operator is placed next to the creation
                    operator it tries to annihilate:

                        c_0 | (0, 3) > = c_0 c_0^\dagger c_3^\dagger | core >
                                       = c_3^\dagger | core >
                                       = | (3) >

                        c_0 | (3, 0) > = c_0 c_3^\dagger c_0^\dagger | core >
                                       = - c_0 c_0^\dagger c_3^\dagger | core >
                                       = - c_3^\dagger | core >
                                       = - | (3) >
                    """
                    annihalated_substate_idx = new_right_state.index(annihilation_comp_m_idx)
                except ValueError:
                    """
                    If the index cannot be found, then it does not exist
                    in the list. Aka. we are trying to annihilate a
                    substate which is un-occupied and the result is
                    zero.
                    """
                    continue
                
                annihilation_sign = (-1)**annihalated_substate_idx
                new_right_state.pop(annihalated_substate_idx)

                if creation_comp_m_idx in new_right_state: continue

                new_right_state.insert(0, creation_comp_m_idx)

                if left_state_copy == new_right_state:
                    """
                    NOTE: This only works for two valence particles. Has
                    to be generalised for N valence particles.
                    """
                    creation_sign = 1
                elif left_state_copy == new_right_state[::-1]:
                    creation_sign = -1
                else:
                    continue

                onebody_res += annihilation_sign*creation_sign*interaction.spe[creation_orb_idx] # Or annihilation_orb_idx, they are the same.

    return onebody_res

def calculate_twobody_matrix_element(
    interaction: Interaction,
    left_state: tuple[int, ...],
    right_state: tuple[int, ...],
) -> float:
    
    twobody_res: float = 0.0
    n_orbitals = interaction.model_space_neutron.n_orbitals # Just to make the name shorter.
    
    for creation_orb_idx_0 in range(n_orbitals):
        for creation_orb_idx_1 in range(creation_orb_idx_0, n_orbitals):
            for annihilation_orb_idx_0 in range(n_orbitals):
                for annihilation_orb_idx_1 in range(annihilation_orb_idx_0, n_orbitals):

                    j_min = max(
                        abs(j_idx_to_j_map[creation_orb_idx_0] - j_idx_to_j_map[creation_orb_idx_1]),
                        abs(j_idx_to_j_map[annihilation_orb_idx_0] - j_idx_to_j_map[annihilation_orb_idx_1]),
                    )
                    j_max = min(
                        j_idx_to_j_map[creation_orb_idx_0] + j_idx_to_j_map[creation_orb_idx_1],
                        j_idx_to_j_map[annihilation_orb_idx_0] + j_idx_to_j_map[annihilation_orb_idx_1],
                    )

                    for j_coupled in range(j_min, j_max + 2, 2):
                        """
                        j_coupled is the total angular momentum to which 
                        (creation_orb_idx_0, creation_orb_idx_1) and
                        (annihilation_orb_idx_0, annihilation_orb_idx_1)
                        couple. Follows the standard angular momentum
                        coupling rules:

                            j = | j1 - j2 |, | j1 - j2 | + 1, ..., j1 + j2

                        but we have to respect the allowed range for
                        both pairs of total angular momentum values so
                        that j_coupled is contained in both ranges.

                        Step length of 2 because all angular momentum
                        values are multiplied by 2 to avoid fractions.
                        + 2 so that the end point is included.
                        """
                        for m_coupled in range(-j_coupled, j_coupled + 2, 2):
                            """
                            m_coupled is simply the z component of the
                            coupled total angular momentum, j_coupled.
                            """
                            try:
                                tbme = interaction.tbme[(
                                    creation_orb_idx_0,
                                    creation_orb_idx_1,
                                    annihilation_orb_idx_0,
                                    annihilation_orb_idx_1,
                                    j_coupled
                                )]
                            except KeyError:
                                """
                                If the current interaction file is not
                                defining any two-body matrix elements
                                for this choice of orbitals and coupled
                                j then the result is 0.
                                """
                                continue

                            # Annihilation term:
                            annihilation_norm = 1/sqrt(1 + (annihilation_orb_idx_0 == annihilation_orb_idx_1))
                            
                            annihilation_results = []
                            
                            for annihilation_m_idx_0 in m_indices[annihilation_orb_idx_0]:
                                """
                                See the docstrings in
                                calculate_onebody_matrix_element for a
                                description on whats going on with these
                                indices.
                                """
                                annihilation_comp_m_idx_0 = orbital_m_pair_to_m_idx[(annihilation_orb_idx_0, annihilation_m_idx_0)]
                                
                                for annihilation_m_idx_1 in m_indices[annihilation_orb_idx_1]:
                                    annihilation_comp_m_idx_1 = orbital_m_pair_to_m_idx[(annihilation_orb_idx_1, annihilation_m_idx_1)]

                                    new_right_state = list(right_state)

                                    if annihilation_comp_m_idx_0 not in new_right_state: continue

                                    annihilation_idx = new_right_state.index(annihilation_comp_m_idx_0)
                                    new_right_state.pop(annihilation_idx)
                                    annihilation_sign = (-1)**annihilation_idx

                                    if annihilation_comp_m_idx_1 not in new_right_state: continue

                                    new_right_state.remove(annihilation_comp_m_idx_1)

                                    assert len(new_right_state) == 0    # Sanity check.

                                    cg_annihilation = clebsch_gordan[(
                                        j_idx_to_j_map[annihilation_orb_idx_0],
                                        m_comp_idx_to_m_map[annihilation_comp_m_idx_0],
                                        j_idx_to_j_map[annihilation_orb_idx_1],
                                        m_comp_idx_to_m_map[annihilation_comp_m_idx_1],
                                        j_coupled,
                                        m_coupled,
                                    )]

                                    if cg_annihilation == 0: continue

                                    annihilation_results.append(annihilation_sign*annihilation_norm*cg_annihilation)

                            # Creation term:
                            creation_norm = 1/sqrt(1 + (creation_orb_idx_0 == creation_orb_idx_1))  # TODO: Move this!

                            for creation_m_idx_0 in m_indices[creation_orb_idx_0]:
                                creation_comp_m_idx_0 = orbital_m_pair_to_m_idx[(creation_orb_idx_0, creation_m_idx_0)]

                                for creation_m_idx_1 in m_indices[creation_orb_idx_1]:
                                    creation_comp_m_idx_1 = orbital_m_pair_to_m_idx[(creation_orb_idx_1, creation_m_idx_1)]

                                    cg_creation = clebsch_gordan[(
                                        j_idx_to_j_map[creation_orb_idx_0],
                                        m_comp_idx_to_m_map[creation_comp_m_idx_0],
                                        j_idx_to_j_map[creation_orb_idx_1],
                                        m_comp_idx_to_m_map[creation_comp_m_idx_1],
                                        j_coupled,
                                        m_coupled,
                                    )]
                                    if cg_creation == 0: continue

                                    new_right_state = []
                                    tmp_left_state = list(left_state)   # TODO: Move this!

                                    new_right_state.insert(0, creation_comp_m_idx_1)
                                    new_right_state.insert(0, creation_comp_m_idx_0)

                                    if tmp_left_state == new_right_state:
                                        creation_sign = 1
                                    elif tmp_left_state == new_right_state[::-1]:
                                        creation_sign = -1
                                    else:
                                        continue
                                    
                                    for annihilation_result in annihilation_results:
                                        twobody_res += creation_sign*tbme*creation_norm*cg_creation*annihilation_result

    return twobody_res

def create_hamiltonian(
    interaction: Interaction,
) -> np.ndarray:
    """
    """


    basis_states = calculate_m_basis_states(interaction=interaction, M_target=0)
    print(basis_states)
    m_dim = len(basis_states)   # This is the 'M-scheme dimension'. The H matrix, if represented in its entirety, is of dimensions m_dim x m_dim.

    H = np.zeros((m_dim, m_dim), dtype=np.float64)

    for left_idx in range(m_dim):
        for right_idx in range(m_dim):

            H[left_idx, right_idx] += calculate_onebody_matrix_element(
                interaction = interaction,
                left_state = basis_states[left_idx],
                right_state = basis_states[right_idx],
            )
            H[left_idx, right_idx] += calculate_twobody_matrix_element(
                interaction = interaction,
                left_state = basis_states[left_idx],
                right_state = basis_states[right_idx],
            )

    return H