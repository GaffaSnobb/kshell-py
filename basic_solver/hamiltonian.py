import time, sys
from bisect import bisect_right
from math import sqrt
import numpy as np
from tqdm import tqdm
from kshell_utilities.data_structures import Interaction
from basis import calculate_m_basis_states
from parameters import clebsch_gordan
from tools import generate_indices
from data_structures import Indices, timings

def calculate_onebody_matrix_element(
    interaction: Interaction,
    indices: Indices,
    left_state: tuple[int, ...],
    right_state: tuple[int, ...],
) -> float:
    timing = time.perf_counter()
    onebody_res: float = 0.0
    left_state_copy = list(left_state)  # I want this as a list so that I can perform list comparisons. Will not be modified.

    for creation_orb_idx in range(interaction.model_space.n_orbitals):
        """
        More generally, there should also be a loop over the same values
        for annihilation_idx, but the SPEs for most, if not all of the
        interaction files, are only defined for when the two indices are
        the same.
        """
        annihilation_orb_idx = creation_orb_idx
    
        for creation_m_idx in indices.orbital_idx_to_m_idx_map[creation_orb_idx]:
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
            creation_comp_m_idx = indices.orbital_m_pair_to_composite_m_idx_map[(creation_orb_idx, creation_m_idx)]
            
            for annihilation_m_idx in indices.orbital_idx_to_m_idx_map[annihilation_orb_idx]:
                """
                Same translation for the annihilation_m_idx as for the
                creation.
                """
                annihilation_comp_m_idx = indices.orbital_m_pair_to_composite_m_idx_map[(annihilation_orb_idx, annihilation_m_idx)]

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
                
                created_substate_idx = bisect_right(a=new_right_state, x=creation_comp_m_idx)
                new_right_state.insert(created_substate_idx, creation_comp_m_idx)
                creation_sign = (-1)**created_substate_idx

                if left_state_copy != new_right_state:
                    continue

                onebody_res += annihilation_sign*creation_sign*interaction.spe[creation_orb_idx] # Or annihilation_orb_idx, they are the same.

    timing = time.perf_counter() - timing
    timings.calculate_onebody_matrix_element_003 += timing
    return onebody_res

def twobody_annihilation_term(
    interaction: Interaction,
    indices: Indices,
    right_state: tuple[int, ...],
    annihilation_orb_idx_0: int,
    annihilation_orb_idx_1: int,
    j_coupled: int,
    m_coupled: int,
) -> list[tuple[float, list[int]]]:
    """
    All calculations related to the annihilation term in the two-body
    part of the Hamiltonian.
    """
    annihilation_norm = 1/sqrt(1 + (annihilation_orb_idx_0 == annihilation_orb_idx_1))
    
    annihilation_results: list[tuple[float, list[int]]] = []
    
    for annihilation_m_idx_0 in indices.orbital_idx_to_m_idx_map[annihilation_orb_idx_0]:
        """
        See the docstrings in
        calculate_onebody_matrix_element for a
        description on whats going on with these
        indices.
        """
        annihilation_comp_m_idx_0 = indices.orbital_m_pair_to_composite_m_idx_map[(annihilation_orb_idx_0, annihilation_m_idx_0)]
        
        for annihilation_m_idx_1 in indices.orbital_idx_to_m_idx_map[annihilation_orb_idx_1]:
            annihilation_comp_m_idx_1 = indices.orbital_m_pair_to_composite_m_idx_map[(annihilation_orb_idx_1, annihilation_m_idx_1)]

            new_right_state = list(right_state)

            if annihilation_comp_m_idx_0 not in new_right_state: continue

            annihilation_idx = new_right_state.index(annihilation_comp_m_idx_0)
            new_right_state.pop(annihilation_idx)
            annihilation_sign = (-1)**annihilation_idx

            if annihilation_comp_m_idx_1 not in new_right_state: continue

            # new_right_state.remove(annihilation_comp_m_idx_1)

            annihilation_idx = new_right_state.index(annihilation_comp_m_idx_1)
            new_right_state.pop(annihilation_idx)
            annihilation_sign *= (-1)**annihilation_idx

            assert len(new_right_state) == (interaction.model_space.n_valence_nucleons - 2)    # Sanity check.

            cg_annihilation = clebsch_gordan[(
                indices.orbital_idx_to_j_map[annihilation_orb_idx_0],
                indices.m_composite_idx_to_m_map[annihilation_comp_m_idx_0],
                indices.orbital_idx_to_j_map[annihilation_orb_idx_1],
                indices.m_composite_idx_to_m_map[annihilation_comp_m_idx_1],
                j_coupled,
                m_coupled,
            )]

            if cg_annihilation == 0: continue

            annihilation_results.append((annihilation_sign*annihilation_norm*cg_annihilation, new_right_state))

    return annihilation_results

def twobody_creation_term(
    indices: Indices,
    annihilation_results: list[tuple[float, list[int]]],
    left_state: tuple[int, ...],
    creation_orb_idx_0: int,
    creation_orb_idx_1: int,
    j_coupled: int,
    m_coupled: int,
) -> float:
    """
    All calculations related to the creation term in the two-body part
    of the Hamiltonian.
    """
    creation_res: float = 0.0
    creation_norm = 1/sqrt(1 + (creation_orb_idx_0 == creation_orb_idx_1))  # TODO: Move this!

    for creation_m_idx_0 in indices.orbital_idx_to_m_idx_map[creation_orb_idx_0]:
        creation_comp_m_idx_0 = indices.orbital_m_pair_to_composite_m_idx_map[(creation_orb_idx_0, creation_m_idx_0)]

        for creation_m_idx_1 in indices.orbital_idx_to_m_idx_map[creation_orb_idx_1]:
            creation_comp_m_idx_1 = indices.orbital_m_pair_to_composite_m_idx_map[(creation_orb_idx_1, creation_m_idx_1)]

            cg_creation = clebsch_gordan[(
                indices.orbital_idx_to_j_map[creation_orb_idx_0],
                indices.m_composite_idx_to_m_map[creation_comp_m_idx_0],
                indices.orbital_idx_to_j_map[creation_orb_idx_1],
                indices.m_composite_idx_to_m_map[creation_comp_m_idx_1],
                j_coupled,
                m_coupled,
            )]
            if cg_creation == 0: continue
            
            for annihilation_coeff, right_state in annihilation_results:

                new_right_state = right_state.copy()
                tmp_left_state = list(left_state)   # TODO: Move this!

                if creation_comp_m_idx_1 in new_right_state: continue
                created_substate_idx = bisect_right(a=new_right_state, x=creation_comp_m_idx_1)
                new_right_state.insert(created_substate_idx, creation_comp_m_idx_1)
                creation_sign = (-1)**created_substate_idx

                if creation_comp_m_idx_0 in new_right_state: continue
                created_substate_idx = bisect_right(a=new_right_state, x=creation_comp_m_idx_0)
                new_right_state.insert(created_substate_idx, creation_comp_m_idx_0)
                creation_sign *= (-1)**created_substate_idx

                if tmp_left_state != new_right_state:
                    continue
                
                creation_res += creation_sign*creation_norm*cg_creation*annihilation_coeff

    return creation_res

def calculate_twobody_matrix_element(
    interaction: Interaction,
    indices: Indices,
    left_state: tuple[int, ...],
    right_state: tuple[int, ...],
) -> float:
    
    timing = time.perf_counter()
    twobody_res: float = 0.0
    n_orbitals = interaction.model_space.n_orbitals # Just to make the name shorter.
    
    for creation_orb_idx_0 in range(n_orbitals):
        for creation_orb_idx_1 in range(creation_orb_idx_0, n_orbitals):
            for annihilation_orb_idx_0 in range(n_orbitals):
                for annihilation_orb_idx_1 in range(annihilation_orb_idx_0, n_orbitals):

                    j_min = max(
                        abs(indices.orbital_idx_to_j_map[creation_orb_idx_0] - indices.orbital_idx_to_j_map[creation_orb_idx_1]),
                        abs(indices.orbital_idx_to_j_map[annihilation_orb_idx_0] - indices.orbital_idx_to_j_map[annihilation_orb_idx_1]),
                    )
                    j_max = min(
                        indices.orbital_idx_to_j_map[creation_orb_idx_0] + indices.orbital_idx_to_j_map[creation_orb_idx_1],
                        indices.orbital_idx_to_j_map[annihilation_orb_idx_0] + indices.orbital_idx_to_j_map[annihilation_orb_idx_1],
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

                            annihilation_results = twobody_annihilation_term(
                                interaction = interaction,
                                indices = indices,
                                right_state = right_state,
                                annihilation_orb_idx_0 = annihilation_orb_idx_0,
                                annihilation_orb_idx_1 = annihilation_orb_idx_1,
                                j_coupled = j_coupled,
                                m_coupled = m_coupled,
                            )
                            twobody_res += tbme*twobody_creation_term(
                                indices = indices,
                                annihilation_results = annihilation_results,
                                left_state = left_state,
                                creation_orb_idx_0 = creation_orb_idx_0,
                                creation_orb_idx_1 = creation_orb_idx_1,
                                j_coupled = j_coupled,
                                m_coupled = m_coupled,
                            )

    timing = time.perf_counter() - timing
    timings.calculate_twobody_matrix_element_004 += timing
    return twobody_res

def create_hamiltonian(
    interaction: Interaction,
) -> np.ndarray:
    """
    """
    timing = time.perf_counter()

    if interaction.tbme_mass_dependence_method == 0:
        """
        No mass dependence on the TBMEs
        """
        pass
    
    elif interaction.tbme_mass_dependence_method == 1:
        """
        The TBMEs need to be scaled according to mass dependence 1.
        """
        nucleus_mass = interaction.n_core_neutrons + interaction.n_core_protons + interaction.model_space.n_valence_nucleons
        factor = (nucleus_mass/interaction.tbme_mass_dependence_denominator)**interaction.tbme_mass_dependence_exponent
        for key in interaction.tbme:
            interaction.tbme[key] *= factor

    if interaction.model_space.n_valence_nucleons%2 == 0:
        """
        For an even number of valence nucleons, the M = 0 basis states
        are enough to describe all angular momenta.
        """
        M_target = 0
    else:
        """
        For odd-numbered M = 1/2 is enough, but remember, all angular
        momenta in this code are multiplied by 2.
        """
        M_target = 1
    
    indices: Indices = generate_indices(interaction=interaction)
    basis_states = calculate_m_basis_states(interaction=interaction, M_target=M_target)
    m_dim = len(basis_states)   # This is the 'M-scheme dimension'. The H matrix, if represented in its entirety, is of dimensions m_dim x m_dim.
    
    print(basis_states)
    print(f"{m_dim = }")
    
    H = np.zeros((m_dim, m_dim), dtype=np.float64)
    
    with tqdm(total=m_dim**2) as pbar:
        for left_idx in range(m_dim):
            for right_idx in range(m_dim):

                H[left_idx, right_idx] += calculate_onebody_matrix_element(
                    interaction = interaction,
                    indices = indices,
                    left_state = basis_states[left_idx],
                    right_state = basis_states[right_idx],
                )
                H[left_idx, right_idx] += calculate_twobody_matrix_element(
                    interaction = interaction,
                    indices = indices,
                    left_state = basis_states[left_idx],
                    right_state = basis_states[right_idx],
                )
            pbar.update(m_dim)
            

    timing = time.perf_counter() - timing
    timings.create_hamiltonian_000 = timing
    return H