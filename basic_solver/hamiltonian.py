import time, sys
from bisect import bisect_right
from math import sqrt
import numpy as np
from tqdm import tqdm
from kshell_utilities.data_structures import Interaction
from basis import calculate_m_basis_states
from parameters import (
    clebsch_gordan, clebsch_gordan_array, normalisation_factors
)
from tools import generate_indices
from data_structures import Indices, timings

def calculate_onebody_matrix_element(
    interaction: Interaction,
    indices: Indices,
    left_state_copy: list[int],
    right_state: tuple[int, ...],
) -> float:
    timing = time.perf_counter()
    onebody_res: float = 0.0

    for creation_orb_idx in range(interaction.model_space.n_orbitals):
        """
        More generally, there should also be a loop over the same values
        for annihilation_idx, but the SPEs for most, if not all of the
        interaction files, are only defined for when the two indices are
        the same.
        """
        annihilation_orb_idx = creation_orb_idx
    
        for creation_comp_m_idx in indices.orbital_idx_to_composite_m_idx_map[creation_orb_idx]:
            for annihilation_comp_m_idx in indices.orbital_idx_to_composite_m_idx_map[annihilation_orb_idx]:
                """
                Index scheme for the sd model space:

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

                orbital_idx_to_composite_m_idx_map translates the orbital
                indices to the composite m indices of the magnetic
                substates. For example, proton d3/2 has orbital index 0
                and composite m substate indices 0, 1, 2, 3.
                """
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
    timings.calculate_onebody_matrix_element.time += timing
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

    NOTE: Inlining this code instead of having it as a function has
    almost no impact on program performance. Might as well let it stay
    in this function since it makes the code more readable.
    """
    # annihilation_norm = 1/sqrt(1 + (annihilation_orb_idx_0 == annihilation_orb_idx_1))  # Changing to lookup table had almost no impact.
    
    annihilation_results: list[tuple[float, list[int]]] = []

    for annihilation_comp_m_idx_0 in indices.orbital_idx_to_composite_m_idx_map[annihilation_orb_idx_0]:
        for annihilation_comp_m_idx_1 in indices.orbital_idx_to_composite_m_idx_map[annihilation_orb_idx_1]:

            new_right_state = list(right_state)

            if annihilation_comp_m_idx_0 not in new_right_state: continue

            annihilation_idx = new_right_state.index(annihilation_comp_m_idx_0)
            new_right_state.pop(annihilation_idx)
            annihilation_sign = (-1)**annihilation_idx

            if annihilation_comp_m_idx_1 not in new_right_state: continue

            annihilation_idx = new_right_state.index(annihilation_comp_m_idx_1)
            new_right_state.pop(annihilation_idx)
            annihilation_sign *= (-1)**annihilation_idx

            assert len(new_right_state) == (interaction.model_space.n_valence_nucleons - 2)    # Sanity check. Extremely small impact on program run time.

            cg_annihilation = clebsch_gordan[(
                indices.orbital_idx_to_j_map[annihilation_orb_idx_0],
                indices.composite_m_idx_to_m_map[annihilation_comp_m_idx_0],
                indices.orbital_idx_to_j_map[annihilation_orb_idx_1],
                indices.composite_m_idx_to_m_map[annihilation_comp_m_idx_1],
                j_coupled,
                m_coupled,
            )]

            if cg_annihilation == 0: continue

            # annihilation_results.append((annihilation_sign*annihilation_norm*cg_annihilation, new_right_state))
            annihilation_results.append((annihilation_sign*cg_annihilation, new_right_state))

    return annihilation_results

def twobody_creation_term(
    indices: Indices,
    annihilation_results: list[tuple[float, list[int]]],
    left_state_copy: list[int],
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

    for creation_comp_m_idx_0 in indices.orbital_idx_to_composite_m_idx_map[creation_orb_idx_0]:
        for creation_comp_m_idx_1 in indices.orbital_idx_to_composite_m_idx_map[creation_orb_idx_1]:

            cg_creation = clebsch_gordan[(
                indices.orbital_idx_to_j_map[creation_orb_idx_0],
                indices.composite_m_idx_to_m_map[creation_comp_m_idx_0],
                indices.orbital_idx_to_j_map[creation_orb_idx_1],
                indices.composite_m_idx_to_m_map[creation_comp_m_idx_1],
                j_coupled,
                m_coupled,
            )]

            if cg_creation == 0: continue
            
            for annihilation_coeff, right_state in annihilation_results:

                new_right_state = right_state.copy()

                if creation_comp_m_idx_1 in new_right_state: continue
                created_substate_idx = bisect_right(a=new_right_state, x=creation_comp_m_idx_1)
                new_right_state.insert(created_substate_idx, creation_comp_m_idx_1)
                creation_sign = (-1)**created_substate_idx

                if creation_comp_m_idx_0 in new_right_state: continue
                created_substate_idx = bisect_right(a=new_right_state, x=creation_comp_m_idx_0)
                new_right_state.insert(created_substate_idx, creation_comp_m_idx_0)
                creation_sign *= (-1)**created_substate_idx

                if left_state_copy != new_right_state:
                    continue
                
                creation_res += creation_sign*cg_creation*annihilation_coeff

    return creation_res

def calculate_twobody_matrix_element(
    interaction: Interaction,
    indices: Indices,
    left_state_copy: list[int],
    right_state: tuple[int, ...],
) -> float:
    
    timing = time.perf_counter()
    twobody_res: float = 0.0
    new_right_state: list[int] = []

    for i in range(len(indices.creation_orb_indices_0)):
        """
        The values in the following lists corresponds to using these
        nested loops:

        for creation_orb_idx_0 in range(n_orbitals):
            for creation_orb_idx_1 in range(creation_orb_idx_0, n_orbitals):
                
                for annihilation_orb_idx_0 in range(n_orbitals):
                    for annihilation_orb_idx_1 in range(annihilation_orb_idx_0, n_orbitals):
                    
                        for j_coupled in range(j_min, j_max + 2, 2):
                            for m_coupled in range(-j_coupled, j_coupled + 2, 2):
        
        It gives good reduction in program run time by using
        pre-calculated indices instead of four nested loops.
        """
        creation_orb_idx_0 = indices.creation_orb_indices_0[i]
        creation_orb_idx_1 = indices.creation_orb_indices_1[i]
        annihilation_orb_idx_0 = indices.annihilation_orb_indices_0[i]
        annihilation_orb_idx_1 = indices.annihilation_orb_indices_1[i]
        j_coupled = indices.j_coupled[i]
        m_coupled = indices.m_coupled[i]
        tbme = indices.tbme[i]

        creation_norm = 1/sqrt(1 + (creation_orb_idx_0 == creation_orb_idx_1))
        annihilation_norm = 1/sqrt(1 + (annihilation_orb_idx_0 == annihilation_orb_idx_1))

        # Annihilation terms
        for annihilation_comp_m_idx_0 in indices.orbital_idx_to_composite_m_idx_map[annihilation_orb_idx_0]:
            for annihilation_comp_m_idx_1 in indices.orbital_idx_to_composite_m_idx_map[annihilation_orb_idx_1]:


                if annihilation_comp_m_idx_0 not in right_state: continue
                annihilation_idx = right_state.index(annihilation_comp_m_idx_0)
                
                new_right_state[:] = right_state
                
                new_right_state.pop(annihilation_idx)
                annihilation_sign = (-1)**annihilation_idx

                if annihilation_comp_m_idx_1 not in new_right_state: continue

                annihilation_idx = new_right_state.index(annihilation_comp_m_idx_1)
                new_right_state.pop(annihilation_idx)
                annihilation_sign *= (-1)**annihilation_idx

                assert len(new_right_state) == (interaction.model_space.n_valence_nucleons - 2)    # Sanity check. Extremely small impact on program run time.

                cg_annihilation = clebsch_gordan[(
                    indices.orbital_idx_to_j_map[annihilation_orb_idx_0],
                    indices.composite_m_idx_to_m_map[annihilation_comp_m_idx_0],
                    indices.orbital_idx_to_j_map[annihilation_orb_idx_1],
                    indices.composite_m_idx_to_m_map[annihilation_comp_m_idx_1],
                    j_coupled,
                    m_coupled,
                )]

                if cg_annihilation == 0: continue

                # Creation terms
                creation_res: float = 0.0
                for creation_comp_m_idx_0 in indices.orbital_idx_to_composite_m_idx_map[creation_orb_idx_0]:
                    for creation_comp_m_idx_1 in indices.orbital_idx_to_composite_m_idx_map[creation_orb_idx_1]:

                        cg_creation = clebsch_gordan[(
                            indices.orbital_idx_to_j_map[creation_orb_idx_0],
                            indices.composite_m_idx_to_m_map[creation_comp_m_idx_0],
                            indices.orbital_idx_to_j_map[creation_orb_idx_1],
                            indices.composite_m_idx_to_m_map[creation_comp_m_idx_1],
                            j_coupled,
                            m_coupled,
                        )]

                        if cg_creation == 0: continue

                        new_right_state_copy = new_right_state.copy()

                        if creation_comp_m_idx_1 in new_right_state_copy: continue
                        created_substate_idx = bisect_right(a=new_right_state_copy, x=creation_comp_m_idx_1)
                        new_right_state_copy.insert(created_substate_idx, creation_comp_m_idx_1)
                        creation_sign = (-1)**created_substate_idx

                        if creation_comp_m_idx_0 in new_right_state_copy: continue
                        created_substate_idx = bisect_right(a=new_right_state_copy, x=creation_comp_m_idx_0)
                        new_right_state_copy.insert(created_substate_idx, creation_comp_m_idx_0)
                        creation_sign *= (-1)**created_substate_idx

                        if left_state_copy != new_right_state_copy:
                            continue

                        creation_res += creation_sign*cg_creation

                twobody_res += annihilation_norm*creation_norm*tbme*creation_res*annihilation_sign*cg_annihilation

    timing = time.perf_counter() - timing
    timings.calculate_twobody_matrix_element.time += timing
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
    print(f"{len(indices.creation_orb_indices_0) = }")
    # print(f"{indices.creation_orb_indices_0}")
    
    H = np.zeros((m_dim, m_dim), dtype=np.float64)
    
    with tqdm(total=m_dim**2/2) as pbar:
        for left_idx in range(m_dim):
            """
            Calculate only the upper triangle of the Hamiltonian matrix.
            H is hermitian so we dont have to calculate both triangles.
            """
            left_state_copy = list(basis_states[left_idx])  # For list comparing modified right states.
            for right_idx in range(left_idx, m_dim):

                H[left_idx, right_idx] += calculate_onebody_matrix_element(
                    interaction = interaction,
                    indices = indices,
                    left_state_copy = left_state_copy,
                    right_state = basis_states[right_idx],
                )
                H[left_idx, right_idx] += calculate_twobody_matrix_element(
                    interaction = interaction,
                    indices = indices,
                    left_state_copy = left_state_copy,
                    right_state = basis_states[right_idx],
                )
                pbar.update(1)

    H += H.T - np.diag(np.diag(H))  # Add the lower triangle to complete the matrix.

    np.savetxt(fname="hamham.txt", X=H, fmt="%6.2f")

    timing = time.perf_counter() - timing
    timings.create_hamiltonian.time = timing
    return H