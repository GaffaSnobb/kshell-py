import time
from math import sqrt
import numpy as np
import kshell_utilities as ksutil
from kshell_utilities.data_structures import Interaction
from parameters import clebsch_gordan

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

    if left_state == right_state:
        """
        Only diagonals have single particle energies.
        """
        spe_idx_0 = m_idx_to_orb_idx_mapping[right_state[0]]
        spe_idx_1 = m_idx_to_orb_idx_mapping[right_state[1]]
        res = interaction.spe[spe_idx_0] + interaction.spe[spe_idx_1]
    else:
        res = 0
    
    left_state = left_state[::-1]   # The bra has the reverse ordering of the ket.

    n_m_substates: int = len(interaction.model_space_neutron.all_jz_values)
    # new_right_state = [0]*len(right_state)

    phase: int = 1  # Can be either +1 or -1 depending on the order of the creation and annihilation operators.
    # res = []
    # res = 0.0
    msg = "."

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
                    
                    # if (a != c) and (b != d):
                    #     """
                    #     Something about that if more than 1 particle
                    #     changes position, the tbme is 0.
                    #     TODO: Check this!
                    #     NOTE: Think this should be checked against the orbital indices and not the m substate indices.
                    #     """
                    #     continue

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

                    cg_creation = clebsch_gordan[(
                        interaction.model_space_neutron.orbitals[o0].j,     # j1
                        interaction.model_space_neutron.all_jz_values[a],   # m1
                        interaction.model_space_neutron.orbitals[o1].j,     # j2
                        interaction.model_space_neutron.all_jz_values[b],   # m2
                        0,  # j
                        0,  # m
                    )]
                    cg_annihilation = clebsch_gordan[(
                        interaction.model_space_neutron.orbitals[o2].j,     # j1
                        interaction.model_space_neutron.all_jz_values[c],   # m1
                        interaction.model_space_neutron.orbitals[o3].j,     # j2
                        interaction.model_space_neutron.all_jz_values[d],   # m2
                        0,  # j
                        0,  # m
                    )]

                    norm_creation = 1/sqrt(1 + (o0 == o1))
                    # norm_annihilation = 1/sqrt(1 + (o2 == o3))
                    norm_annihilation = 1
                    cg_annihilation = 1
                    # norm = 1
                    res_tmp = norm_creation*norm_annihilation*cg_creation*cg_annihilation*phase*interaction.tbme.get((o0, o1, o2, o3, 0), 0)
                    res += res_tmp
                    if res_tmp != 0:
                        print(f"{left_state = }")
                        print(f"{right_state = }")
                        print(f"{norm_creation = }")
                        print(f"{norm_annihilation = }")
                        print(f"{cg_creation = }")
                        print(f"{cg_annihilation = }")
                        print(f"{res_tmp = }")
                        print()
                    # msg += f"tbme({a}, {b}, {c}, {d}) "
                    msg += f"{'+' if phase == +1 else '-'}tbme({o0}, {o1}, {o2}, {o3})"

    # print(msg, end="")
    return res
                    # print(f"{'+' if phase == +1 else '-'}tbme({o0}, {o1}, {o2}, {o3})")

def hamiltonian_operator_orbit_loop(
    interaction: Interaction,
    left_state:  tuple[int, ...],
    right_state: tuple[int, ...],
):
    """
    Construct the Hamiltonian with formalism close to that of the
    description in the KSHELL manual.
    """
    res = 0.0
    orb_idx_to_m_idx_mapper = {
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
    n_annihilations: int = 0    # Debug.

    for orb_idx_0 in range(interaction.model_space_neutron.n_orbitals):
        """
        First creation operator loop.
        """
        for orb_idx_1 in range(orb_idx_0, interaction.model_space_neutron.n_orbitals):
            """
            Second creation operator loop.
            """
            for orb_idx_2 in range(interaction.model_space_neutron.n_orbitals):
                """
                First annihilation operator loop.
                """
                # if orb_idx_2 not in right_state: continue
                for orb_idx_3 in range(orb_idx_2, interaction.model_space_neutron.n_orbitals):
                    """
                    Second annihilation operator loop.
                    """
                    # ANNIHILATION OPERATOR
                    norm_annihilation = 1/sqrt(1 + (orb_idx_2 == orb_idx_3))
                    for m_idx_2 in range(interaction.model_space_neutron.orbitals[orb_idx_2].degeneracy):
                        for m_idx_3 in range(interaction.model_space_neutron.orbitals[orb_idx_3].degeneracy):
                            
                            if orb_idx_to_m_idx_mapper[(orb_idx_2, m_idx_2)] not in right_state: continue
                            if orb_idx_to_m_idx_mapper[(orb_idx_3, m_idx_3)] not in right_state: continue
                        
                            cg_annihilation = clebsch_gordan[(
                                interaction.model_space_neutron.orbitals[orb_idx_2].j,              # j1
                                interaction.model_space_neutron.orbitals[orb_idx_2].jz[m_idx_2],    # m1
                                interaction.model_space_neutron.orbitals[orb_idx_3].j,              # j2
                                interaction.model_space_neutron.orbitals[orb_idx_3].jz[m_idx_3],    # m2
                                0,  # j
                                0,  # m
                            )]
                            n_annihilations += 1
                    
                    # CREATION OPERATOR
                    norm_creation = 1/sqrt(1 + (orb_idx_0 == orb_idx_1))
                    # cg_creation = 0.0
                    
                    for m_idx_0 in range(interaction.model_space_neutron.orbitals[orb_idx_0].degeneracy):
                        for m_idx_1 in range(interaction.model_space_neutron.orbitals[orb_idx_1].degeneracy):
                            
                            if orb_idx_to_m_idx_mapper[(orb_idx_0, m_idx_0)] in right_state: continue
                            if orb_idx_to_m_idx_mapper[(orb_idx_1, m_idx_1)] in right_state: continue

                            cg_creation = clebsch_gordan[(
                                interaction.model_space_neutron.orbitals[orb_idx_0].j,              # j1
                                interaction.model_space_neutron.orbitals[orb_idx_0].jz[m_idx_0],    # m1
                                interaction.model_space_neutron.orbitals[orb_idx_1].j,              # j2
                                interaction.model_space_neutron.orbitals[orb_idx_1].jz[m_idx_1],    # m2
                                0,  # j
                                0,  # m
                            )]
                    
                    interaction.tbme.get((orb_idx_0, orb_idx_1, orb_idx_2, orb_idx_3, 0), 0)



    print(f"{n_annihilations = }")
    return res