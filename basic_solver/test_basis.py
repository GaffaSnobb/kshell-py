from kshell_utilities.data_structures import Interaction
from basis import calculate_m_basis_states

def test_calculate_O18_m_basis_states():
    """
    Test that calculate_m_basis_states indeed calculates the correct
    basis states for a few different selections of valence nucleons and
    model spaces.
    """
    interaction = Interaction()
    n_valence_protons = 0
    n_valence_neutrons = 2
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    interaction.model_space.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Protons
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Neutrons
    ]
    interaction.model_space_proton.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    interaction.model_space_neutron.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    expected_basis_states = (
        (12, 15), (12, 20), (13, 14), (13, 19), (13, 23), (14, 18), (14, 22),
        (15, 17), (16, 21), (17, 20), (18, 19), (18, 23), (19, 22), (22, 23),
    )

    calculated_basis_states = calculate_m_basis_states(interaction=interaction, M_target=0)

    for expected, calculated in zip(expected_basis_states, calculated_basis_states):
        assert expected == calculated

def test_calculate_O19_m_basis_states():
    interaction = Interaction()
    n_valence_protons = 0
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    interaction.model_space.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Protons
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Neutrons
    ]
    interaction.model_space_proton.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    interaction.model_space_neutron.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    expected_basis_states = (
        (12, 13, 21), (12, 14, 15), (12, 14, 20), (12, 15, 19), (12, 15, 23),
        (12, 18, 21), (12, 19, 20), (12, 20, 23), (12, 21, 22), (13, 14, 19),
        (13, 14, 23), (13, 15, 18), (13, 15, 22), (13, 17, 21), (13, 18, 20),
        (13, 19, 23), (13, 20, 22), (14, 15, 17), (14, 16, 21), (14, 17, 20),
        (14, 18, 19), (14, 18, 23), (14, 19, 22), (14, 22, 23), (15, 16, 20),
        (15, 17, 19), (15, 17, 23), (15, 18, 22), (16, 19, 21), (16, 21, 23),
        (17, 18, 21), (17, 19, 20), (17, 20, 23), (17, 21, 22), (18, 19, 23),
        (18, 20, 22), (19, 22, 23),
    )

    calculated_basis_states = calculate_m_basis_states(interaction=interaction, M_target=1)

    for expected, calculated in zip(expected_basis_states, calculated_basis_states):
        assert expected == calculated

def test_calculate_O20_m_basis_states():
    interaction = Interaction()
    n_valence_protons = 0
    n_valence_neutrons = 4
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    interaction.model_space.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Protons
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Neutrons
    ]
    interaction.model_space_proton.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    interaction.model_space_neutron.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    expected_basis_states = (
        (12, 13, 14, 15), (12, 13, 14, 20), (12, 13, 15, 19), (12, 13, 15, 23),
        (12, 13, 18, 21), (12, 13, 19, 20), (12, 13, 20, 23), (12, 13, 21, 22),
        (12, 14, 15, 18), (12, 14, 15, 22), (12, 14, 17, 21), (12, 14, 18, 20),
        (12, 14, 19, 23), (12, 14, 20, 22), (12, 15, 16, 21), (12, 15, 17, 20),
        (12, 15, 18, 19), (12, 15, 18, 23), (12, 15, 19, 22), (12, 15, 22, 23),
        (12, 16, 20, 21), (12, 17, 19, 21), (12, 17, 21, 23), (12, 18, 19, 20),
        (12, 18, 20, 23), (12, 18, 21, 22), (12, 19, 20, 22), (12, 20, 22, 23),
        (13, 14, 15, 17), (13, 14, 16, 21), (13, 14, 17, 20), (13, 14, 18, 19),
        (13, 14, 18, 23), (13, 14, 19, 22), (13, 14, 22, 23), (13, 15, 16, 20),
        (13, 15, 17, 19), (13, 15, 17, 23), (13, 15, 18, 22), (13, 16, 19, 21),
        (13, 16, 21, 23), (13, 17, 18, 21), (13, 17, 19, 20), (13, 17, 20, 23),
        (13, 17, 21, 22), (13, 18, 19, 23), (13, 18, 20, 22), (13, 19, 22, 23),
        (14, 15, 16, 19), (14, 15, 16, 23), (14, 15, 17, 18), (14, 15, 17, 22),
        (14, 16, 18, 21), (14, 16, 19, 20), (14, 16, 20, 23), (14, 16, 21, 22),
        (14, 17, 18, 20), (14, 17, 19, 23), (14, 17, 20, 22), (14, 18, 19, 22),
        (14, 18, 22, 23), (15, 16, 17, 21), (15, 16, 18, 20), (15, 16, 19, 23),
        (15, 16, 20, 22), (15, 17, 18, 19), (15, 17, 18, 23), (15, 17, 19, 22),
        (15, 17, 22, 23), (16, 17, 20, 21), (16, 18, 19, 21), (16, 18, 21, 23),
        (16, 19, 20, 23), (16, 19, 21, 22), (16, 21, 22, 23), (17, 18, 19, 20),
        (17, 18, 20, 23), (17, 18, 21, 22), (17, 19, 20, 22), (17, 20, 22, 23),
        (18, 19, 22, 23),
    )

    calculated_basis_states = calculate_m_basis_states(interaction=interaction, M_target=0)

    for expected, calculated in zip(expected_basis_states, calculated_basis_states):
        assert expected == calculated

def test_calculate_F19_m_basis_states():
    interaction = Interaction()
    n_valence_protons = 1
    n_valence_neutrons = 2
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    interaction.model_space.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Protons
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1, # Neutrons
    ]
    interaction.model_space_proton.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    interaction.model_space_neutron.all_jz_values = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    expected_basis_states = (
        (0, 13, 21), (0, 14, 15), (0, 14, 20), (0, 15, 19), (0, 15, 23),
        (0, 18, 21), (0, 19, 20), (0, 20, 23), (0, 21, 22), (1, 12, 21),
        (1, 13, 15), (1, 13, 20), (1, 14, 19), (1, 14, 23), (1, 15, 18),
        (1, 15, 22), (1, 17, 21), (1, 18, 20), (1, 19, 23), (1, 20, 22),
        (2, 12, 15), (2, 12, 20), (2, 13, 14), (2, 13, 19), (2, 13, 23), 
        (2, 14, 18), (2, 14, 22), (2, 15, 17), (2, 16, 21), (2, 17, 20),
        (2, 18, 19), (2, 18, 23), (2, 19, 22), (2, 22, 23), (3, 12, 14),
        (3, 12, 19), (3, 12, 23), (3, 13, 18), (3, 13, 22), (3, 14, 17),
        (3, 15, 16), (3, 16, 20), (3, 17, 19), (3, 17, 23), (3, 18, 22),
        (4, 14, 21), (4, 15, 20), (4, 19, 21), (4, 21, 23), (5, 13, 21),
        (5, 14, 15), (5, 14, 20), (5, 15, 19), (5, 15, 23), (5, 18, 21),
        (5, 19, 20), (5, 20, 23), (5, 21, 22), (6, 12, 21), (6, 13, 15),
        (6, 13, 20), (6, 14, 19), (6, 14, 23), (6, 15, 18), (6, 15, 22),
        (6, 17, 21), (6, 18, 20), (6, 19, 23), (6, 20, 22), (7, 12, 15),
        (7, 12, 20), (7, 13, 14), (7, 13, 19), (7, 13, 23), (7, 14, 18),
        (7, 14, 22), (7, 15, 17), (7, 16, 21), (7, 17, 20), (7, 18, 19),
        (7, 18, 23), (7, 19, 22), (7, 22, 23), (8, 12, 14), (8, 12, 19),
        (8, 12, 23), (8, 13, 18), (8, 13, 22), (8, 14, 17), (8, 15, 16),
        (8, 16, 20), (8, 17, 19), (8, 17, 23), (8, 18, 22), (9, 12, 13),
        (9, 12, 18), (9, 12, 22), (9, 13, 17), (9, 14, 16), (9, 16, 19),
        (9, 16, 23), (9, 17, 18), (9, 17, 22), (10, 12, 21), (10, 13, 15),
        (10, 13, 20), (10, 14, 19), (10, 14, 23), (10, 15, 18), (10, 15, 22),
        (10, 17, 21), (10, 18, 20), (10, 19, 23), (10, 20, 22), (11, 12, 15),
        (11, 12, 20), (11, 13, 14), (11, 13, 19), (11, 13, 23), (11, 14, 18),
        (11, 14, 22), (11, 15, 17), (11, 16, 21), (11, 17, 20), (11, 18, 19),
        (11, 18, 23), (11, 19, 22), (11, 22, 23),
    )

    calculated_basis_states = calculate_m_basis_states(interaction=interaction, M_target=1)

    for expected, calculated in zip(expected_basis_states, calculated_basis_states):
        assert expected == calculated

if __name__ == "__main__":
    test_calculate_O18_m_basis_states()
    test_calculate_O19_m_basis_states()
    test_calculate_O20_m_basis_states()
    test_calculate_F19_m_basis_states()