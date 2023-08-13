from kshell_utilities.data_structures import Partition, Interaction
from kshell_utilities.loaders import load_interaction, load_partition
from partition import calculate_all_possible_orbital_occupations

def test_calculate_all_possible_orbital_occupations():
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    orbital_occupations = calculate_all_possible_orbital_occupations(interaction=interaction)
    
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    partition_neutron: Partition
    _, partition_neutron, _ = load_partition(interaction=interaction, filename_partition="../ptn/O19_w_p.ptn")

    for expected, calculated in zip(partition_neutron.configurations, orbital_occupations):
        expected = expected.configuration
        assert all(expected_occ == calculated_occ for expected_occ, calculated_occ in zip(expected, calculated))


if __name__ == "__main__":
    test_calculate_all_possible_orbital_occupations()