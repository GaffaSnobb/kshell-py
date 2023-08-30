# from kshell_utilities.data_structures import Partition, Interaction
import kshell_utilities.data_structures as ksutild
from kshell_utilities.loaders import load_interaction, load_partition
from data_structures import Partition
from partition import (
    calculate_all_possible_orbital_occupations, calculate_all_possible_pairs
)

def test_calculate_all_possible_orbital_occupations():
    """
    Test that the function calculate_all_possible_orbital_occupations
    creates O19 neutron partition identical to the O19 partition file
    from KSHELL.
    """
    interaction: ksutild.Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    partition: Partition = calculate_all_possible_orbital_occupations(interaction=interaction)
    
    interaction: ksutild.Interaction = load_interaction(filename_interaction="../snt/w.snt")
    partition_neutron: ksutild.Partition | str
    _, partition_neutron, _ = load_partition(interaction=interaction, filename_partition="../ptn/O19_w_p.ptn")

    if isinstance(partition_neutron, str): raise RuntimeError   # Mostly to make the type checker in VSCode happy.

    for expected, calculated in zip(partition_neutron.configurations, partition.configurations):
        expected = expected.configuration
        calculated = calculated.configuration
        assert all(expected_occ == calculated_occ for expected_occ, calculated_occ in zip(expected, calculated))

def test_calculate_all_possible_pairs():
    """
    Test that calculate_all_possible_pairs manages to calculate all
    possible pairs from a bunch of configurations.
    """
    configurations: list[tuple[int, ...]] = [
        (0, 1, 2), (0, 2, 1), (0, 3, 0), (1, 0, 2), (1, 1, 1), (1, 2, 0),
        (2, 0, 1), (2, 1, 0), (3, 0, 0)
    ]
    expected: list[list[tuple[int, ...]]] = [
        [(1, 2), (1, 2), (2, 2)], [(1, 1), (1, 2), (1, 2)],
        [(1, 1), (1, 1), (1, 1)], [(0, 2), (0, 2), (2, 2)],
        [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 1), (1, 1)],
        [(0, 0), (0, 2), (0, 2)], [(0, 0), (0, 1), (0, 1)],
        [(0, 0), (0, 0), (0, 0)],

    ]
    expected = [sorted(i) for i in expected]

    for i in range(len(configurations)):

        calculated = calculate_all_possible_pairs(
            configuration = configurations[i],
            M_target = 1,
        )
        assert calculated == expected[i], f"{calculated} != {expected[i]}"

if __name__ == "__main__":
    test_calculate_all_possible_orbital_occupations()
    test_calculate_all_possible_pairs()