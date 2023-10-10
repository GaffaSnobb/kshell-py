from kshell_utilities.loaders import load_interaction
from kshell_utilities.data_structures import Interaction
from tools import generate_indices
from data_structures import Indices

def test_generate_indices():
    """
    Test that the indices are generated correlty. Uses sd model space.
    """
    orbital_idx_to_m_idx_map = [
        (0, 1, 2, 3),   # d3/2 has 4 m substates
        (0, 1, 2, 3, 4, 5),
        (0, 1),
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

    orbital_idx_to_j_map = [3, 5, 1] # 0 is 3/2, 1 is 5/2, 0 is s1/2

    m_composite_idx_to_m_map = [
        -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
    ]
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    indices: Indices = generate_indices(interaction=interaction)

    assert len(orbital_idx_to_m_idx_map) == len(indices.orbital_idx_to_m_idx_map)
    assert len(orbital_m_pair_to_composite_m_idx_map) == len(indices.orbital_m_pair_to_composite_m_idx_map)
    assert len(orbital_idx_to_j_map) == len(indices.orbital_idx_to_j_map)
    assert len(m_composite_idx_to_m_map) == len(indices.m_composite_idx_to_m_map)

    for i in range(len(orbital_idx_to_m_idx_map)):
        assert orbital_idx_to_m_idx_map[i] == indices.orbital_idx_to_m_idx_map[i]

    for i in range(len(orbital_idx_to_j_map)):
        assert orbital_idx_to_j_map[i] == indices.orbital_idx_to_j_map[i]

    for key in orbital_m_pair_to_composite_m_idx_map:
        assert orbital_m_pair_to_composite_m_idx_map[key] == indices.orbital_m_pair_to_composite_m_idx_map[key]

    for i in range(len(m_composite_idx_to_m_map)):
        assert m_composite_idx_to_m_map[i] == indices.m_composite_idx_to_m_map[i]

if __name__ == "__main__":
    test_generate_indices()