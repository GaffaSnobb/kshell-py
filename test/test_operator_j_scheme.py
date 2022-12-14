import sys, os
try:
    """
    Added this try-except to make VSCode understand what is being
    imported.
    """
    from ..kshell_py.operator_j_scheme import operator_j_scheme
    from ..kshell_py.loaders import read_interaction_file
    from ..kshell_py.data_structures import Interaction, OperatorJ
    from ..kshell_py.parameters import flags, debug
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kshell_py')) # Hacky way of using relative imports without installing as a package.
    from operator_j_scheme import operator_j_scheme
    from loaders import read_interaction_file
    from data_structures import Interaction, OperatorJ
    from parameters import flags, debug

flags.debug: bool = False
INTERACTION_FILE_PATH_USDA: str = "../snt/usda.snt"
interaction_usda: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_USDA)
operator_j: OperatorJ = operator_j_scheme(interaction=interaction_usda, nucleus_mass=20)  # Populates the debug dataclass.

def test_operator_j_reverse_indices_usda():
    """
    Test the reverse indices for the USDA model space. Check the
    ordering and values, as generated by the function operator_j_scheme.
    """
    ij_01_expected = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 3, 3, 3, 2, 4,
        2, 4, 2, 5, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 2, 4, 4, 3, 2, 4, 3,
        4, 4, 4, 3, 4, 2, 5, 5, 4, 3, 1, 5, 4, 5, 5, 5, 4, 2, 5, 6, 5, 6, 6, 5,
        6, 7, 7, 6, 8, 6, 3, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1,
        2, 1, 2, 3, 3, 3, 2, 4, 2, 4, 2, 5, 2, 3
    ]

    success = len(ij_01_expected) == len(debug.ij_01)
    msg = "The number of expected ij_01 values does not match with the"
    msg += f" calculations! Expected: {len(ij_01_expected)}, got: {len(debug.ij_01)}."
    assert success, msg

    for i in range(len(debug.ij_01)):
        success = (ij_01_expected[i] - 1) == debug.ij_01[i]
        msg = f"Incorrect value in ij_01 index {i}."
        msg += f" Expected: {ij_01_expected[i]}, got: {debug.ij_01[i]}."
        assert success, msg

    ij_23_expected = [
        1, 1, 2, 3, 2, 4, 5, 3, 1, 2, 1, 1, 2, 3, 4, 2, 5, 2, 2, 3, 4, 5, 2, 4,
        2, 5, 3, 5, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 3, 2, 5, 5, 4, 6, 5,
        6, 7, 8, 6, 3, 7, 2, 2, 2, 1, 3, 3, 4, 4, 3, 2, 5, 5, 4, 3, 6, 5, 6, 7,
        8, 6, 7, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 7, 4, 4, 3, 2, 5, 5, 4, 3, 6, 5,
        6, 7, 8, 6, 7, 2, 5, 5, 4, 3, 1, 6, 5, 6, 7, 8, 6, 3, 7, 6, 5, 7, 8, 6,
        6, 7, 8, 7, 8, 6, 3, 7, 1, 1, 2, 3, 2, 4, 5, 3, 1, 2, 1, 1, 2, 3, 4, 2,
        5, 2, 2, 3, 4, 5, 2, 4, 2, 5, 3, 5, 2, 3
    ]

    success = len(ij_23_expected) == len(debug.ij_23)
    msg = "The number of expected ij_23 values does not match with the"
    msg += f" calculations! Expected: {len(ij_23_expected)}, got: {len(debug.ij_23)}."
    assert success, msg

    for i in range(len(debug.ij_23)):
        success = (ij_23_expected[i] - 1) == debug.ij_23[i]
        msg = f"Incorrect value in ij_23 index {i}."
        msg += f" Expected: {ij_23_expected[i]}, got: {debug.ij_23[i]}."
        assert success, msg

def test_operator_j_one_body():
    m_element_expected = [
        3.9596000000000000, -9.6598077496397394, -4.3291905571365188,
        3.9596000000000000, -9.6598077496397394, -4.3291905571365188
    ]

    success = len(m_element_expected) == len(operator_j.one_body_reduced_matrix_element)
    msg = "The number of expected one-body reduced matrix elements does not match with the"
    msg += f" calculations! Expected: {len(m_element_expected)}, got: {len(operator_j.one_body_reduced_matrix_element)}."
    assert success, msg

    for i in range(len(operator_j.one_body_reduced_matrix_element)):
        success = m_element_expected[i] == operator_j.one_body_reduced_matrix_element[i]
        msg = f"Incorrect value in one-body reduced matrix element {i}."
        msg += f" Expected: {m_element_expected[i]}, got: {operator_j.one_body_reduced_matrix_element[i]}."
        assert success, msg

def test_operator_j_two_body():
    """
    Check that the treated two-body reduced matrix elements are correct.
    """
    m_element_expected = [
        -1.4581736726018812, -0.15211512730797033, 0.50459591275153481,
        0.13254362685178564, -3.4582453751613920, -1.0982324637170979,
        -0.53695671053552341, -0.95280265092138883, 0.24319042646051309,
        0.21780560903714480, 0.46283691920393272, -1.2119796990416567,
        7.1310021464118578E-002, -0.27235389991255077, 0.29957960104219378,
        1.2745697450549998, -9.9020165674360311E-002, -0.43667699285160660,
        0.30083915305175024, -0.24541886463126683, 0.86240557208168411,
        1.6540824543959680, -2.4024501253047341, -0.95910041096917098,
        -0.20695408403173546, -0.75049922046340012, -1.1211950657374732,
        -0.87577620110620635, 0.62686934629462943, -1.7886607421862679,
        -1.4581736726018812, -1.4462563728191549, -0.15211512730797033,
        -2.8872807603678448, -9.3174391020776608E-003, 0.35680319007475925,
        -1.3006736891649320, -0.68606512248603502, 9.3722499611027521E-002,
        9.3174391020776608E-003, -0.35680319007475925, 1.3006736891649320,
        -3.4582453751613920, 1.9046364156815803, -1.0982324637170979,
        0.95067110136675470, -0.37968573303163461, 0.31508023909235372,
        0.68606512248603502, -9.3722499611027521E-002, -0.37968573303163461,
        0.31508023909235372, -0.95280265092138883, 9.1947296697620293E-002,
        -3.0324199073151945, -2.0929878854183284, -0.39656510577803988,
        -2.7691250930098383, -0.71576465158447833, -0.75640942604670347,
        3.2756103337757079, -2.3107934944554729, 0.85940202498197260,
        -1.5571453939681819, -2.0909434387295858, 0.21183477055787378,
        -1.5364871714158750, 0.90125690919294832, -0.54625801768301707,
        -0.65249638525829701, 0.78707467304859691, -0.48405552613415276,
        0.44723785200865679, -0.21581939240669040, -1.3031400760880141,
        -1.6929347894599782, -0.97193815260503469, 0.78707467304859691,
        -0.48405552613415276, 0.27178154010168504, 0.60981283209137249,
        -0.42994323403128559, 1.9937739425117285, -0.72651928797376797,
        2.0840256884272534, -0.45593802342629725, -3.0324199073151945,
        -2.0929878854183284, -0.39656510577803988, -2.7691250930098383,
        2.0909434387295858, -0.21183477055787378, 1.5364871714158750,
        -0.90125690919294832, -0.44723785200865679, 0.21581939240669040,
        -0.71576465158447833, -0.75640942604670347, 0.54625801768301707,
        0.65249638525829701, 1.3031400760880141, -2.4024501253047341,
        -1.3832787723413329, -0.95910041096917098, -1.3581846207663237,
        -0.20695408403173546, -4.2447871608213301, -0.53068308544201659,
        -1.2300392623321548, -0.27178154010168504, -0.60981283209137249,
        -0.53068308544201659, -1.2300392623321548, -1.1211950657374732,
        -0.86230868346556444, -0.67400565803687629, -1.5922190730035228,
        -2.0840256884272534, -0.20177054306933009, -2.2190884192981524,
        -1.6929347894599782, -0.97193815260503469, 0.42994323403128559,
        0.45593802342629725, -0.67400565803687629, -1.5922190730035228,
        -1.7886607421862679, -3.7489112235205710, -1.4581736726018812,
        -0.15211512730797033, 0.50459591275153481, 0.13254362685178564,
        -3.4582453751613920, -1.0982324637170979, -0.53695671053552341,
        -0.95280265092138883, 0.24319042646051309, 0.21780560903714480,
        0.46283691920393272, -1.2119796990416567, 7.1310021464118578E-002,
        -0.27235389991255077, 0.29957960104219378, 1.2745697450549998,
        -9.9020165674360311E-002, -0.43667699285160660, 0.30083915305175024,
        -0.24541886463126683, 0.86240557208168411, 1.6540824543959680,
        -2.4024501253047341, -0.95910041096917098, -0.20695408403173546,
        -0.75049922046340012, -1.1211950657374732, -0.87577620110620635,
        0.62686934629462943, -1.7886607421862679
    ]

    for i in range(interaction_usda.two_body.n_elements):
        calculated = operator_j.two_body_reduced_matrix_element[
            interaction_usda.two_body.jj[i], interaction_usda.two_body.parity[i], debug.proton_neutron_idx[i]
        ][debug.ij_01[i], debug.ij_23[i]]
        
        success = calculated == m_element_expected[i]
        msg = f"Two-body matrix element number {i} is incorrect."
        msg += f"Expected: {m_element_expected[i]}, got: {calculated}."
        assert success, msg

if __name__ == "__main__":
    test_operator_j_reverse_indices_usda()
    test_operator_j_one_body()
    test_operator_j_two_body()