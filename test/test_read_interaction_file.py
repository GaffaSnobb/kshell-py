import sys, os
import numpy as np
try:
    """
    Added this try-except to make VSCode understand what is being
    imported.
    """
    from ..kshell_py.kshell_py import read_interaction_file
    from ..kshell_py.parameters import flags
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kshell_py')) # Hacky way of using relative imports without installing as a package.
    from kshell_py import read_interaction_file
    from parameters import flags

flags.debug = False
INTERACTION_FILE_PATH = "../snt/usda.snt"
interaction = read_interaction_file(path=INTERACTION_FILE_PATH)

def test_model_space():
    n_expected = [
        0, 0, 1, 0, 0, 1
    ]
    l_expected = [
        2, 2, 0, 2, 2, 0
    ]
    j_expected = [
        3, 5, 1, 3, 5, 1
    ]
    isospin_expected = [
        -1, -1, -1, 1, 1, 1
    ]
    parity_expected = [
        1, 1, 1, 1, 1, 1
    ]
    n_proton_orbitals_expected = 3
    n_neutron_orbitals_expected = 3
    n_orbitals_expected = 6
    n_core_protons_expected = 8
    n_core_neutrons_expected = 8
    n_core_nucleons_expected = 16

    success = n_proton_orbitals_expected == interaction.model_space.n_proton_orbitals
    msg = f"The number of proton orbitals was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_proton_orbitals_expected}, got: {interaction.model_space.n_proton_orbitals}."
    assert success, msg

    success = n_neutron_orbitals_expected == interaction.model_space.n_neutron_orbitals
    msg = f"The number of neutron orbitals was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_neutron_orbitals_expected}, got: {interaction.model_space.n_neutron_orbitals}."
    assert success, msg

    success = n_orbitals_expected == interaction.model_space.n_orbitals
    msg = f"The number of orbitals was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_orbitals_expected}, got: {interaction.model_space.n_orbitals}."
    assert success, msg

    success = n_core_protons_expected == interaction.n_core_protons
    msg = f"The number of core protons was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_core_protons_expected}, got: {interaction.n_core_protons}."
    assert success, msg

    success = n_core_neutrons_expected == interaction.n_core_neutrons
    msg = f"The number of core neutrons was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_core_neutrons_expected}, got: {interaction.n_core_neutrons}."
    assert success, msg

    success = n_core_nucleons_expected == interaction.n_core_nucleons
    msg = f"The number of core nucleons was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_core_nucleons_expected}, got: {interaction.n_core_nucleons}."
    assert success, msg

    for i in range(interaction.model_space.n_orbitals):
        success = n_expected[i] == interaction.model_space.n[i]
        msg = f"Error in n orbital number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {n_expected[i]}, got: {interaction.model_space.n[i]}."
        assert success, msg

    for i in range(interaction.model_space.n_orbitals):
        success = l_expected[i] == interaction.model_space.l[i]
        msg = f"Error in l orbital number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {l_expected[i]}, got: {interaction.model_space.l[i]}."
        assert success, msg

    for i in range(interaction.model_space.n_orbitals):
        success = j_expected[i] == interaction.model_space.j[i]
        msg = f"Error in j orbital number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {j_expected[i]}, got: {interaction.model_space.j[i]}."
        assert success, msg

    for i in range(interaction.model_space.n_orbitals):
        success = isospin_expected[i] == interaction.model_space.isospin[i]
        msg = f"Error in isospin orbital number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {isospin_expected[i]}, got: {interaction.model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction.model_space.n_orbitals):
        success = parity_expected[i] == interaction.model_space.parity[i]
        msg = f"Error in parity orbital number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {parity_expected[i]}, got: {interaction.model_space.parity[i]}."
        assert success, msg

def test_one_body():
    reduced_matrix_element_expected = [
        1.97980000, -3.94360000, -3.06120000, 1.97980000, -3.94360000,
        -3.06120000,
    ]
    method_expected = 0
    n_elements_expected = 6

    success = method_expected == interaction.one_body.method
    msg = f"Method incorrectly read from {INTERACTION_FILE_PATH}."
    msg += f" Expected: {method_expected}, got: {interaction.one_body.method}"
    assert success, msg

    success = n_elements_expected == interaction.one_body.n_elements
    msg = f"n_elements incorrectly read from {INTERACTION_FILE_PATH}."
    msg += f" Expected: {n_elements_expected}, got: {interaction.one_body.n_elements}"
    assert success, msg

    for i in range(interaction.one_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction.one_body.reduced_matrix_element[i]
        msg = f"Error in one-body reduced matrix element number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction.one_body.reduced_matrix_element[i]}."
        assert success, msg

def test_two_body():
    reduced_matrix_element_expected = [
        -1.50500000, -0.15700000, 0.52080000, 0.13680000, -3.56930000,
        -1.13350000, -0.55420000, -0.98340000, 0.25100000, 0.22480000,
        0.47770000, -1.25090000, 0.07360000, -0.28110000, 0.30920000,
        1.31550000, -0.10220000, -0.45070000, 0.31050000, -0.25330000,
        0.89010000, 1.70720000, -2.47960000, -0.98990000, -0.21360000,
        -0.77460000, -1.15720000, -0.90390000, 0.64700000, -1.84610000,
        -1.50500000, -1.49270000, -0.15700000, -2.98000000, -0.00961665,
        0.36826121, -1.34244222, -0.70809673, 0.09673221, 0.00961665,
        -0.36826121, 1.34244222, -3.56930000, 1.96580000, -1.13350000,
        0.98120000, -0.39187858, 0.32519841, 0.70809673, -0.09673221,
        -0.39187858, 0.32519841, -0.98340000, 0.09490000, -3.12980000,
        -2.16020000, -0.40930000, -2.85805000, -0.73875000, -0.78070000,
        3.38080000, -2.38500000, 0.88700000, -1.60715000, -2.15808990,
        0.21863742, -1.58582838, 0.93019897, -0.56380000, -0.67345000,
        0.81235000, -0.49960000, 0.46160000, -0.22275000, -1.34498781,
        -1.74730000, -1.00315000, 0.81235000, -0.49960000, 0.28050926,
        0.62939575, -0.44375000, 2.05780000, -0.74985000, 2.15095000,
        -0.47057956, -3.12980000, -2.16020000, -0.40930000, -2.85805000,
        2.15808990, -0.21863742, 1.58582838, -0.93019897, -0.46160000,
        0.22275000, -0.73875000, -0.78070000, 0.56380000, 0.67345000,
        1.34498781, -2.47960000, -1.42770000, -0.98990000, -1.40180000,
        -0.21360000, -4.38110000, -0.54772491, -1.26953951, -0.28050926,
        -0.62939575, -0.54772491, -1.26953951, -1.15720000, -0.89000000,
        -0.69565000, -1.64335000, -2.15095000, -0.20825000, -2.29035000,
        -1.74730000, -1.00315000, 0.44375000, 0.47057956, -0.69565000,
        -1.64335000, -1.84610000, -3.86930000, -1.50500000, -0.15700000,
        0.52080000, 0.13680000, -3.56930000, -1.13350000, -0.55420000,
        -0.98340000, 0.25100000, 0.22480000, 0.47770000, -1.25090000,
        0.07360000, -0.28110000, 0.30920000, 1.31550000, -0.10220000,
        -0.45070000, 0.31050000, -0.25330000, 0.89010000, 1.70720000,
        -2.47960000, -0.98990000, -0.21360000, -0.77460000, -1.15720000,
        -0.90390000, 0.64700000, -1.84610000
    ]
    orbital_0_expected = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6
    ]) - 1  # Indices start at 1 in the .snt files but from 0 when read.

    orbital_1_expected = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2,
        2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
        4, 4, 4, 4, 5, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6
    ]) - 1

    orbital_2_expected = np.array([
        1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2,
        2, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 4, 4, 4, 4, 4, 4, 5, 5,
        5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 6, 5, 5, 6
    ]) - 1

    orbital_3_expected = np.array([
        1, 1, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 2, 3, 2, 2,
        2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 4, 4, 4, 5, 5, 5, 5, 6, 6,
        4, 4, 5, 5, 6, 6, 5, 5, 5, 5, 6, 6, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 4, 4,
        5, 5, 6, 6, 6, 4, 4, 5, 5, 6, 4, 4, 5, 6, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
        4, 4, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 6, 6, 4, 5, 5,
        4, 4, 5, 6, 5, 5, 6, 6, 4, 4, 5, 6, 5, 5, 6, 6, 5, 5, 5, 5, 6, 6, 5, 5,
        6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6
    ]) - 1

    jj_expected = np.array([
        0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 3, 4, 1, 2, 2, 4, 2, 3, 1, 2, 2, 2, 0, 2,
        4, 2, 0, 2, 3, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 0, 1, 2, 3, 2, 3,
        1, 2, 2, 3, 0, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 1, 2,
        2, 3, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3,
        1, 2, 2, 3, 1, 0, 1, 2, 3, 4, 5, 2, 3, 1, 2, 2, 3, 0, 1, 2, 3, 2, 2, 3,
        1, 2, 2, 1, 2, 3, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 3, 4, 1, 2, 2, 4,
        2, 3, 1, 2, 2, 2, 0, 2, 4, 2, 0, 2, 3, 0
    ])

    n_elements_expected = 158
    method_expected = 1
    im0_expected = 18
    pwr_expected = -0.300000

    success = n_elements_expected == interaction.two_body.n_elements
    msg = f"The number of two-body matrix elements was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {n_elements_expected}, got: {interaction.two_body.n_elements}."
    assert success, msg

    success = method_expected == interaction.two_body.method
    msg = f"The two-body method was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {method_expected}, got: {interaction.two_body.method}."
    assert success, msg

    success = im0_expected == interaction.two_body.im0
    msg = f"The two-body im0 was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {im0_expected}, got: {interaction.two_body.im0}."
    assert success, msg

    success = pwr_expected == interaction.two_body.pwr
    msg = f"The two-body pwr was not correctly read from '{INTERACTION_FILE_PATH}'."
    msg += f" Expected: {pwr_expected}, got: {interaction.two_body.pwr}."
    assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction.two_body.reduced_matrix_element[i]
        msg = f"Error in two-body reduced matrix element number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction.two_body.reduced_matrix_element[i]}."
        assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = orbital_0_expected[i] == interaction.two_body.orbital_0[i]
        msg = f"Error in orbital 0 number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {orbital_0_expected[i]}, got: {interaction.two_body.orbital_0[i]}."
        assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = orbital_1_expected[i] == interaction.two_body.orbital_1[i]
        msg = f"Error in orbital 1 number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {orbital_1_expected[i]}, got: {interaction.two_body.orbital_1[i]}."
        assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = orbital_2_expected[i] == interaction.two_body.orbital_2[i]
        msg = f"Error in orbital 2 number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {orbital_2_expected[i]}, got: {interaction.two_body.orbital_2[i]}."
        assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = orbital_3_expected[i] == interaction.two_body.orbital_3[i]
        msg = f"Error in orbital 3 number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {orbital_3_expected[i]}, got: {interaction.two_body.orbital_3[i]}."
        assert success, msg

    for i in range(interaction.two_body.n_elements):
        success = jj_expected[i] == interaction.two_body.jj[i]
        msg = f"Error in jj number {i} in {INTERACTION_FILE_PATH}."
        msg += f" Expected: {jj_expected[i]}, got: {interaction.two_body.jj[i]}."
        assert success, msg

if __name__ == "__main__":
    test_model_space()
    test_one_body()
    test_two_body()