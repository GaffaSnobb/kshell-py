import sys, os
import numpy as np
try:
    """
    Added this try-except to make VSCode understand what is being
    imported.
    """
    from ..kshell_py.kshell_py import read_interaction_file, initialise_operator_j_couplings, operator_j_scheme
    from ..kshell_py.data_structures import Interaction, CouplingIndices, OperatorJ
    from ..kshell_py.parameters import flags, debug
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kshell_py')) # Hacky way of using relative imports without installing as a package.
    from kshell_py import read_interaction_file, initialise_operator_j_couplings, operator_j_scheme
    from data_structures import Interaction, CouplingIndices, OperatorJ
    from parameters import flags, debug

flags.debug: bool = False
INTERACTION_FILE_PATH_USDA: str = "../snt/usda.snt"
INTERACTION_FILE_PATH_SDPFMU: str = "../snt/sdpf-mu.snt"
interaction_usda: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_USDA)
interaction_sdpfmu: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_SDPFMU)
j_couple_usda: CouplingIndices = initialise_operator_j_couplings(interaction=interaction_usda)
j_couple_sdpfmu: CouplingIndices = initialise_operator_j_couplings(interaction=interaction_sdpfmu)
operator_j: OperatorJ = operator_j_scheme(path=INTERACTION_FILE_PATH_USDA, nucleus_mass=20)  # Populates the debug dataclass.

def test_model_space_usda():
    n_expected = [
        0, 0, 1, 0, 0, 1
    ]
    l_expected = [
        2, 2, 0, 2, 2, 0
    ]
    j_expected = [
        3, 5, 1, 3, 5, 1
    ]
    jz_expected = [
        [-3, -1, 1, 3], [-5, -3, -1, 1, 3, 5], [-1, 1],
        [-3, -1, 1, 3], [-5, -3, -1, 1, 3, 5], [-1, 1]
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

    success = n_proton_orbitals_expected == interaction_usda.model_space.n_proton_orbitals
    msg = f"The number of proton orbitals was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_proton_orbitals_expected}, got: {interaction_usda.model_space.n_proton_orbitals}."
    assert success, msg

    success = n_neutron_orbitals_expected == interaction_usda.model_space.n_neutron_orbitals
    msg = f"The number of neutron orbitals was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_neutron_orbitals_expected}, got: {interaction_usda.model_space.n_neutron_orbitals}."
    assert success, msg

    success = n_orbitals_expected == interaction_usda.model_space.n_orbitals
    msg = f"The number of orbitals was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_orbitals_expected}, got: {interaction_usda.model_space.n_orbitals}."
    assert success, msg

    success = n_core_protons_expected == interaction_usda.n_core_protons
    msg = f"The number of core protons was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_core_protons_expected}, got: {interaction_usda.n_core_protons}."
    assert success, msg

    success = n_core_neutrons_expected == interaction_usda.n_core_neutrons
    msg = f"The number of core neutrons was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_core_neutrons_expected}, got: {interaction_usda.n_core_neutrons}."
    assert success, msg

    success = n_core_nucleons_expected == interaction_usda.n_core_nucleons
    msg = f"The number of core nucleons was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_core_nucleons_expected}, got: {interaction_usda.n_core_nucleons}."
    assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = n_expected[i] == interaction_usda.model_space.n[i]
        msg = f"Error in n orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {n_expected[i]}, got: {interaction_usda.model_space.n[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = l_expected[i] == interaction_usda.model_space.l[i]
        msg = f"Error in l orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {l_expected[i]}, got: {interaction_usda.model_space.l[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = j_expected[i] == interaction_usda.model_space.j[i]
        msg = f"Error in j orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {j_expected[i]}, got: {interaction_usda.model_space.j[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = isospin_expected[i] == interaction_usda.model_space.isospin[i]
        msg = f"Error in isospin orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {isospin_expected[i]}, got: {interaction_usda.model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = parity_expected[i] == interaction_usda.model_space.parity[i]
        msg = f"Error in parity orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {parity_expected[i]}, got: {interaction_usda.model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_orbitals):
        success = all(jz_expected[i] == interaction_usda.model_space.jz[i])
        msg = f"Error in jz orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {jz_expected[i]}, got: {interaction_usda.model_space.jz[i]}."
        assert success, msg

def test_proton_model_space_usda():
    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = interaction_usda.proton_model_space.n[i] == interaction_usda.model_space.n[i]
        msg = f"Error in n proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.n[i]}, got: {interaction_usda.proton_model_space.n[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = interaction_usda.proton_model_space.l[i] == interaction_usda.model_space.l[i]
        msg = f"Error in l proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.l[i]}, got: {interaction_usda.proton_model_space.l[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = interaction_usda.proton_model_space.j[i] == interaction_usda.model_space.j[i]
        msg = f"Error in j proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.j[i]}, got: {interaction_usda.proton_model_space.j[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = interaction_usda.proton_model_space.isospin[i] == interaction_usda.model_space.isospin[i]
        msg = f"Error in isospin proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.isospin[i]}, got: {interaction_usda.proton_model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = interaction_usda.proton_model_space.parity[i] == interaction_usda.model_space.parity[i]
        msg = f"Error in parity proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.parity[i]}, got: {interaction_usda.proton_model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_proton_orbitals):
        success = all(interaction_usda.proton_model_space.jz[i] == interaction_usda.model_space.jz[i])
        msg = f"Error in jz proton orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.jz[i]}, got: {interaction_usda.proton_model_space.jz[i]}."
        assert success, msg

def test_neutron_model_space_usda():
    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = interaction_usda.neutron_model_space.n[i] == interaction_usda.model_space.n[i + interaction_usda.model_space.n_proton_orbitals]
        msg = f"Error in n neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.n[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.n[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = interaction_usda.neutron_model_space.l[i] == interaction_usda.model_space.l[i + interaction_usda.model_space.n_proton_orbitals]
        msg = f"Error in l neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.l[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.l[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = interaction_usda.neutron_model_space.j[i] == interaction_usda.model_space.j[i + interaction_usda.model_space.n_proton_orbitals]
        msg = f"Error in j neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.j[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.j[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = interaction_usda.neutron_model_space.isospin[i] == interaction_usda.model_space.isospin[i + interaction_usda.model_space.n_proton_orbitals]
        msg = f"Error in isospin neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.isospin[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = interaction_usda.neutron_model_space.parity[i] == interaction_usda.model_space.parity[i + interaction_usda.model_space.n_proton_orbitals]
        msg = f"Error in parity neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.parity[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_usda.model_space.n_neutron_orbitals):
        success = all(interaction_usda.neutron_model_space.jz[i] == interaction_usda.model_space.jz[i + interaction_usda.model_space.n_proton_orbitals])
        msg = f"Error in jz neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_usda.model_space.jz[i + interaction_usda.model_space.n_proton_orbitals]}, got: {interaction_usda.neutron_model_space.jz[i]}."
        assert success, msg

def test_one_body_usda():
    reduced_matrix_element_expected = [
        1.97980000, -3.94360000, -3.06120000, 1.97980000, -3.94360000,
        -3.06120000,
    ]
    method_expected = 0
    n_elements_expected = 6

    success = method_expected == interaction_usda.one_body.method
    msg = f"Method incorrectly read from {INTERACTION_FILE_PATH_USDA}."
    msg += f" Expected: {method_expected}, got: {interaction_usda.one_body.method}"
    assert success, msg

    success = n_elements_expected == interaction_usda.one_body.n_elements
    msg = f"n_elements incorrectly read from {INTERACTION_FILE_PATH_USDA}."
    msg += f" Expected: {n_elements_expected}, got: {interaction_usda.one_body.n_elements}"
    assert success, msg

    for i in range(interaction_usda.one_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction_usda.one_body.reduced_matrix_element[i]
        msg = f"Error in one-body reduced matrix element number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction_usda.one_body.reduced_matrix_element[i]}."
        assert success, msg

def test_two_body_usda():
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

    success = n_elements_expected == interaction_usda.two_body.n_elements
    msg = f"The number of two-body matrix elements was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {n_elements_expected}, got: {interaction_usda.two_body.n_elements}."
    assert success, msg

    success = method_expected == interaction_usda.two_body.method
    msg = f"The two-body method was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {method_expected}, got: {interaction_usda.two_body.method}."
    assert success, msg

    success = im0_expected == interaction_usda.two_body.im0
    msg = f"The two-body im0 was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {im0_expected}, got: {interaction_usda.two_body.im0}."
    assert success, msg

    success = pwr_expected == interaction_usda.two_body.pwr
    msg = f"The two-body pwr was not correctly read from '{INTERACTION_FILE_PATH_USDA}'."
    msg += f" Expected: {pwr_expected}, got: {interaction_usda.two_body.pwr}."
    assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction_usda.two_body.reduced_matrix_element[i]
        msg = f"Error in two-body reduced matrix element number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction_usda.two_body.reduced_matrix_element[i]}."
        assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = orbital_0_expected[i] == interaction_usda.two_body.orbital_0[i]
        msg = f"Error in orbital 0 number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {orbital_0_expected[i]}, got: {interaction_usda.two_body.orbital_0[i]}."
        assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = orbital_1_expected[i] == interaction_usda.two_body.orbital_1[i]
        msg = f"Error in orbital 1 number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {orbital_1_expected[i]}, got: {interaction_usda.two_body.orbital_1[i]}."
        assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = orbital_2_expected[i] == interaction_usda.two_body.orbital_2[i]
        msg = f"Error in orbital 2 number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {orbital_2_expected[i]}, got: {interaction_usda.two_body.orbital_2[i]}."
        assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = orbital_3_expected[i] == interaction_usda.two_body.orbital_3[i]
        msg = f"Error in orbital 3 number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {orbital_3_expected[i]}, got: {interaction_usda.two_body.orbital_3[i]}."
        assert success, msg

    for i in range(interaction_usda.two_body.n_elements):
        success = jj_expected[i] == interaction_usda.two_body.jj[i]
        msg = f"Error in jj number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {jj_expected[i]}, got: {interaction_usda.two_body.jj[i]}."
        assert success, msg

def test_operator_j_one_body_reduced_matrix_element_usda():
    """
    num1:            6 dep_mass1:    1.0000000000000000      method1:           0
    AIDS!!!
    k1:            1 k2:            1 v:    1.9798000000000000     
    j_orbitals(k1):            3
    factor:    2.0000000000000000     
    matrix_element(k1, k2):    3.9596000000000000     
    k1:            2 k2:            2 v:   -3.9436000000000000     
    j_orbitals(k1):            5
    factor:    2.4494897427831779     
    matrix_element(k1, k2):   -9.6598077496397394     
    k1:            3 k2:            3 v:   -3.0611999999999999     
    j_orbitals(k1):            1
    factor:    1.4142135623730951     
    matrix_element(k1, k2):   -4.3291905571365188     
    k1:            4 k2:            4 v:    1.9798000000000000     
    j_orbitals(k1):            3
    factor:    2.0000000000000000     
    matrix_element(k1, k2):    3.9596000000000000     
    k1:            5 k2:            5 v:   -3.9436000000000000     
    j_orbitals(k1):            5
    factor:    2.4494897427831779     
    matrix_element(k1, k2):   -9.6598077496397394     
    k1:            6 k2:            6 v:   -3.0611999999999999     
    j_orbitals(k1):            1
    factor:    1.4142135623730951     
    matrix_element(k1, k2):   -4.3291905571365188     
    TBME mass dependence (mass/  18)^ -0.30000000
    """
    pass

def test_proton_j_couple_usda():
    """
    Test the construction of the proton j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 0  # Protons.

    n_couplings_expected = [
        3, 2, 5, 2, 2, 0
    ]
    idx_expected = [
        [1, 1, 2, 2, 3, 3],
        [1, 2, 1, 3],
        [1, 1, 1, 2, 1, 3, 2, 2, 2, 3],
        [1, 2, 2, 3],
        [1, 2, 2, 2],
        []
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 2, 0, 0, 0, 3],
        [0, 0, 0, 1, 0, 0, 2, 0, 0],
        [1, 0, 0, 2, 4, 0, 3, 5, 0],
        [0, 0, 0, 1, 0, 0, 0, 2, 0],
        [0, 0, 0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 0  # Protons.

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == 0
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {0}, got: {coupling.n_couplings}."
        assert success, msg

        success = coupling.idx.shape == (2, 0)
        msg = f"Incorrect idx for coupling {i}."
        msg += f" Expected: shape (2, 0), got: shape {coupling.idx.shape}."
        assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == 0
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {0}, got: {idx_reverse}."
            assert success, msg

def test_neutron_j_couple_usda():
    """
    Test the construction of the neutron j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 1  # Neutrons.

    n_couplings_expected = [
        3, 2, 5, 2, 2, 0
    ]
    idx_expected = [
        [1, 1, 2, 2, 3, 3],
        [1, 2, 1, 3],
        [1, 1, 1, 2, 1, 3, 2, 2, 2, 3],
        [1, 2, 2, 3],
        [1, 2, 2, 2],
        []
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 2, 0, 0, 0, 3],
        [0, 0, 0, 1, 0, 0, 2, 0, 0],
        [1, 0, 0, 2, 4, 0, 3, 5, 0],
        [0, 0, 0, 1, 0, 0, 0, 2, 0],
        [0, 0, 0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 1  # Neutrons.

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == 0
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {0}, got: {coupling.n_couplings}."
        assert success, msg

        success = coupling.idx.shape == (2, 0)
        msg = f"Incorrect idx for coupling {i}."
        msg += f" Expected: shape (2, 0), got: shape {coupling.idx.shape}."
        assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == 0
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {0}, got: {idx_reverse}."
            assert success, msg

def test_proton_neutron_j_couple_usda():
    """
    Test the construction of the proton-neutron j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 2  # Proton-neutron.

    n_couplings_expected = [
        3, 7, 8, 6, 3, 1
    ]
    idx_expected = [
        [1, 4, 2, 5, 3, 6],
        [1, 4, 1, 5, 1, 6, 2, 4, 2, 5, 3, 4, 3, 6],
        [1, 4, 1, 5, 1, 6, 2, 4, 2, 5, 2, 6, 3, 4, 3, 5],
        [1, 4, 1, 5, 2, 4, 2, 5, 2, 6, 3, 5],
        [1, 5, 2, 4, 2, 5],
        [2, 5],
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 2, 0, 0, 0, 3],
        [1, 4, 6, 2, 5, 0, 3, 0, 7],
        [1, 4, 7, 2, 5, 8, 3, 6, 0],
        [1, 3, 0, 2, 4, 6, 0, 5, 0],
        [0, 2, 0, 1, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            if j%2 != 0:
                """
                In the Fortran KSHELL code, the neutron orbitals are
                counted after the proton orbitals. This is why we must
                add the number of proton orbitals to the neutron
                orbitals which are every other orbital.

                In the Fortran code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: n_proton_orbitals, n_proton_orbitals + 1, ..., n_orbitals - 1

                In the Python code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: 0, 1, 2, ..., n_neutron_orbitals - 1
                """
                idx += interaction_usda.proton_model_space.n_orbitals

            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 2  # Neutrons.

    for i, coupling in enumerate(j_couple_usda[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == 0
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {0}, got: {coupling.n_couplings}."
        assert success, msg

        success = coupling.idx.shape == (2, 0)
        msg = f"Incorrect idx for coupling {i}."
        msg += f" Expected: shape (2, 0), got: shape {coupling.idx.shape}."
        assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == 0
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {0}, got: {idx_reverse}."
            assert success, msg

def test_model_space_sdpfmu():
    n_expected = [
        0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1
    ]
    l_expected = [
        2, 2, 0, 3, 3, 1, 1, 2, 2, 0, 3, 3, 1, 1
    ]
    j_expected = [
        5, 3, 1, 7, 5, 3, 1, 5, 3, 1, 7, 5, 3, 1
    ]
    isospin_expected = [
        -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1   
    ]
    parity_expected = [
        1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1
    ]
    jz_expected = [
        [-5, -3, -1, 1, 3, 5], [-3, -1, 1, 3], [-1, 1],
        [-7, -5, -3, -1, 1, 3, 5, 7], [-5, -3, -1, 1, 3, 5], [-3, -1, 1, 3],
        [-1, 1], [-5, -3, -1, 1, 3, 5], [-3, -1, 1, 3], [-1, 1],
        [-7, -5, -3, -1, 1, 3, 5, 7], [-5, -3, -1, 1, 3, 5], [-3, -1, 1, 3],
        [-1, 1]
    ]
    n_proton_orbitals_expected = 7
    n_neutron_orbitals_expected = 7
    n_orbitals_expected = 14
    n_core_protons_expected = 8
    n_core_neutrons_expected = 8
    n_core_nucleons_expected = 16

    success = n_proton_orbitals_expected == interaction_sdpfmu.model_space.n_proton_orbitals
    msg = f"The number of proton orbitals was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_proton_orbitals_expected}, got: {interaction_sdpfmu.model_space.n_proton_orbitals}."
    assert success, msg

    success = n_neutron_orbitals_expected == interaction_sdpfmu.model_space.n_neutron_orbitals
    msg = f"The number of neutron orbitals was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_neutron_orbitals_expected}, got: {interaction_sdpfmu.model_space.n_neutron_orbitals}."
    assert success, msg

    success = n_orbitals_expected == interaction_sdpfmu.model_space.n_orbitals
    msg = f"The number of orbitals was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_orbitals_expected}, got: {interaction_sdpfmu.model_space.n_orbitals}."
    assert success, msg

    success = n_core_protons_expected == interaction_sdpfmu.n_core_protons
    msg = f"The number of core protons was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_core_protons_expected}, got: {interaction_sdpfmu.n_core_protons}."
    assert success, msg

    success = n_core_neutrons_expected == interaction_sdpfmu.n_core_neutrons
    msg = f"The number of core neutrons was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_core_neutrons_expected}, got: {interaction_sdpfmu.n_core_neutrons}."
    assert success, msg

    success = n_core_nucleons_expected == interaction_sdpfmu.n_core_nucleons
    msg = f"The number of core nucleons was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_core_nucleons_expected}, got: {interaction_sdpfmu.n_core_nucleons}."
    assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = n_expected[i] == interaction_sdpfmu.model_space.n[i]
        msg = f"Error in n orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {n_expected[i]}, got: {interaction_sdpfmu.model_space.n[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = l_expected[i] == interaction_sdpfmu.model_space.l[i]
        msg = f"Error in l orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {l_expected[i]}, got: {interaction_sdpfmu.model_space.l[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = j_expected[i] == interaction_sdpfmu.model_space.j[i]
        msg = f"Error in j orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {j_expected[i]}, got: {interaction_sdpfmu.model_space.j[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = isospin_expected[i] == interaction_sdpfmu.model_space.isospin[i]
        msg = f"Error in isospin orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {isospin_expected[i]}, got: {interaction_sdpfmu.model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = parity_expected[i] == interaction_sdpfmu.model_space.parity[i]
        msg = f"Error in parity orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {parity_expected[i]}, got: {interaction_sdpfmu.model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_orbitals):
        success = all(jz_expected[i] == interaction_sdpfmu.model_space.jz[i])
        msg = f"Error in jz orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {jz_expected[i]}, got: {interaction_sdpfmu.model_space.jz[i]}."
        assert success, msg

def test_proton_model_space_sdpfmu():
    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = interaction_sdpfmu.proton_model_space.n[i] == interaction_sdpfmu.model_space.n[i]
        msg = f"Error in n proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.n[i]}, got: {interaction_sdpfmu.proton_model_space.n[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = interaction_sdpfmu.proton_model_space.l[i] == interaction_sdpfmu.model_space.l[i]
        msg = f"Error in l proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.l[i]}, got: {interaction_sdpfmu.proton_model_space.l[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = interaction_sdpfmu.proton_model_space.j[i] == interaction_sdpfmu.model_space.j[i]
        msg = f"Error in j proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.j[i]}, got: {interaction_sdpfmu.proton_model_space.j[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = interaction_sdpfmu.proton_model_space.isospin[i] == interaction_sdpfmu.model_space.isospin[i]
        msg = f"Error in isospin proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.isospin[i]}, got: {interaction_sdpfmu.proton_model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = interaction_sdpfmu.proton_model_space.parity[i] == interaction_sdpfmu.model_space.parity[i]
        msg = f"Error in parity proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.parity[i]}, got: {interaction_sdpfmu.proton_model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_proton_orbitals):
        success = all(interaction_sdpfmu.proton_model_space.jz[i] == interaction_sdpfmu.model_space.jz[i])
        msg = f"Error in jz proton orbital number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {interaction_sdpfmu.model_space.jz[i]}, got: {interaction_sdpfmu.proton_model_space.jz[i]}."
        assert success, msg

def test_neutron_model_space_sdpfmu():
    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = interaction_sdpfmu.neutron_model_space.n[i] == interaction_sdpfmu.model_space.n[i + interaction_sdpfmu.model_space.n_proton_orbitals]
        msg = f"Error in n neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.n[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.n[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = interaction_sdpfmu.neutron_model_space.l[i] == interaction_sdpfmu.model_space.l[i + interaction_sdpfmu.model_space.n_proton_orbitals]
        msg = f"Error in l neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.l[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.l[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = interaction_sdpfmu.neutron_model_space.j[i] == interaction_sdpfmu.model_space.j[i + interaction_sdpfmu.model_space.n_proton_orbitals]
        msg = f"Error in j neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.j[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.j[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = interaction_sdpfmu.neutron_model_space.isospin[i] == interaction_sdpfmu.model_space.isospin[i + interaction_sdpfmu.model_space.n_proton_orbitals]
        msg = f"Error in isospin neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.isospin[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.isospin[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = interaction_sdpfmu.neutron_model_space.parity[i] == interaction_sdpfmu.model_space.parity[i + interaction_sdpfmu.model_space.n_proton_orbitals]
        msg = f"Error in parity neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.parity[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.parity[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.model_space.n_neutron_orbitals):
        success = all(interaction_sdpfmu.neutron_model_space.jz[i] == interaction_sdpfmu.model_space.jz[i + interaction_sdpfmu.model_space.n_proton_orbitals])
        msg = f"Error in jz neutron orbital number {i} in {INTERACTION_FILE_PATH_USDA}."
        msg += f" Expected: {interaction_sdpfmu.model_space.jz[i + interaction_sdpfmu.model_space.n_proton_orbitals]}, got: {interaction_sdpfmu.neutron_model_space.jz[i]}."
        assert success, msg

def test_one_body_sdpfmu():
    reduced_matrix_element_expected = [
        -3.94780, 1.64658, -3.16354, 5.06310, 10.02201, 1.18417, 1.90327,
        -3.94780, 1.64658, -3.16354, 5.06310, 10.02201, 1.18417, 1.90327
    ]
    method_expected = 0
    n_elements_expected = 14

    success = method_expected == interaction_sdpfmu.one_body.method
    msg = f"Method incorrectly read from {INTERACTION_FILE_PATH_SDPFMU}."
    msg += f" Expected: {method_expected}, got: {interaction_sdpfmu.one_body.method}"
    assert success, msg

    success = n_elements_expected == interaction_sdpfmu.one_body.n_elements
    msg = f"n_elements incorrectly read from {INTERACTION_FILE_PATH_SDPFMU}."
    msg += f" Expected: {n_elements_expected}, got: {interaction_sdpfmu.one_body.n_elements}"
    assert success, msg

    for i in range(interaction_sdpfmu.one_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction_sdpfmu.one_body.reduced_matrix_element[i]
        msg = f"Error in one-body reduced matrix element number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction_sdpfmu.one_body.reduced_matrix_element[i]}."
        assert success, msg

def test_two_body_sdpfmu():
    reduced_matrix_element_expected = [
        -2.18680, -0.77710, -0.12727, -0.21932, -0.95881, -0.66821, -2.47058, -1.25801,  0.48068, -1.02736,  1.12715,  0.29063,  0.06247,  0.71400,  0.60381,  0.38069,  0.15405,  0.28621,  3.23032,  0.64923,  0.16862, -0.08560,  0.07708,  0.07000,  0.58936,  0.14136,  0.30870,  0.85007,  0.98215, -0.07119,  0.63781, -0.94361, -0.36993, -0.52279, -0.47688,  0.14534,  0.40693,  0.61143,  0.73170, -0.67535, -0.53335, -0.14029,  0.50230, -0.07969,  0.05706,  0.15469,  0.02178,  0.13107,  0.58010,  0.51211, -0.12811, -0.24585, -0.04188, -0.34942,  0.40723, -0.15127,  0.25943, -0.17393,  0.10551, -0.63463,  0.59143, -0.31340,  1.50533,  0.49612,  0.15070, -0.01125,  0.42270, -0.14936,  0.21282,  0.73583, -0.64781,  0.07264,  0.64133,  0.12095,  0.21562, -0.02692, -1.00110, -0.08283, -0.22235, -0.03504,  0.13977, -0.00870, -0.98295, -0.17516, -1.00338, -0.02374, -1.20208, -0.76982, -0.04352, -0.21785, -0.01141,  0.03496, -0.49285,  0.08174,  0.74422,  0.00429,  0.78936, -3.40211,  0.00696, -1.04147,  0.04838,  0.10440,  0.04556,  0.07167, -0.47033, -0.00194, -0.51353,  0.01429, -0.02892, -0.80045, -0.56152, -0.02304, -0.65989, -0.27536,  1.83069,  0.07026,  0.34510, -0.14349, -0.98635,  0.48146,  0.00757, -0.04880,  0.08942, -0.15063,  0.22753, -0.16918,  0.04352, -0.08763,  1.26287, -0.40145,  0.02238, -0.66318,  0.07964, -0.03482,  0.73216, -0.00967,  0.61024, -0.66507,  0.06688, -0.42634, -0.12530, -0.10571, -0.38898, -0.59545,  0.02906, -0.14770,  0.51696, -0.52155,  0.21132, -0.08861,  0.33495, -0.23200, -0.39103, -0.34440,  0.12969, -0.05533, -0.65831, -0.15323, -0.27038, -0.18375,  1.02594,  0.10172,  0.84277, -1.59045, -0.25805, -0.19085, -0.25320, -0.01156, -0.61998, -0.57790,  0.10735, -0.00191,  0.25764, -0.18207,  0.16618, -0.05575,  0.06711, -0.36043, -0.05959,  0.93675, -0.03657, -0.43011, -0.28657, -0.44290, -0.10548, -1.69418, -0.05157,  0.39972, -0.84030,  3.19738,  0.71133,  0.37116,  0.30653,  0.00822,  0.01969,  0.04118,  0.08977,  0.89818,  0.10020,  0.24126,  0.10439,  0.47045, -0.31518, -0.63620,  0.10121, -0.42478, -0.85517, -0.15422, -0.23040, -0.23293, -0.26308,  0.14327, -0.14884, -0.20381,  0.56065,  0.08557,  0.03080, -1.14270,  0.28518,  0.62775,  0.19771,  0.00966, -0.34882,  0.42853,  0.34378,  0.03271, -0.29890,  0.40754, -0.15116,  0.10629, -0.07919,  0.26057,  0.05193,  0.16279, -0.00634, -0.00799, -0.34068,  0.10319, -0.80989,  0.19990, -0.03375, -0.14417, -0.60338, -0.04802, -0.36320, -0.11484,  1.06752, -0.03208, -0.01473,  0.01731,  0.00941,  0.72487, -0.12357,  0.19222, -0.29211, -0.01088, -0.27114,  0.68616,  0.09257,  0.28523,  0.12807, -0.21879, -0.08873, -0.30116, -1.64772,  1.00381,  0.86933,  0.87995,  0.62222, -0.47623,  0.17982, -0.98944,  0.12658, -0.19061, -0.14546, -0.67488,  0.08181, -1.62295, -0.06754,  0.47272, -2.07783, -1.24783, -0.28743,  0.12047,  0.21670, -0.49990, -0.56430, -0.51600, -0.29690, -0.20960, -1.38320, -0.20380, -0.03310,  0.17250,  0.22240, -0.12950, -0.71740, -0.20210, -0.03670, -0.38000, -0.22964, -0.31574,  0.48946,  0.33556,  0.60256, -1.13234,  0.09590, -0.52300, -0.24860, -0.48100,  0.32240,  0.19070, -0.50220, -0.27090,  0.05210,  0.42470, -0.02680,  0.26990, -0.15370,  0.11050,  0.07170,  0.05520, -0.01530, -0.71788,  0.04632, -0.24958,  0.48202, -0.10480, -0.33510,  0.08800, -0.21460,  0.54360,  0.18360,  0.45460, -0.80300, -0.18140, -0.37380, -0.42620,  0.38517, -0.23683, -0.22480,  0.38910,  0.61110, -0.15860, -1.31026, -0.56426, -0.26456,  0.05600,  0.36150, -0.32080, -1.24570,  0.07190,  0.06000, -0.60930,  0.24519,  0.27759,  0.26279, -0.34161,  0.40430,  0.06000,  0.46310, -0.10760,  0.45450, -0.57518,  0.41502, -0.19230, -0.24900, -1.17175, -0.14395, -0.63400, -1.29280,  0.39272, -0.54168, -0.28645, -2.18680, -0.77710, -0.12727, -0.21932, -0.95881, -0.66821, -2.47058, -1.25801,  0.48068, -1.02736,  1.12715,  0.29063,  0.06247,  0.71400,  0.60381,  0.38069,  0.15405,  0.28621,  3.23032,  0.64923,  0.16862, -0.08560,  0.07708,  0.07000,  0.58936,  0.14136,  0.30870,  0.85007,  0.98215, -0.07119,  0.63781, -0.94361, -0.36993, -0.52279, -0.47688,  0.14534,  0.40693,  0.61143,  0.73170, -0.67535, -0.53335, -0.14029,  0.50230, -0.07969,  0.05706,  0.15469,  0.02178,  0.13107,  0.58010,  0.51211, -0.12811, -0.24585, -0.04188, -0.34942,  0.40723, -0.15127,  0.25943, -0.17393,  0.10551, -0.63463,  0.59143, -0.31340,  1.50533,  0.49612,  0.15070, -0.01125,  0.42270, -0.14936,  0.21282,  0.73583, -0.64781,  0.07264,  0.64133,  0.12095,  0.21562, -0.02692, -1.00110, -0.08283, -0.22235, -0.03504,  0.13977, -0.00870, -0.98295, -0.17516, -1.00338, -0.02374, -1.20208, -0.76982, -0.04352, -0.21785, -0.01141,  0.03496, -0.49285,  0.08174,  0.74422,  0.00429,  0.78936, -3.40211,  0.00696, -1.04147,  0.04838,  0.10440,  0.04556,  0.07167, -0.47033, -0.00194, -0.51353,  0.01429, -0.02892, -0.80045, -0.56152, -0.02304, -0.65989, -0.27536,  1.83069,  0.07026,  0.34510, -0.14349, -0.98635,  0.48146,  0.00757, -0.04880,  0.08942, -0.15063,  0.22753, -0.16918,  0.04352, -0.08763,  1.26287, -0.40145,  0.02238, -0.66318,  0.07964, -0.03482,  0.73216, -0.00967,  0.61024, -0.66507,  0.06688, -0.42634, -0.12530, -0.10571, -0.38898, -0.59545,  0.02906, -0.14770,  0.51696, -0.52155,  0.21132, -0.08861,  0.33495, -0.23200, -0.39103, -0.34440,  0.12969, -0.05533, -0.65831, -0.15323, -0.27038, -0.18375,  1.02594,  0.10172,  0.84277, -1.59045, -0.25805, -0.19085, -0.25320, -0.01156, -0.61998, -0.57790,  0.10735, -0.00191,  0.25764, -0.18207,  0.16618, -0.05575,  0.06711, -0.36043, -0.05959,  0.93675, -0.03657, -0.43011, -0.28657, -0.44290, -0.10548, -1.69418, -0.05157,  0.39972, -0.84030,  3.19738,  0.71133,  0.37116,  0.30653,  0.00822,  0.01969,  0.04118,  0.08977,  0.89818,  0.10020,  0.24126,  0.10439,  0.47045, -0.31518, -0.63620,  0.10121, -0.42478, -0.85517, -0.15422, -0.23040, -0.23293, -0.26308,  0.14327, -0.14884, -0.20381,  0.56065,  0.08557,  0.03080, -1.14270,  0.28518,  0.62775,  0.19771,  0.00966, -0.34882,  0.42853,  0.34378,  0.03271, -0.29890,  0.40754, -0.15116,  0.10629, -0.07919,  0.26057,  0.05193,  0.16279, -0.00634, -0.00799, -0.34068,  0.10319, -0.80989,  0.19990, -0.03375, -0.14417, -0.60338, -0.04802, -0.36320, -0.11484,  1.06752, -0.03208, -0.01473,  0.01731,  0.00941,  0.72487, -0.12357,  0.19222, -0.29211, -0.01088, -0.27114,  0.68616,  0.09257,  0.28523,  0.12807, -0.21879, -0.08873, -0.30116,  -1.64772,   1.00381,   0.86933,   0.87995,   0.62222,  -0.47623,   0.17982,  -0.98944,   0.12658,  -0.19061,  -0.14546,  -0.67488,   0.08181,  -1.62295,  -0.06754,   0.47272,  -2.07783,  -1.24783,  -0.28743,   0.12047,   0.21670,  -0.49990,  -0.56430,  -0.51600,  -0.29690,  -0.20960,  -1.38320,  -0.20380,  -0.03310,   0.17250,   0.22240,  -0.12950,  -0.71740,  -0.20210,  -0.03670,  -0.38000,  -0.22964,  -0.31574,   0.48946,   0.33556,   0.60256,  -1.13234,   0.09590,  -0.52300,  -0.24860,  -0.48100,   0.32240,   0.19070,  -0.50220,  -0.27090,   0.05210,   0.42470,  -0.02680,   0.26990,  -0.15370,   0.11050,   0.07170,   0.05520,  -0.01530,  -0.71788,   0.04632,  -0.24958,   0.48202,  -0.10480,  -0.33510,   0.08800,  -0.21460,   0.54360,   0.18360,   0.45460,  -0.80300,  -0.18140,  -0.37380,  -0.42620,   0.38517,  -0.23683,  -0.22480,   0.38910,   0.61110,  -0.15860,  -1.31026,  -0.56426,  -0.26456,   0.05600,   0.36150,  -0.32080,  -1.24570,   0.07190,   0.06000,  -0.60930,   0.24519,   0.27759,   0.26279,  -0.34161,   0.40430,   0.06000,   0.46310,  -0.10760,   0.45450,  -0.57518,   0.41502,  -0.19230,  -0.24900,  -1.17175,  -0.14395,  -0.63400,  -1.29280,   0.39272,  -0.54168,  -0.28645, -2.18680, -1.26577, -0.77710, -1.16425, -0.12727, -3.27714,  1.39484, -1.39484, -0.15509,  0.15509,  1.21831, -1.21831, -0.67798,  0.67798, -0.47250, -0.47250, -0.68110, -0.68110, -2.47058,  0.56002, -1.25801,  1.46958,  0.60466, -0.60466,  0.33989, -0.33989, -1.02736, -0.91173,  1.12715,  1.51229,  0.29063,  0.79057,  0.06247,  1.19648, -2.16484,  2.16484,  0.50487, -0.50487, -0.85399,  0.85399,  0.42696, -0.42696, -0.82915,  0.82915,  0.26919,  0.26919,  0.29798,  0.29798,  0.10893,  0.10893,  0.59789,  0.59789, -0.48744,  0.48744,  0.20238, -0.20238,  3.23032, -0.78898,  0.64923,  0.12280,  0.16862,  0.86083,  0.38057, -0.38057, -0.06053,  0.06053,  0.27636, -0.27636,  0.05450, -0.05450,  0.04950,  0.04950,  0.28414,  0.28414,  0.58936,  0.36932,  0.14136,  0.48620, -0.37688,  0.37688,  0.21828, -0.21828,  0.85007, -0.05096, -2.30314,  3.28529, -2.30314, -1.79038, -1.71919, -1.79038, -0.16104,  0.79885, -0.16104, -2.49062, -1.54702, -2.49062, -0.22250,  0.14743, -0.14743,  0.22250,  0.20517, -0.72797,  0.72797, -0.20517,  0.30968, -0.30968, -0.33721,  0.33721,  1.11527, -1.11527,  0.73498, -0.58965, -0.58965,  0.73498,  0.31328, -0.09365, -0.09365,  0.31328,  1.15393, -1.15393, -1.23548,  1.23548,  0.43235, -0.43235, -0.37982,  0.37982,  0.51739, -0.51739,  2.11298, -2.78833, -2.78833,  2.11298,  1.33592,  1.86927,  1.86927,  1.33592,  0.59812, -0.73841, -0.73841,  0.59812,  1.66734,  1.16504,  1.16504,  1.66734,  0.17723,  0.25693, -0.25693, -0.17723, -0.07510,  0.13216, -0.13216,  0.07510, -0.06940, -0.22409,  0.22409,  0.06940,  0.12089, -0.09911, -0.09911,  0.12089,  0.30017,  0.16911,  0.16911,  0.30017,  0.33953, -0.33953,  0.41019, -0.41019, -0.38532,  0.38532,  0.36211, -0.36211, -1.00857,  0.88046,  0.88046, -1.00857,  0.07879,  0.32464,  0.32464,  0.07879, -0.36810,  0.32622,  0.32622, -0.36810, -0.07716,  0.27227,  0.27227, -0.07716,  0.02721, -0.38002,  0.38002, -0.02721, -0.40359,  0.25232, -0.25232,  0.40359, -0.80875,  0.80875,  0.18345, -0.18345, -0.58963,  0.58963,  0.98685, -1.16078, -1.16078,  0.98685,  0.49653,  0.39101,  0.39101,  0.49653, -0.23985,  0.23985, -0.87858,  0.24395, -0.87858, -1.20101, -1.79244, -1.20101, -0.22161, -0.22161,  0.10348,  0.10348, -0.04863,  1.55396, -1.55396,  0.04863,  0.35081,  0.35081,  0.70267,  0.70267,  0.96369, -0.81299,  0.81299, -0.96369, -0.62913, -0.61788,  0.61788,  0.62913,  0.53629, -0.11360,
        -0.11360,  0.53629,  0.44724,  0.59660,  0.59660,  0.44724, -0.76563, -0.97844,  0.97844,  0.76563,  0.52031,  0.52031, -0.06230, -0.06230,  0.12418, -0.77199,  0.77199, -0.12418,  0.35702,  0.28438, -0.28438, -0.35702,  0.13472,  0.50660,  0.50660,  0.13472,  0.39136,  0.27041,  0.27041,  0.39136,  0.15246,  0.15246,  0.20982,  0.20982,  0.02805, -0.05497,  0.05497, -0.02805, -1.34983,  0.34873, -1.34983, -1.05359, -0.97076, -1.05359, -0.14836, -0.07399, -0.14836, -1.02014, -0.98510, -1.02014,  0.06733,  0.07244,  0.06733, -2.43744, -2.42873, -2.43744,  1.08016, -2.06312,  2.06312, -1.08016,  0.23727,  0.41242, -0.41242, -0.23727, -0.21616, -0.78722,  0.78722,  0.21616,  0.34220,  0.36593, -0.36593, -0.34220, -0.51565, -0.68643,  0.68643,  0.51565, -0.26731, -0.50252, -0.50252, -0.26731, -0.38382, -0.34030, -0.34030, -0.38382, -0.15606, -0.06179, -0.06179, -0.15606, -0.48565, -0.47424, -0.47424, -0.48565,  0.52200,  0.48704, -0.48704, -0.52200, -0.19439, -0.29846,  0.29846,  0.19439, -1.22150, -1.30324,  1.30324,  1.22150,  0.53136,  0.21286, -0.21286, -0.53136, -0.64876, -0.65305,  0.65305,  0.64876,  0.48140,  0.30796, -0.30796, -0.48140, -1.10601, -2.29610, -2.29610, -1.10601,  0.56535,  0.55839,  0.55839,  0.56535, -0.58717, -0.45430, -0.45430, -0.58717,  0.13731,  0.08893,  0.08893,  0.13731,  0.11029, -0.00588,  0.00588, -0.11029, -0.07335, -0.11891,  0.11891,  0.07335,  0.04485,  0.02682, -0.02682, -0.04485, -0.36697, -0.10336, -0.10336, -0.36697,  0.04402,  0.04595,  0.04595,  0.04402, -0.34674, -0.16679, -0.16679, -0.34674, -0.88890, -0.90319, -0.90319, -0.88890, -0.16895, -0.14003,  0.14003,  0.16895, -0.29633, -0.50412,  0.50412,  0.29633, -0.46869, -0.09283, -0.09283, -0.46869, -0.34980, -0.32675, -0.32675, -0.34980, -0.06417, -0.59572,  0.59572,  0.06417, -2.66651,  2.39115, -2.66651, -1.40078, -3.23147, -1.40078, -1.07699,  1.14725, -1.07699, -0.97987, -1.32497, -0.97987, -0.63632,  0.49283, -0.63632, -1.80102, -0.81468, -1.80102, -0.14768, -0.62914,  0.62914,  0.14768, -0.04232,  0.04989, -0.04989,  0.04232,  0.00564,  0.05444, -0.05444, -0.00564,  0.18230, -0.09288,  0.09288, -0.18230, -0.17338,  0.02276,  0.02276, -0.17338, -0.16941, -0.39693, -0.39693, -0.16941,  0.58669, -0.75587, -0.75587,  0.58669, -0.96627, -1.00979, -1.00979, -0.96627,  0.36307, -0.45070, -0.45070,  0.36307, -0.66641, -1.92928, -1.92928, -0.66641, -1.46018, -1.05873,  1.05873,  1.46018,  0.44344, -0.42107,  0.42107, -0.44344, -0.49570,  0.16748, -0.16748,  0.49570,  0.85898, -0.77934,  0.77934, -0.85898,  1.78435, -1.81917, -1.81917,  1.78435, -0.19687, -0.92903, -0.92903, -0.19687,  0.28018, -0.28985, -0.28985,  0.28018,  0.07115, -0.53909, -0.53909,  0.07115, -0.06318,  0.60189, -0.60189,  0.06318,  0.46896, -0.40207,  0.40207, -0.46896,  0.41428,  0.84063, -0.84063, -0.41428,  0.21791, -0.34321,  0.34321, -0.21791, -0.83748,  0.73177,  0.73177, -0.83748, -0.91902, -0.53003, -0.53003, -0.91902,  0.14404,  0.73949, -0.73949, -0.14404,  0.49331, -0.46424,  0.46424, -0.49331, -1.23963,  1.09193,  1.09193, -1.23963, -0.36627, -0.88323, -0.88323, -0.36627, -0.87845,  0.35690, -0.87845, -0.58899, -0.80030, -0.58899, -0.19494,  0.10633, -0.19494, -1.08320, -1.41815, -1.08320,  0.56879,  0.80080, -0.80080, -0.56879, -0.28960, -0.10144,  0.10144,  0.28960, -0.70392, -0.35952,  0.35952,  0.70392,  0.10151,  0.02819, -0.02819, -0.10151, -0.30371, -0.24838,  0.24838,  0.30371, -0.36591, -0.29240, -0.29240, -0.36591,  0.17279,  0.32603,  0.32603,  0.17279, -0.07154, -0.19884, -0.19884, -0.07154,  0.05506,  0.23882,  0.23882,  0.05506, -0.26001,  1.28594, -1.28594,  0.26001, -0.24468, -0.34640,  0.34640,  0.24468,  0.21251,  0.63026, -0.63026, -0.21251, -0.31222, -1.27824, -1.27824, -0.31222,  0.07739,  0.33544,  0.33544,  0.07739, -0.00565, -0.18520, -0.18520, -0.00565, -1.04922, -0.79601, -0.79601, -1.04922, -0.39668, -0.38512,  0.38512,  0.39668, -0.41366, -0.20632,  0.20632,  0.41366, -0.16259, -0.41530, -0.41530, -0.16259, -0.33954, -0.44689, -0.44689, -0.33954, -0.17965,  0.17774, -0.17774,  0.17965, -0.81387,  1.07151, -0.81387, -0.76433, -0.58226, -0.76433,  0.56268, -0.39650, -0.39650,  0.56268, -0.32803, -0.27228, -0.27228, -0.32803, -0.33952,  0.40664, -0.40664,  0.33952, -0.23991,  0.12052, -0.12052,  0.23991,  0.44116, -0.50075, -0.50075,  0.44116, -0.22307, -1.15982, -1.15982, -0.22307,  0.17633, -0.21290,  0.21290, -0.17633,  0.11590,  0.54601, -0.54601, -0.11590, -0.19814, -0.08843, -0.08843, -0.19814, -0.60360, -0.16071, -0.16071, -0.60360,  0.28397, -0.38945,  0.38945, -0.28397, -1.69418, -1.09747, -0.05157, -2.23683, -0.21842,  0.21842,  0.28264, -0.28264, -0.84030,  0.02133,  3.19738, -1.06251,  0.71133, -0.25550,  1.06626, -1.06626,  0.26245, -0.26245, -0.35592,  0.35592,  0.21675,  0.21675, -0.19313, -0.19313,  0.46172, -0.46172,  0.00822,  0.75767,  0.01969,  0.75963,  0.15502, -0.15502,  0.02912, -0.02912,  0.43583, -0.43583,  0.06348,  0.06348,  0.37711,  0.37711,  0.89818,  0.07854,  0.10020,  0.71499, -0.24583,  0.24583,  0.17060, -0.17060,  0.10439,  0.22347, -1.42948,  1.89993, -1.42948, -0.86310, -0.54792, -0.86310, -0.68555,  0.68555, -0.25105,  0.25105, -0.44986,  0.44986, -0.21026,  0.31147,  0.31147, -0.21026,  0.87561,  1.30038,  1.30038,  0.87561, -0.02961,  0.82556, -0.82556,  0.02961,  0.40836, -0.40836, -0.10905,  0.10905,  0.82795, -1.05835, -1.05835,  0.82795,  0.43232,  0.66525,  0.66525,  0.43232, -0.35927, -0.09619,  0.09619,  0.35927,  0.23003, -0.23003,  0.10131, -0.10131, -0.35438,  0.20554,  0.20554, -0.35438, -0.05107,  0.15274,  0.15274, -0.05107,  0.03221, -0.03221, -1.95554,  2.51619, -1.95554, -1.05396, -1.13954, -1.05396, -0.60300,  0.63380, -0.60300, -1.85798, -0.71528, -1.85798,  0.44024, -0.15507,  0.15507, -0.44024,  0.09754, -0.53021,  0.53021, -0.09754, -0.34078,  0.53849, -0.53849,  0.34078, -0.50070,  0.51036,  0.51036, -0.50070, -0.41848, -0.06966, -0.06966, -0.41848,  0.12936,  0.29917, -0.29917, -0.12936,  0.69908,  0.35530, -0.35530, -0.69908, -0.64035,  0.67306, -0.67306,  0.64035,  0.19115, -0.49005, -0.49005,  0.19115, -0.40498, -0.81252, -0.81252, -0.40498, -0.96609,  0.81494, -0.81494,  0.96609, -0.32147,  0.42776, -0.32147, -0.82236, -0.74316, -0.82236,  0.18880,  0.07178,  0.18880, -1.89013, -1.94206, -1.89013,  0.16242,  0.00037, -0.00037, -0.16242, -0.23432, -0.22798,  0.22798,  0.23432, -0.00652, -0.00147,  0.00147,  0.00652, -0.08479, -0.25589, -0.25589, -0.08479, -0.25592, -0.35911, -0.35911, -0.25592, -0.40816, -0.40173, -0.40173, -0.40816,  0.15118, -0.04872, -0.04872,  0.15118,  0.70370,  0.73744, -0.73744, -0.70370, -0.06837, -0.07579,  0.07579,  0.06837, -0.19514, -0.40824, -0.40824, -0.19514, -0.21265, -0.16463, -0.16463, -0.21265, -0.33229, -0.03091,  0.03091,  0.33229, -1.09519,  0.98035, -1.09519, -0.52519, -1.59271, -0.52519, -0.45305,  0.42097, -0.45305, -0.80912, -0.79439, -0.80912,  0.57213,  0.55483, -0.55483, -0.57213, -0.36539,  0.37480, -0.37480,  0.36539,  0.73786,  0.01299, -0.01299, -0.73786,  0.20299, -0.32656, -0.32656,  0.20299, -0.33739, -0.52961, -0.52961, -0.33739,  0.00373,  0.29584, -0.29584, -0.00373, -0.04699,  0.03611, -0.03611,  0.04699,  0.59799, -0.86914, -0.86914,  0.59799,  0.13125, -0.55491, -0.55491,  0.13125, -0.34974,  0.44232, -0.34974, -0.70168, -0.98691, -0.70168,  0.97071,  0.84264, -0.84264, -0.97071, -0.21193, -0.00686, -0.00686, -0.21193,  0.23981,  0.32854,  0.32854,  0.23981, -0.00557, -0.29559,  0.29559,  0.00557, -1.64772, -2.53045,  1.00381,  0.64091, -1.65816,  1.65816,  0.86933, -0.00312, -0.13288,  0.13288,  0.87995,  0.71185, -1.40400,  1.40400,  0.62222,  0.86002, -0.69491,  0.21868, -0.69491, -1.06035, -1.24018, -1.06035,  0.03274, -1.02218,  1.02218, -0.03274, -1.03991,  1.16650, -1.03991, -0.70437, -0.51376, -0.70437,  0.04254, -0.18800,  0.18800, -0.04254, -0.88571,  0.21082, -0.88571, -1.44740, -1.52921, -1.44740, -0.03611, -1.58684,  1.58684,  0.03611, -1.32652,  1.25898, -1.32652, -0.86017, -1.33289, -0.86017, -2.07783, -1.40357, -1.24783, -0.96157, -0.28743, -0.90367,  0.12047, -2.78587,  1.34336, -1.34336,  0.15323, -0.15323,  0.77195, -0.77195, -0.35348,  0.35348,  0.90884, -0.90884, -0.39902,  0.39902, -0.36487, -0.36487, -0.62275, -0.62275, -0.20994, -0.20994, -0.30158, -0.30158,  0.62812, -0.62812, -0.14821,  0.14821, -1.38320,  0.65110, -0.20380,  0.43580, -0.03310,  0.12390,  0.06413, -0.06413,  0.12198, -0.12198, -0.05317,  0.05317,  0.15726, -0.15726, -0.09157, -0.09157, -0.07651, -0.07651, -0.71740, -0.43130, -0.20210, -0.34150,  0.22274, -0.22274, -0.02595,  0.02595, -0.38000,  0.02710, -2.45477,  2.22513, -2.45477, -1.83372, -1.51798, -1.83372, -0.50607,  0.99553, -0.50607, -1.01122, -1.34678, -1.01122, -0.05277,  0.65533, -0.05277, -2.13357, -1.00123, -2.13357, -0.27110, -0.36700,  0.36700,  0.27110, -0.13450, -0.38850,  0.38850,  0.13450, -0.02675,  0.22185, -0.22185,  0.02675,  0.09665, -0.57765,  0.57765, -0.09665, -0.05140,  0.37380,  0.37380, -0.05140, -0.09410, -0.28480, -0.28480, -0.09410, -0.19339,  0.19339, -0.35511,  0.35511,  0.45099, -0.45099, -0.19156,  0.19156,  0.79917, -0.79917,  0.66210, -0.61000, -0.61000,  0.66210, -0.08665, -0.51135, -0.51135, -0.08665,  0.37240, -0.39920, -0.39920,  0.37240, -0.18545, -0.45535, -0.45535, -0.18545,  0.19550,  0.34920, -0.34920, -0.19550,  0.36835, -0.25785,  0.25785, -0.36835,  0.63031, -0.63031,  0.05070, -0.05070,  0.44293, -0.44293, -0.70495,  0.76015,  0.76015, -0.70495, -0.37935, -0.36405, -0.36405, -0.37935,  0.13633, -0.13633, -0.66622, -0.05166, -0.66622, -0.51732, -0.56364, -0.51732, -0.34727,  0.09769, -0.34727, -1.28022, -1.76224, -1.28022,  0.79010,  0.89490, -0.89490, -0.79010, -0.08225, -0.25285,  0.25285,  0.08225,  0.06223,  0.06223,  0.11738,  0.11738, -0.15175, -0.15175,  0.02362,  0.02362, -0.36360,  0.90720, -0.90720,  0.36360, -0.19770, -0.38130,  0.38130,  0.19770, -0.12785,  0.58245, -0.58245,  0.12785,  0.14515, -0.94815, -0.94815,  0.14515,  0.27065,  0.45205,  0.45205,  0.27065, -0.26432,
        -0.26432, -0.30469, -0.30469, -0.52450,  0.09830, -0.09830,  0.52450, -0.69055,  1.07572, -0.69055, -0.68325, -0.44642, -0.68325, -0.18533,  0.18533, -0.15896,  0.15896,  0.21225,  0.17685,  0.17685,  0.21225, -0.37480, -0.98590, -0.98590, -0.37480,  0.14595, -0.30455,  0.30455, -0.14595,  0.45333, -0.45333, -1.31026, -0.90438, -0.56426, -0.60918, -0.26456, -2.33088, -0.33729,  0.33729,  0.03960, -0.03960, -0.22627,  0.22627,  0.25562, -0.25562, -0.22684, -0.22684, -0.44378, -0.44378, -1.24570,  0.04830,  0.07190, -0.05460, -0.02383,  0.02383,  0.04243, -0.04243, -0.60930, -0.31610, -1.25587,  1.50106, -1.25587, -0.63207, -0.90966, -0.63207, -0.17692,  0.43971, -0.17692, -0.73027, -0.38866, -0.73027,  0.37915, -0.02515,  0.02515, -0.37915, -0.47755,  0.53755, -0.53755,  0.47755, -0.16780,  0.16780,  0.32746, -0.32746, -0.16094,  0.16094,  0.44270, -0.55030, -0.55030,  0.44270, -0.01700, -0.47150, -0.47150, -0.01700, -0.57537,  0.57537, -0.44996, -0.12521, -0.44996, -0.49731, -0.91233, -0.49731, -0.13598, -0.13598,  0.08132,  0.08132,  0.22050, -0.46950,  0.46950, -0.22050, -1.17175, -0.65821, -0.14395, -2.31641,  1.27696, -1.27696, -0.44831,  0.44831, -1.29280,  0.76750, -1.06303,  1.45575, -1.06303, -1.43293, -0.89125, -1.43293,  0.60033, -0.60033, -0.28645, -1.25547
    ]

    orbital_0_expected = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 5, 5, 1, 5, 5, 1, 5, 5, 1, 5, 5, 1, 5, 5, 1, 5, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 7, 7, 1, 7, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 5, 5, 2, 5, 5, 2, 5, 5, 2, 5, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 6, 6, 2, 6, 6, 2, 6, 6, 2, 6, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 7, 7, 2, 7, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 3, 5, 5, 3, 5, 5, 3, 5, 3, 5, 3, 6, 6, 3, 6, 6, 3, 6, 3, 6, 3, 7, 7, 3, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 6, 6, 4, 6, 6, 4, 6, 6, 4, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 7, 7, 4, 7, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 6, 6, 5, 6, 6, 5, 6, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 7, 7, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 6, 7, 7, 7
    ]) - 1  # Indices start at 1 in the .snt files but from 0 when read.

    orbital_1_expected = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 13, 13, 13, 13, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 9, 8, 8, 9, 8, 8, 9, 8, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 10, 8, 8, 10, 8, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 11, 8, 8, 11, 8, 8, 11, 8, 8, 11, 8, 8, 11, 8, 8, 11, 8, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 13, 8, 8, 13, 8, 8, 13, 8, 8, 13, 8, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 13, 8, 14, 8, 8, 14, 8, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 14, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 11, 9, 9, 11, 9, 9, 11, 9, 9, 11, 9, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 12, 9, 9, 12, 9, 9, 12, 9, 9, 12, 9, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 12, 9, 13, 9, 9, 13, 9, 9, 13, 9, 9, 13, 9, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 13, 9, 14, 9, 9, 14, 9, 9, 14, 9, 14, 9, 14, 9, 14, 9, 14, 9, 14, 9, 14, 9, 14, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10, 11, 10, 10, 11, 10, 11, 10, 12, 10, 10, 12, 10, 10, 12, 10, 12, 10, 13, 10, 10, 13, 10, 10, 13, 10, 13, 10, 14, 10, 10, 14, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 13, 11, 11, 13, 11, 11, 13, 11, 11, 13, 11, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 14, 11, 11, 14, 11, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 14, 12, 12, 14, 12, 12, 14, 12, 14, 12, 14, 12, 14, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 14, 14
    ]) - 1

    orbital_2_expected = np.array([
        1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 5, 5, 5, 5, 5, 5, 6, 6, 6, 5, 5, 6, 6, 6, 6, 6, 7, 6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 8, 8, 8, 8, 8, 8, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 8, 8, 9, 9, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 9, 9, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 9, 9, 10, 10, 10, 10, 10, 11, 12, 13, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 12, 12, 12, 12, 12, 12, 13, 13, 13, 12, 12, 13, 13, 13, 13, 13, 14, 13, 13, 14, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 3, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 6, 4, 6, 4, 6, 4, 6, 4, 7, 4, 7, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 7, 7, 4, 4, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 7, 7, 7, 7, 1, 1, 3, 1, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 7, 7, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 7, 7, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 7, 7, 1, 1, 7, 7, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 6, 6, 1, 1, 7, 7, 1, 1, 7, 7, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 7, 7, 1, 1, 7, 7, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 1, 1, 7, 1, 1, 7, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5, 4, 6, 4, 6, 4, 7, 5, 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 2, 2, 3, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 7, 7, 7, 7, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 7, 7, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 7, 7, 2, 2, 7, 7, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 2, 2, 7, 2, 2, 7, 3, 3, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 7, 7, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 3, 3, 4, 3, 3, 4, 3, 3, 5, 5, 3, 3, 5, 3, 3, 5, 3, 3, 6, 6, 3, 3, 6, 3, 3, 6, 3, 3, 7, 7, 3, 3, 7, 3, 3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 6, 4, 6, 4, 6, 4, 6, 4, 7, 4, 7, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 7, 7, 4, 4, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 7, 7, 7, 7, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 7, 7, 4, 4, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 7, 7, 4, 4, 7, 4, 4, 7, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 7, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 7, 7, 7, 7, 5, 5, 7, 5, 5, 7, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 6, 6, 7, 6, 6, 7, 7, 7, 7, 7
    ]) - 1

    orbital_3_expected = np.array([
        1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 5, 5, 5, 6, 6, 7, 6, 6, 7, 7, 2, 2, 2, 2, 3, 3, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 5, 5, 6, 6, 6, 6, 7, 7, 6, 7, 7, 3, 3, 2, 3, 4, 5, 5, 6, 6, 7, 5, 6, 6, 7, 7, 6, 7, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7, 7, 4, 4, 5, 5, 6, 6, 7, 4, 5, 5, 6, 2, 2, 3, 3, 4, 4, 5, 6, 5, 5, 6, 7, 6, 6, 7, 7, 3, 3, 4, 5, 5, 6, 5, 6, 6, 7, 6, 7, 7, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 4, 4, 5, 5, 6, 5, 5, 5, 5, 6, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 6, 6, 6, 6, 7, 7, 4, 5, 5, 6, 6, 7, 7, 7, 7, 5, 6, 6, 7, 3, 4, 5, 6, 7, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 5, 5, 5, 6, 6, 7, 6, 6, 7, 7, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 5, 5, 6, 6, 6, 6, 7, 7, 6, 7, 7, 6, 6, 6, 6, 7, 7, 5, 5, 6, 6, 6, 7, 7, 6, 7, 7, 7, 5, 6, 6, 7, 5, 5, 5, 6, 6, 7, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 7, 6, 7, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 12, 12, 12, 13, 13, 14, 13, 13, 14, 14, 9, 9, 9, 9, 10, 10, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 14, 14, 13, 14, 14, 10, 10, 9, 10, 11, 12, 12, 13, 13, 14, 12, 13, 13, 14, 14, 13, 14, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 11, 11, 12, 12, 13, 13, 14, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 13, 13, 14, 14, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 11, 11, 12, 12, 13, 13, 14, 14, 14, 11, 11, 12, 12, 13, 13, 14, 11, 12, 12, 13, 9, 9, 10, 10, 11, 11, 12, 13, 12, 12, 13, 14, 13, 13, 14, 14, 10, 10, 11, 12, 12, 13, 12, 13, 13, 14, 13, 14, 14, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 11, 11, 12, 12, 13, 12, 12, 12, 12, 13, 13, 13, 14, 14, 11, 11, 12, 12, 13, 13, 14, 13, 13, 13, 13, 14, 14, 11, 12, 12, 13, 13, 14, 14, 14, 14, 12, 13, 13, 14, 10, 11, 12, 13, 14, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 12, 12, 12, 13, 13, 14, 13, 13, 14, 14, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 14, 14, 13, 14, 14, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 14, 14, 13, 14, 14, 14, 12, 13, 13, 14, 12, 12, 12, 13, 13, 14, 13, 13, 14, 14, 13, 13, 13, 13, 14, 14, 13, 14, 14, 14, 14, 13, 14, 13, 13, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 9, 8, 9, 8, 9, 8, 9, 8, 10, 8, 10, 8, 9, 9, 9, 9, 10, 9, 10, 9, 10, 10, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 13, 11, 13, 11, 13, 11, 13, 11, 14, 11, 14, 11, 12, 12, 12, 12, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 14, 12, 14, 12, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 10, 10, 8, 8, 10, 10, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 13, 13, 11, 11, 13, 13, 11, 11, 13, 13, 11, 11, 14, 14, 11, 11, 14, 14, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 10, 10, 8, 10, 10, 8, 9, 9, 9, 9, 10, 10, 9, 9, 11, 11, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 13, 13, 11, 11, 13, 13, 11, 11, 14, 14, 11, 11, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 14, 14, 13, 13, 11, 11, 8, 11, 11, 8, 11, 11, 8, 11, 11, 8, 11, 11, 8, 11, 11, 8, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 8, 8, 13, 13, 8, 8, 13, 13, 8, 8, 13, 13, 8, 8, 13, 13, 8, 8, 14, 14, 8, 8, 14, 14, 8, 8, 11, 11, 9, 9, 11, 11, 9, 9, 11, 11, 9, 9, 11, 11, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 12, 12, 8, 12, 12, 8, 12, 12, 8, 12, 12, 8, 12, 12, 8, 12, 12, 8, 13, 13, 8, 8, 13, 13, 8, 8, 13, 13, 8, 8, 13, 13, 8, 8, 14, 14, 8, 8, 14, 14, 8, 8, 11, 11, 9, 9, 11, 11, 9, 9, 11, 11, 9, 9, 11, 11, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 14, 14, 10, 10, 13, 13, 8, 13, 13, 8, 13, 13, 8, 13, 13, 8, 14, 14, 8, 8, 14, 14, 8, 8, 11, 11, 9, 9, 11, 11, 9, 9, 11, 11, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 14, 14, 8, 14, 14, 8, 11, 11, 9, 9, 11, 11, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 9, 9, 9, 9, 10, 9, 10, 9, 10, 10, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 13, 11, 13, 11, 14, 11, 12, 12, 12, 12, 13, 12, 13, 12, 13, 12, 14, 12, 14, 12, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 10, 10, 9, 10, 10, 9, 10, 10, 11, 11, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 13, 13, 11, 11, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 11, 11, 9, 11, 11, 9, 11, 11, 9, 11, 11, 9, 12, 12, 9, 9, 12, 12, 9, 9, 12, 12, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 12, 12, 9, 12, 12, 9, 12, 12, 9, 12, 12, 9, 13, 13, 9, 9, 13, 13, 9, 9, 13, 13, 9, 9, 14, 14, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 13, 13, 9, 13, 13, 9, 13, 13, 9, 13, 13, 9, 14, 14, 9, 9, 14, 14, 9, 9, 11, 11, 10, 10, 12, 12, 10, 10, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 14, 14, 10, 10, 14, 14, 9, 14, 14, 9, 12, 12, 10, 10, 13, 13, 10, 10, 13, 13, 10, 10, 14, 14, 10, 10, 10, 10, 11, 11, 12, 11, 12, 12, 13, 12, 13, 13, 14, 13, 14, 14, 11, 11, 10, 11, 11, 10, 12, 12, 10, 10, 12, 12, 10, 12, 12, 10, 13, 13, 10, 10, 13, 13, 10, 13, 13, 10, 14, 14, 10, 10, 14, 14, 10, 14, 14, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 13, 11, 13, 11, 13, 11, 13, 11, 14, 11, 14, 11, 12, 12, 12, 12, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 14, 12, 14, 12, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 12, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 11, 12, 12, 11, 13, 13, 11, 11, 13, 13, 11, 11, 13, 13, 11, 11, 13, 13, 11, 11, 14, 14, 11, 11, 14, 14, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 13, 13, 11, 13, 13, 11, 13, 13, 11, 13, 13, 11, 14, 14, 11, 11, 14, 14, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 14, 14, 13, 13, 14, 14, 11, 14, 14, 11, 12, 12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 12, 14, 14, 12, 12, 13, 13, 12, 12, 12, 12, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 14, 12, 14, 12, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 13, 13, 12, 13, 13, 12, 13, 13, 12, 13, 13, 12, 14, 14, 12, 12, 14, 14, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 14, 14, 12, 14, 14, 12, 13, 13, 13, 13, 14, 14, 13, 13, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14
    ]) - 1

    jj_expected = np.array([
        0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 0, 2, 4, 2, 4, 2, 4, 4, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 2, 4, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 1, 2, 3, 4, 2, 3, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 3, 4, 5, 2, 3, 4, 2, 3, 2, 3, 4, 2, 3, 2, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 0, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 0, 2, 4, 6, 2, 4, 6, 2, 4, 4, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 3, 4, 2, 4, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 4, 5, 3, 4, 2, 4, 2, 3, 4, 2, 3, 2, 2, 3, 4, 4, 3, 4, 3, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 2, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 0, 2, 4, 2, 4, 2, 4, 4, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 2, 4, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 1, 2, 3, 4, 2, 3, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 3, 4, 5, 2, 3, 4, 2, 3, 2, 3, 4, 2, 3, 2, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 1, 0, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 0, 2, 4, 6, 2, 4, 6, 2, 4, 4, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 3, 4, 2, 4, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 4, 5, 3, 4, 2, 4, 2, 3, 4, 2, 3, 2, 2, 3, 4, 4, 3, 4, 3, 0, 2, 4, 2, 4, 2, 0, 2, 2, 0, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 3, 2, 2, 0, 2, 2, 0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 0, 1, 2, 3, 4, 5, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 2, 2, 3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 0, 1, 2, 3, 4, 5, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 0, 1, 2, 3, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 2, 2, 3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 0, 1, 2, 3, 4, 5, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 0, 1, 2, 3, 4, 5, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 0, 1
    ])

    n_elements_expected = 2116
    method_expected = 1
    im0_expected = 42
    pwr_expected = -0.300000

    success = n_elements_expected == interaction_sdpfmu.two_body.n_elements
    msg = f"The number of two-body matrix elements was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {n_elements_expected}, got: {interaction_sdpfmu.two_body.n_elements}."
    assert success, msg

    success = method_expected == interaction_sdpfmu.two_body.method
    msg = f"The two-body method was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {method_expected}, got: {interaction_sdpfmu.two_body.method}."
    assert success, msg

    success = im0_expected == interaction_sdpfmu.two_body.im0
    msg = f"The two-body im0 was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {im0_expected}, got: {interaction_sdpfmu.two_body.im0}."
    assert success, msg

    success = pwr_expected == interaction_sdpfmu.two_body.pwr
    msg = f"The two-body pwr was not correctly read from '{INTERACTION_FILE_PATH_SDPFMU}'."
    msg += f" Expected: {pwr_expected}, got: {interaction_sdpfmu.two_body.pwr}."
    assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = reduced_matrix_element_expected[i] == interaction_sdpfmu.two_body.reduced_matrix_element[i]
        msg = f"Error in two-body reduced matrix element number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {reduced_matrix_element_expected[i]}, got: {interaction_sdpfmu.two_body.reduced_matrix_element[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = orbital_0_expected[i] == interaction_sdpfmu.two_body.orbital_0[i]
        msg = f"Error in orbital 0 number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {orbital_0_expected[i]}, got: {interaction_sdpfmu.two_body.orbital_0[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = orbital_1_expected[i] == interaction_sdpfmu.two_body.orbital_1[i]
        msg = f"Error in orbital 1 number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {orbital_1_expected[i]}, got: {interaction_sdpfmu.two_body.orbital_1[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = orbital_2_expected[i] == interaction_sdpfmu.two_body.orbital_2[i]
        msg = f"Error in orbital 2 number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {orbital_2_expected[i]}, got: {interaction_sdpfmu.two_body.orbital_2[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = orbital_3_expected[i] == interaction_sdpfmu.two_body.orbital_3[i]
        msg = f"Error in orbital 3 number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {orbital_3_expected[i]}, got: {interaction_sdpfmu.two_body.orbital_3[i]}."
        assert success, msg

    for i in range(interaction_sdpfmu.two_body.n_elements):
        success = jj_expected[i] == interaction_sdpfmu.two_body.jj[i]
        msg = f"Error in jj number {i} in {INTERACTION_FILE_PATH_SDPFMU}."
        msg += f" Expected: {jj_expected[i]}, got: {interaction_sdpfmu.two_body.jj[i]}."
        assert success, msg

def test_proton_j_couple_sdpfmu():
    """
    Test the construction of the proton j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 0  # Protons.

    n_couplings_expected = [
        7, 5, 13, 7, 8, 2, 2, 0
    ]
    idx_expected = [
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        [1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
        [1, 1, 1, 2, 1, 3, 2, 2, 2, 3, 4, 4, 4, 5, 4, 6, 5, 5, 5, 6, 5, 7, 6, 6, 6, 7],
        [1, 2, 1, 3, 4, 5, 4, 6, 4, 7, 5, 6, 5, 7],
        [1, 1, 1, 2, 4, 4, 4, 5, 4, 6, 4, 7, 5, 5, 5, 6],
        [4, 5, 4, 6],
        [4, 4, 4, 5],
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 7, 9, 0, 0, 0, 0, 0, 8,10,12, 0, 0, 0, 0, 0,11,13, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 5, 7, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 5, 8, 0, 0, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 0  # Protons.

    n_couplings_expected = [
        3, 8, 10, 9, 6, 3, 1, 0
    ]
    idx_expected = [
        [1, 5, 2, 6, 3, 7],
        [1, 4, 1, 5, 1, 6, 2, 5, 2, 6, 2, 7, 3, 6, 3, 7],
        [1, 4, 1, 5, 1, 6, 1, 7, 2, 4, 2, 5, 2, 6, 2, 7, 3, 5, 3, 6],
        [1, 4, 1, 5, 1, 6, 1, 7, 2, 4, 2, 5, 2, 6, 3, 4, 3, 5],
        [1, 4, 1, 5, 1, 6, 2, 4, 2, 5, 3, 4],
        [1, 4, 1, 5, 2, 4],
        [1, 4],
    ]
    idx_reverse_expected = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7,10, 0, 0, 0, 0, 4, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 8, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

def test_neutron_j_couple_sdpfmu():
    """
    Test the construction of the neutron j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 1  # Neutrons.

    n_couplings_expected = [
        7, 5, 13, 7, 8, 2, 2, 0
    ]
    idx_expected = [
        [8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14],
        [8, 9, 9,10,11,12,12,13,13,14],
        [8, 8, 8, 9, 8,10, 9, 9, 9,10,11,11,11,12,11,13,12,12,12,13,12,14,13,13,13,14],
        [8, 9, 8,10,11,12,11,13,11,14,12,13,12,14],
        [8, 8, 8, 9,11,11,11,12,11,13,11,14,12,12,12,13],
        [11,12,11,13],
        [11,11,11,12],
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 7, 9, 0, 0, 0, 0, 0, 8,10,12, 0, 0, 0, 0, 0,11,13, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 5, 7, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 5, 8, 0, 0, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            idx += interaction_sdpfmu.proton_model_space.n_orbitals
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 1  # Neutrons.

    n_couplings_expected = [
        3, 8, 10, 9, 6, 3, 1, 0
    ]
    idx_expected = [
        [8,12, 9,13,10,14],
        [8,11, 8,12, 8,13, 9,12, 9,13, 9,14,10,13,10,14],
        [8,11, 8,12, 8,13, 8,14, 9,11, 9,12, 9,13, 9,14,10,12,10,13],
        [8,11, 8,12, 8,13, 8,14, 9,11, 9,12, 9,13,10,11,10,12],
        [8,11, 8,12, 8,13, 9,11, 9,12,10,11],
        [8,11, 8,12, 9,11],
        [8,11],
    ]
    idx_reverse_expected = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7,10, 0, 0, 0, 0, 4, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 8, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            idx += interaction_sdpfmu.proton_model_space.n_orbitals
            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

def test_proton_neutron_j_couple_sdpfmu():
    """
    Test the construction of the proton-neutron j_couple data structure.
    """
    parity_idx = 0  # Positive.
    proton_neutron_idx = 2  # Proton-neutron.

    n_couplings_expected = [
        7, 17, 21, 19, 13, 7, 3, 1
    ]
    idx_expected = [
        [1, 8, 2, 9, 3,10, 4,11, 5,12, 6,13, 7,14],
        [1, 8, 1, 9, 2, 8, 2, 9, 2,10, 3, 9, 3,10, 4,11, 4,12, 5,11, 5,12, 5,13, 6,12, 6,13, 6,14, 7,13, 7,14],
        [1, 8, 1, 9, 1,10, 2, 8, 2, 9, 2,10, 3, 8, 3, 9, 4,11, 4,12, 4,13, 5,11, 5,12, 5,13, 5,14, 6,11, 6,12, 6,13, 6,14, 7,12, 7,13],
        [1, 8, 1, 9, 1,10, 2, 8, 2, 9, 3, 8, 4,11, 4,12, 4,13, 4,14, 5,11, 5,12, 5,13, 5,14, 6,11, 6,12, 6,13, 7,11, 7,12],
        [1, 8, 1, 9, 2, 8, 4,11, 4,12, 4,13, 4,14, 5,11, 5,12, 5,13, 6,11, 6,12, 7,11],
        [1, 8, 4,11, 4,12, 4,13, 5,11, 5,12, 6,11],
        [4,11, 4,12, 5,11],
        [4,11],
    ]
    idx_reverse_expected = [
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7],
        [1, 3, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 0, 8,10, 0, 0, 0, 0, 0, 9,11,13, 0, 0, 0, 0, 0,12,14,16, 0, 0, 0, 0, 0,15,17],
        [1, 4, 7, 0, 0, 0, 0, 2, 5, 8, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 9,12,16, 0, 0, 0, 0,10,13,17,20, 0, 0, 0,11,14,18,21, 0, 0, 0, 0,15,19, 0],
        [1, 4, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,11,15,18, 0, 0, 0, 8,12,16,19, 0, 0, 0, 9,13,17, 0, 0, 0, 0,10,14, 0, 0],
        [1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8,11,13, 0, 0, 0, 5, 9,12, 0, 0, 0, 0, 6,10, 0, 0, 0, 0, 0, 7, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 7, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            if j%2 != 0:
                """
                In the Fortran KSHELL code, the neutron orbitals are
                counted after the proton orbitals. This is why we must
                add the number of proton orbitals to the neutron
                orbitals which are every other orbital.

                In the Fortran code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: n_proton_orbitals, n_proton_orbitals + 1, ..., n_orbitals - 1

                In the Python code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: 0, 1, 2, ..., n_neutron_orbitals - 1
                """
                idx += interaction_sdpfmu.proton_model_space.n_orbitals

            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

    parity_idx = 1  # Negative.
    proton_neutron_idx = 2  # Proton-neutron.

    n_couplings_expected = [
        6, 16, 20, 18, 12, 6, 2, 0
    ]
    idx_expected = [
        [1,12, 2,13, 3,14, 5, 8, 6, 9, 7,10],
        [1,11, 1,12, 1,13, 2,12, 2,13, 2,14, 3,13, 3,14, 4, 8, 5, 8, 5, 9, 6, 8, 6, 9, 6,10, 7, 9, 7,10],
        [1,11, 1,12, 1,13, 1,14, 2,11, 2,12, 2,13, 2,14, 3,12, 3,13, 4, 8, 4, 9, 5, 8, 5, 9, 5,10, 6, 8, 6, 9, 6,10, 7, 8, 7, 9],
        [1,11, 1,12, 1,13, 1,14, 2,11, 2,12, 2,13, 3,11, 3,12, 4, 8, 4, 9, 4,10, 5, 8, 5, 9, 5,10, 6, 8, 6, 9, 7, 8],
        [1,11, 1,12, 1,13, 2,11, 2,12, 3,11, 4, 8, 4, 9, 4,10, 5, 8, 5, 9, 6, 8],
        [1,11, 1,12, 2,11, 4, 8, 4, 9, 5, 8],
        [1,11, 4, 8],
    ]
    idx_reverse_expected = [
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 9,10,12, 0, 0, 0, 0, 0,11,13,15, 0, 0, 0, 0, 0,14,16, 1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0, 0],
        [0, 0, 0,11,13,16,19, 0, 0, 0,12,14,17,20, 0, 0, 0, 0,15,18, 0, 1, 5, 0, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7,10, 0, 0, 0, 0, 4, 8, 0, 0, 0, 0, 0],
        [0, 0, 0,10,13,16,18, 0, 0, 0,11,14,17, 0, 0, 0, 0,12,15, 0, 0, 1, 5, 8, 0, 0, 0, 0, 2, 6, 9, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 7,10,12, 0, 0, 0, 0, 8,11, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, coupling in enumerate(j_couple_sdpfmu[:, parity_idx, proton_neutron_idx]):
        success = coupling.n_couplings == n_couplings_expected[i]
        msg = f"Incorrect number of couplings for coupling {i}."
        msg += f" Expected: {n_couplings_expected[i]}, got: {coupling.n_couplings}."
        assert success, msg

        for j, idx in enumerate(coupling.idx.flatten(order='F')):
            if j%2 != 0:
                """
                In the Fortran KSHELL code, the neutron orbitals are
                counted after the proton orbitals. This is why we must
                add the number of proton orbitals to the neutron
                orbitals which are every other orbital.

                In the Fortran code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: n_proton_orbitals, n_proton_orbitals + 1, ..., n_orbitals - 1

                In the Python code, the orbitals are counted:
                proton: 0, 1, 2, ..., n_proton_orbitals - 1
                neutron: 0, 1, 2, ..., n_neutron_orbitals - 1
                """
                idx += interaction_sdpfmu.proton_model_space.n_orbitals

            success = idx + 1 == idx_expected[i][j] # +1 because Fortran starts at 1.
            msg = f"Incorrect idx for coupling {i}, idx {j}."
            msg += f" Expected: {idx_expected[i][j]}, got: {idx}."
            assert success, msg

        for j, idx_reverse in enumerate(coupling.idx_reverse.flatten(order='F')):
            success = idx_reverse == idx_reverse_expected[i][j]
            msg = f"Incorrect idx_reverse for coupling {i}, idx_reverse {j}."
            msg += f" Expected: {idx_reverse_expected[i][j]}, got: {idx_reverse}."
            assert success, msg

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
    test_model_space_usda()
    test_proton_model_space_usda()
    test_neutron_model_space_usda()
    test_one_body_usda()
    test_two_body_usda()
    test_proton_j_couple_usda()
    test_neutron_j_couple_usda()
    test_proton_neutron_j_couple_usda()
    
    test_model_space_sdpfmu()
    test_proton_model_space_sdpfmu()
    test_neutron_model_space_sdpfmu()
    test_one_body_sdpfmu()
    test_two_body_sdpfmu()
    test_proton_j_couple_sdpfmu()
    test_neutron_j_couple_sdpfmu()
    test_proton_neutron_j_couple_sdpfmu()

    test_operator_j_reverse_indices_usda()
    test_operator_j_one_body()
    test_operator_j_two_body()