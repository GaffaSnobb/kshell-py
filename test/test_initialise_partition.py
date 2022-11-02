import sys, os
import numpy as np
try:
    """
    Added this try-except to make VSCode understand what is being
    imported.
    """
    from ..kshell_py.partition import initialise_partition
    from ..kshell_py.data_structures import Partition, Interaction
    from ..kshell_py.loaders import read_interaction_file
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kshell_py')) # Hacky way of using relative imports without installing as a package.
    from partition import initialise_partition
    from loaders import read_interaction_file

PARTITION_FILE_PATH_USDA: str = "../ptn/Ne20_usda_p.ptn"
INTERACTION_FILE_PATH_USDA: str = "../snt/usda.snt"
PARTITION_FILE_PATH_SDPFMU_POSITIVE: str = "../ptn/V50_sdpf-mu_p.ptn"
PARTITION_FILE_PATH_SDPFMU_NEGATIVE: str = "../ptn/V50_sdpf-mu_n.ptn"
INTERACTION_FILE_PATH_SDPFMU: str = "../snt/sdpf-mu.snt"

def test_max_j_value_per_configuration():
    interaction: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_USDA)
    partition: Partition = initialise_partition(
        path = PARTITION_FILE_PATH_USDA,
        interaction = interaction
    )

    proton_configurations_max_j_expected = [0, 6, 8, 4, 8, 4]
    for i in range(len(proton_configurations_max_j_expected)):
        success = partition.proton_configurations_max_j[i] == proton_configurations_max_j_expected[i]
        msg = f"The maximum j value for proton configuration {i} is not correct. Expected {proton_configurations_max_j_expected[i]}, got {partition.proton_configurations_max_j[i]}."
        assert success, msg

    neutron_configurations_max_j_expected = [0, 6, 8, 4, 8, 4]
    for i in range(len(neutron_configurations_max_j_expected)):
        success = partition.neutron_configurations_max_j[i] == neutron_configurations_max_j_expected[i]
        msg = f"The maximum j value for neutron configuration {i} is not correct. Expected {neutron_configurations_max_j_expected[i]}, got {partition.neutron_configurations_max_j[i]}."
        assert success, msg

def test_configuration_parity_usda():
    interaction: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_USDA)
    partition: Partition = initialise_partition(
        path = PARTITION_FILE_PATH_USDA,
        interaction = interaction
    )

    proton_configurations_parity_expected = [1, 1, 1, 1, 1, 1]
    neutron_configurations_parity_expected = [1, 1, 1, 1, 1, 1]

    success = len(partition.proton_configurations_parity) == len(proton_configurations_parity_expected)
    msg = f"The number of proton configuration parities is not correct. Expected {len(proton_configurations_parity_expected)}, got {len(partition.proton_configurations_parity)}."
    assert success, msg

    success = len(partition.neutron_configurations_parity) == len(neutron_configurations_parity_expected)
    msg = f"The number of neutron configuration parities is not correct. Expected {len(neutron_configurations_parity_expected)}, got {len(partition.neutron_configurations_parity)}."
    assert success, msg

    for i in range(len(proton_configurations_parity_expected)):
        success = partition.proton_configurations_parity[i] == proton_configurations_parity_expected[i]
        msg = f"The parity for proton configuration {i} is not correct. Expected {proton_configurations_parity_expected[i]}, got {partition.proton_configurations_parity[i]}."
        assert success, msg

    for i in range(len(neutron_configurations_parity_expected)):
        success = partition.neutron_configurations_parity[i] == neutron_configurations_parity_expected[i]
        msg = f"The parity for neutron configuration {i} is not correct. Expected {neutron_configurations_parity_expected[i]}, got {partition.neutron_configurations_parity[i]}."
        assert success, msg

def test_configuration_parity_sdpfmu_positive():
    interaction: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_SDPFMU)
    partition: Partition = initialise_partition(
        path = PARTITION_FILE_PATH_SDPFMU_POSITIVE,
        interaction = interaction
    )

    proton_configurations_parity_expected = [-1]*19
    neutron_configurations_parity_expected = [-1]*74

    success = len(partition.proton_configurations_parity) == len(proton_configurations_parity_expected)
    msg = f"The number of proton configuration parities is not correct. Expected {len(proton_configurations_parity_expected)}, got {len(partition.proton_configurations_parity)}."
    assert success, msg

    success = len(partition.neutron_configurations_parity) == len(neutron_configurations_parity_expected)
    msg = f"The number of neutron configuration parities is not correct. Expected {len(neutron_configurations_parity_expected)}, got {len(partition.neutron_configurations_parity)}."
    assert success, msg

    for i in range(len(proton_configurations_parity_expected)):
        success = partition.proton_configurations_parity[i] == proton_configurations_parity_expected[i]
        msg = f"The parity for proton configuration {i} is not correct. Expected {proton_configurations_parity_expected[i]}, got {partition.proton_configurations_parity[i]}."
        assert success, msg

    for i in range(len(neutron_configurations_parity_expected)):
        success = partition.neutron_configurations_parity[i] == neutron_configurations_parity_expected[i]
        msg = f"The parity for neutron configuration {i} is not correct. Expected {neutron_configurations_parity_expected[i]}, got {partition.neutron_configurations_parity[i]}."
        assert success, msg

def test_configuration_parity_sdpfmu_negative():
    interaction: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_SDPFMU)
    partition: Partition = initialise_partition(
        path = PARTITION_FILE_PATH_SDPFMU_NEGATIVE,
        interaction = interaction
    )

    proton_configurations_parity_expected = [1]*93 + [-1]*19
    neutron_configurations_parity_expected = [1]*258 + [-1]*74

    success = len(partition.proton_configurations_parity) == len(proton_configurations_parity_expected)
    msg = f"The number of proton configuration parities is not correct. Expected {len(proton_configurations_parity_expected)}, got {len(partition.proton_configurations_parity)}."
    assert success, msg

    success = len(partition.neutron_configurations_parity) == len(neutron_configurations_parity_expected)
    msg = f"The number of neutron configuration parities is not correct. Expected {len(neutron_configurations_parity_expected)}, got {len(partition.neutron_configurations_parity)}."
    assert success, msg

    for i in range(len(proton_configurations_parity_expected)):
        success = partition.proton_configurations_parity[i] == proton_configurations_parity_expected[i]
        msg = f"The parity for proton configuration {i} is not correct. Expected {proton_configurations_parity_expected[i]}, got {partition.proton_configurations_parity[i]}."
        assert success, msg

    for i in range(len(neutron_configurations_parity_expected)):
        success = partition.neutron_configurations_parity[i] == neutron_configurations_parity_expected[i]
        msg = f"The parity for neutron configuration {i} is not correct. Expected {neutron_configurations_parity_expected[i]}, got {partition.neutron_configurations_parity[i]}."
        assert success, msg

def test_jz():
    interaction: Interaction = read_interaction_file(path=INTERACTION_FILE_PATH_USDA)
    partition: Partition = initialise_partition(
        path = PARTITION_FILE_PATH_USDA,
        interaction = interaction
    )

    partition.proton_configurations_jz

    proton_configurations_jz_expected: np.ndarray = np.array(
        object = [
            np.arange(0, 0+1, 1),
            np.arange(-6, 6+1, 1),
            np.arange(-8, 8+1, 1),
            np.arange(-4, 4+1, 1),
            np.arange(-8, 8+1, 1),
            np.arange(-4, 4+1, 1)
        ],
        dtype = np.ndarray
    )

    neutron_configurations_jz_expected: np.ndarray = np.array(
        object = [
            np.arange(0, 0+1, 1),
            np.arange(-6, 6+1, 1),
            np.arange(-8, 8+1, 1),
            np.arange(-4, 4+1, 1),
            np.arange(-8, 8+1, 1),
            np.arange(-4, 4+1, 1)
        ],
        dtype = np.ndarray
    )

    for i in range(len(proton_configurations_jz_expected)):
        success = np.array_equal(partition.proton_configurations_jz[i], proton_configurations_jz_expected[i])
        msg = f"The jz for proton configuration {i} is not correct. Expected {proton_configurations_jz_expected[i]}, got {partition.proton_configurations_jz[i]}."
        assert success, msg

    for i in range(len(neutron_configurations_jz_expected)):
        success = np.array_equal(partition.neutron_configurations_jz[i], neutron_configurations_jz_expected[i])
        msg = f"The jz for neutron configuration {i} is not correct. Expected {neutron_configurations_jz_expected[i]}, got {partition.neutron_configurations_jz[i]}."
        assert success, msg

if __name__ == '__main__':
    test_max_j_value_per_configuration()
    test_configuration_parity_usda()
    test_configuration_parity_sdpfmu_positive()
    test_configuration_parity_sdpfmu_negative()
    test_jz()