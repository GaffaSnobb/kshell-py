import os, sys, time
from parameters import timing, flags
from data_structures import Partition, Interaction, ModelSpace
from loaders import read_partition_file
import numpy as np

try:
    """
    Added this try-except to make VSCode understand what is being
    imported.
    """
    from ..test.test_read_partition_file import test_partition_configuration_ordering
except ImportError:
    """
    Hacky way of using relative imports without installing as a package.
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'test'))
    from test_read_partition_file import test_partition_configuration_ordering

def initialise_partition(
    path: str,
    interaction: Interaction
):
    """
    Parameters
    ----------
    path : str
        Path to the partition file.

    interaction : Interaction
        Interaction object containing the interaction data.

    Returns
    -------
    partition : Partition
        Partition object containing the partition data.
    """
    partition: Partition = read_partition_file(path=path)

    if (partition.n_protons + partition.n_neutrons)%2 == 0:
        """
        There is an even number of nucleons in the nucleus. I think this
        is the m value of the basis states. NOTE: Improve this name when
        I understand what it is.
        """
        partition.m_tot = 0
    else:
        """
        There is an odd number of nucleons in the nucleus. I think this
        is the m value of the basis states. Remember that this value is
        multiplied by 2 to avoid fractions.
        """
        partition.m_tot = 1
    
    initialise_partition_time: float = time.perf_counter()
    test_partition_configuration_ordering(partition=partition)

    partition.proton_configurations_max_j[:] = calculate_max_j_per_configuration(
        configurations = partition.proton_configurations[:, 1:],    # Remove index column.
        j_orbitals = interaction.proton_model_space.j
    )
    partition.neutron_configurations_max_j[:] = calculate_max_j_per_configuration(
        configurations = partition.neutron_configurations[:, 1:],   # Remove index column.
        j_orbitals = interaction.neutron_model_space.j
    )
    partition.proton_configurations_parity[:] = calculate_configuration_parity(
        configurations = partition.proton_configurations[:, 1:],    # Remove index column.
        model_space = interaction.proton_model_space
    )
    partition.neutron_configurations_parity[:] = calculate_configuration_parity(
        configurations = partition.neutron_configurations[:, 1:],   # Remove index column.
        model_space = interaction.neutron_model_space
    )

    for i in range(partition.n_proton_configurations):
        """
        Initialise min_m and max_m for protons.
        """
        partition.proton_configurations_min_max[i, 0] = partition.proton_configurations_max_j[i]
        partition.proton_configurations_min_max[i, 1] = -partition.proton_configurations_max_j[i]

    for i in range(partition.n_neutron_configurations):
        """
        Initialise min_m and max_m for neutrons.
        """
        partition.neutron_configurations_min_max[i, 0] = partition.neutron_configurations_max_j[i]
        partition.neutron_configurations_min_max[i, 1] = -partition.neutron_configurations_max_j[i]

    partition.max_proton_neutron_couple_j = 0  # The maximum possible angular momentum of the system.
    for i in range(partition.n_proton_neutron_configurations):
        """
        Loop over proton-neutron configurations. NOTE: I do not know
        what the min_max values are for yet. If the final min value is
        always -max, then why bother storing / calcuating both?

        In this loop, the maximum possible angular momentum of all
        proton-neutron configurations is calculated.
        """
        proton_idx: int = partition.proton_neutron_configurations[i, 0]
        neutron_idx: int = partition.proton_neutron_configurations[i, 1]

        proton_max_j: int = partition.proton_configurations_max_j[proton_idx]
        neutron_max_j: int = partition.neutron_configurations_max_j[neutron_idx]

        partition.max_proton_neutron_couple_j = max(  # Find the combination of proton-neutron configurations with the largest angular momentum.
            proton_max_j + neutron_max_j, partition.max_proton_neutron_couple_j
        )
        partition.proton_configurations_min_max[proton_idx, 0] = min(
            partition.proton_configurations_min_max[proton_idx, 0],
            partition.m_tot - neutron_max_j
        )
        partition.neutron_configurations_min_max[neutron_idx, 0] = min(
            partition.neutron_configurations_min_max[neutron_idx, 0],
            partition.m_tot - proton_max_j
        )
        partition.proton_configurations_min_max[proton_idx, 1] = max(
            partition.proton_configurations_min_max[proton_idx, 1],
            partition.m_tot + neutron_max_j
        )
        partition.neutron_configurations_min_max[neutron_idx, 1] = max(
            partition.neutron_configurations_min_max[neutron_idx, 1],
            partition.m_tot + proton_max_j
        )

    print(interaction.proton_model_space.j)
    print(partition.proton_configurations[:, 1:])
    print(partition.proton_configurations_max_j)
    
    # for i in range(partition.n_proton_configurations):
    #     mi = max(-partition.proton_configurations_max_j[i], partition.proton_configurations_min_max[i, 0])
    #     mj = min( partition.proton_configurations_max_j[i], partition.proton_configurations_min_max[i, 1])

    #     print(f"{mi = }")
    #     print(f"{partition.proton_configurations_min_max[i] = }")


    initialise_partition_time = time.perf_counter() - initialise_partition_time
    timing.initialise_partition_time = initialise_partition_time
    if flags.debug:
        print(f"initialise_partition_time: {initialise_partition_time:.4f} s")

    return partition

def calculate_configuration_parity(
    configurations: np.ndarray,
    model_space: ModelSpace,
):  
    """
    Calculate the parity for each configuration.

    Parameters
    ----------
    configurations : np.ndarray
        Array of proton or neutron configurations.

    model_space : ModelSpace
        ModelSpace object containing the model space data for protons or
        neutrons.

    Returns
    -------
    parities : np.ndarray
        Array of parities for each configuration.
    """
    parities: np.ndarray = np.ones(configurations.shape[0], dtype=int)
    for configuration_idx in range(configurations.shape[0]):
        for orbital_idx in range(configurations.shape[1]):
            parities[configuration_idx] *= model_space.parity[orbital_idx]**configurations[configuration_idx, orbital_idx]

    return parities

def calculate_max_j_per_configuration(
    configurations: np.ndarray,
    j_orbitals: np.ndarray,
):
    """
    The maximum j value of each configuration will be calculated in this
    function. Ne20_usda_p.ptn example:

        # proton partition
            1     0  0  2
            2     0  1  1
            3     0  2  0
            4     1  0  1
            5     1  1  0
            6     2  0  0
    
    Configuration 1 has 2 protons in s1/2. s1/2 has a degeneracy of 2,
    meaning that the only possibility is jz = +- 1/2 and the max j value
    they can couple to is 0. Configuration 1 however, occupies d5/2 and
    s1/2 meaning that the max j value is 5/2 + 1/2 = 6/2. Configuration
    3, like configuration 1, has both protons in the same orbital, but
    since d5/2 has a degeneracy of 6 and since protons are fermions, the
    maximum j value they can couple to is 5/2 + 3/2 = 8/2. Etc. Remember
    that all j values are stored as 2*j to avoid fractions.

    These loops do the same work as max_m_nocc in the Fortran code:
    https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/partition.F90#L993-L1002
    
    Parameters
    ----------
    configurations : np.ndarray
        Array of proton or neutron configurations.

    j_orbitals : np.ndarray
        Array of total angular momentum of proton or neutron orbitals.

    Returns
    -------
    max_j : np.ndarray
        Array of maximum total angular momentum for each configuration.
    """
    max_j = np.zeros(configurations.shape[0], dtype=int)
    for configuration_idx in range(configurations.shape[0]):
        """
        Loop over all configurations.
        """
        for orbital_idx in range(configurations.shape[1]):
            """
            Loop over all orbitals in each configuration.
            """
            j_tmp: int = j_orbitals[orbital_idx]           # Angular momentum of the current orbital.
            occupation_tmp: int = configurations[configuration_idx, orbital_idx] # The number of protons (neutrons) which occupy the current orbital.
            max_j[configuration_idx] += (j_tmp - occupation_tmp + 1)*occupation_tmp

    return max_j