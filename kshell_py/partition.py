import os, sys, time
from parameters import timing, flags
from data_structures import Partition, Interaction, ModelSpace, BitRepresentation
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

    partition.max_proton_neutron_couple_j = 0  # The maximum possible angular momentum of the system ('max_jj' in the Fortran code).
    for i in range(partition.n_proton_neutron_configurations):
        """
        Loop over proton-neutron configurations and calculate the
        maximum possible angular momentum of all proton-neutron
        configurations. In the Fortran code:
        https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/partition.F90#L158
        """
        proton_idx: int = partition.proton_neutron_configurations[i, 0]
        neutron_idx: int = partition.proton_neutron_configurations[i, 1]

        proton_max_j: int = partition.proton_configurations_max_j[proton_idx]
        neutron_max_j: int = partition.neutron_configurations_max_j[neutron_idx]

        partition.max_proton_neutron_couple_j = max(  # Find the combination of proton-neutron configurations with the largest angular momentum.
            proton_max_j + neutron_max_j, partition.max_proton_neutron_couple_j
        )

    for i in range(partition.n_proton_configurations):
        """
        Loop over proton configurations and calculate the possible jz
        values for each configuration. These loops replace the code
        https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/partition.F90#L159-L170
        and
        https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/partition.F90#L180-L184
        """
        partition.proton_configurations_jz[i] = np.arange(
            start = -partition.proton_configurations_max_j[i],
            stop = partition.proton_configurations_max_j[i] + 1,    # +1 to include the last value.
            step = 1
        )

    for i in range(partition.n_neutron_configurations):
        """
        Loop over neutron configurations and calculate the possible jz
        values for each configuration.
        """
        partition.neutron_configurations_jz[i] = np.arange(
            start = -partition.neutron_configurations_max_j[i],
            stop = partition.neutron_configurations_max_j[i] + 1,   # +1 to include the last value.
            step = 1
        )

    initialise_m_scheme_bit_representation(
        interaction = interaction,
        partition = partition,
    )

    initialise_partition_time = time.perf_counter() - initialise_partition_time
    timing.initialise_partition_time = initialise_partition_time
    if flags.debug:
        print(f"initialise_partition_time: {initialise_partition_time:.4f} s")

    return partition

def initialise_m_scheme_bit_representation(
    interaction: Interaction,
    partition: Partition
):
    """
    type type_m_mbit 
        integer :: n 
        integer(kmbit), allocatable :: mbit(:)
        integer, allocatable :: mm(:)
    end type type_m_mbit
    """
    initialise_m_scheme_bit_representation_time: float = time.perf_counter()

    mbit_orb: np.ndarray = np.zeros(
        shape = (
            max(interaction.model_space.n_proton_orbitals, interaction.model_space.n_neutron_orbitals),
            max(partition.n_protons, partition.n_neutrons) + 1,
            2   # 0 is proton, 1 is neutron.
        ),
        dtype = BitRepresentation
    )

    for i in range(mbit_orb.shape[0]):
        """
        Initialise all elements in mbit_orb.
        """
        for j in range(mbit_orb.shape[1]):
            for k in range(mbit_orb.shape[2]):
                mbit_orb[i, j, k] = BitRepresentation(n=0)

    for loop in range(2):
        """
        get rid of this stupid loop
        """

        for i in range(mbit_orb.shape[0]):
            for j in range(mbit_orb.shape[1]):
                for k in range(mbit_orb.shape[2]):
                    mbit_orb[i, j, k].n = 0

        for orbital_idx in range(interaction.proton_model_space.n_orbitals):
            for mm in range(2**(interaction.proton_model_space.j[orbital_idx] + 1) - 1 + 1):
                """
                Calculate the population count (popcnt, bit count) of mm. If
                the popcnt of mm is <= the number of valence protons
                (neutrons) then the current [orbital, popcnt, ipn] is
                counted (n += 1).
                
                Example (Ne20_usda):
                For angular momentum 3, we have mm and corresponding popcnt:
                
                mm     = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
                popcnt = 0, 1, 1, 2, 1, 2, 2, 3, 1, 2,  2,  3,  2,  3,  3,  4

                For Ne20_usda we have 2 valence protons meaning that we are
                left with

                mm     = 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12
                popcnt = 0, 1, 1, 2, 1, 2, 2, 1, 2,  2,  2
                mm_removed = 7, 11, 13, 14, 15

                and consequently, n is incremented 11 times (not at the same
                index every time, mind you).
                
                """
                # print(f"{mm = }")
                popcnt = mm.bit_count()  # Count the number of bits which are set to 1.

                if popcnt > partition.n_protons:
                    """
                    When the number of bits set to 1 in mm is greater than
                    the number of valence protons... Why should this happen?

                    Well for starters, this avoids IndexError in mbit_orb a
                    few lines down from here.
                    """
                    continue

                mbit_orb[orbital_idx, popcnt, 0].n += 1
                
                if loop == 1:
                    n_tmp: int = mbit_orb[orbital_idx, popcnt, 0].n - 1  # Think I have to subtract 1 to get 0-based indexing.
                    if n_tmp == -1:
                        msg = "n_tmp should not be -1!"
                        raise RuntimeError(msg)

                    try:
                        mbit_orb[orbital_idx, popcnt, 0].mbit[n_tmp] = mm
                    except IndexError as e:
                        print(f"{orbital_idx = }")
                        print(f"{popcnt = }")
                        print(f"{mm = }")
                        print(f"{n_tmp = }")
                        raise(e)

        if loop == 1: continue
        
        for orbital_idx in range(interaction.proton_model_space.n_orbitals):
            for nucleon_idx in range(partition.n_protons + 1):
                
                n_tmp: int = mbit_orb[orbital_idx, nucleon_idx, 0].n
                if n_tmp == 0:
                    msg = "n_tmp should not be zero!"
                    raise RuntimeError(msg)
                
                mbit_orb[orbital_idx, nucleon_idx, 0].mbit = np.zeros(
                    shape = n_tmp,
                    dtype = int
                )
                mbit_orb[orbital_idx, nucleon_idx, 0].mm = np.zeros(
                    shape = n_tmp,
                    dtype = int
                )
    
    for i in range(mbit_orb.shape[0]):
        for j in range(mbit_orb.shape[1]):
            for k in range(mbit_orb.shape[2]):
                print(mbit_orb[i, j, k].n)

    initialise_m_scheme_bit_representation_time = time.perf_counter() - initialise_m_scheme_bit_representation_time
    timing.initialise_m_scheme_bit_representation_time = initialise_m_scheme_bit_representation_time
    if flags.debug:
        print(f"initialise_m_scheme_bit_representation_time: {initialise_m_scheme_bit_representation_time:.4f} s")


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