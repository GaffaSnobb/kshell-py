import time, math
from kshell_utilities.data_structures import (
    OrbitalParameters, Interaction
)
from data_structures import timings, Partition, Configuration
from parameters import flags

def _fill_orbitals(
    orbitals: list[OrbitalParameters],
    partition: Partition,
    interaction: Interaction,
    current_orbital_occupation: list[int],
    n_remaining_neutrons: int,
    n_remaining_holes: int,
    current_orbital_idx: int,
):
    """
    Fill all the orbitals in the given model space with all possible
    combinations of occupations. Account for the orbitals' degeneracies.

    To account for a variable number of orbitals, this function will
    call itself recursively and for each call proceeding to the next
    orbital in the model space. For each call, the function loops over
    all possible numbers of occupation, accounting for the current
    orbital degeneracy and the number of remaining nucleons.

    After all recursive calls have been completed, this function will
    have produced the same content as the partition files from KSHELL.

    Example
    -------
    Assume a model space of orbitals = [d5/2, d3/2].

    for occupation in [0, ..., d5/2 max allowed occupation]:
        store the current d5/2 occupation
    --- call fill_orbitals, excluding d5/2 in the list 'orbitals'
    |
    |
    --> for occupation in [0, ..., d3/2 max allowed occupation]:
        store the current d3/2 occupation
    --- call fill_orbitals, excluding d3/2 in the list 'orbitals'
    |
    |
    --> The function returns early because there are no orbitals left.
        If there are no nucleons left, 'current_orbital_occupation' is
        saved to 'orbital_occupations'. If there are nucleons left, the
        current orbital occupation is not stored and the occupation
        iteration in the previous recursive call is continued.

    Parameters
    ----------
    orbitals:
        List of the remaining orbitals in the model space.
        OrbitalParameters contains various parameters for the orbitals.

    orbital_occupations:
        A list for storing the valid configurations.

    current_orbital_occupation:
        Storage for the current occupation. Will be copied to
        orbital_occupations if it is valid.

    n_remaining_neutrons:
        The remaining number of nucleons to place into the remaining
        orbitals.
    
    n_remaining_holes:
        The remaining free places to put the remaining nucleons.
    
    current_orbital_idx:
        The index of the current orbital for the
        current_orbital_occupation list. Is incremented +1 for every
        recursive function call.
    """
    timing = time.perf_counter()
    if n_remaining_neutrons == 0:
        """
        No more neutrons to place, aka a complete configuration.
        """
        current_orbital_parity: int = 1
        for orbital, occupation in zip(interaction.model_space_neutron.orbitals, current_orbital_occupation):
            current_orbital_parity *= orbital.parity**(occupation)

        partition.configurations.append(
            Configuration(
                configuration = tuple(current_orbital_occupation),  # tuple conversion was marginally faster than list.copy().
                parity = current_orbital_parity,
            )
        )
        return

    if not orbitals:
        """
        No remaining orbitals but there are remaining neutrons, aka
        incomplete configuration.
        """
        return
    
    if n_remaining_neutrons > n_remaining_holes:
        """
        Not enough holes for the remaining neutrons, aka incomplete
        configuration.
        """
        return
    
    current_orbital = orbitals[0]

    for occupation in range(0, min(current_orbital.degeneracy, n_remaining_neutrons) + 1):
        current_orbital_occupation[current_orbital_idx] += occupation
        
        _fill_orbitals(
            orbitals = orbitals[1:],
            n_remaining_neutrons = n_remaining_neutrons - occupation,
            n_remaining_holes = n_remaining_holes - current_orbital.degeneracy,
            current_orbital_idx = current_orbital_idx + 1,
            partition = partition,
            interaction = interaction,
            current_orbital_occupation = current_orbital_occupation,
        )
        current_orbital_occupation[current_orbital_idx] -= occupation

    timing = time.perf_counter() - timing
    timings.fill_orbitals = timing  # The initial call of this function will overwrite all the timings from the recursive calls, giving the total time.

def calculate_all_possible_orbital_occupations(
    interaction: Interaction,
) -> Partition:
    """
    Given a number of valence nucleons and a model space, calculate the
    different possible orbital occupations. This is the same as the
    proton / neutron partition in the KSHELL .ptn files.

    Aka. generate the partition file!
    
    Parameters
    ----------
    interaction:
        Custom dataclass containing all necessary parameters of the
        given interaction.

    Returns
    -------
    partition : Partition

    M_target:
        The target value of the sum of the total angular momentum z
        projections of the nucleons in the pair. `M = m0 + m1`. A pair
        will be kept only if the sum of the z projections of the two
        nucleons in the pair is equal to this value.
    """
    timing = time.perf_counter()
    current_orbital_occupation: list[int] = [0]*interaction.model_space_neutron.n_orbitals
    partition: Partition = Partition()

    _fill_orbitals(
        orbitals = interaction.model_space_neutron.orbitals,
        n_remaining_neutrons = interaction.model_space_neutron.n_valence_nucleons,
        n_remaining_holes = sum([orb.degeneracy for orb in interaction.model_space_neutron.orbitals]),
        current_orbital_idx = 0,
        partition = partition,
        interaction = interaction,
        current_orbital_occupation = current_orbital_occupation,
    )
    if flags["debug"]:
        """
        Should already be sorted lexicographically from the way the
        orbitals are traversed.
        """
        assert sorted(partition.configurations) == partition.configurations
        assert all(sum(configuration.configuration) == interaction.model_space_neutron.n_valence_nucleons for configuration in partition.configurations)
    
    timing = time.perf_counter() - timing
    timings.calculate_all_possible_orbital_occupations = timing - timings.fill_orbitals

    return partition

def calculate_all_possible_pairs(
    configuration: tuple[int],
    M_target: int,
) -> list[tuple[int, int]]:
    """
    Calculate all the possible choices of two nucleons from some
    configuration of nucleons in orbitals.

    TODO: @cache this?

    Example
    -------
    ```
    >>> calculate_all_possible_pairs((1, 1, 1))
    [(0, 1), (0, 2), (1, 2)]
    ```
    Parameters
    ----------
    configuration:
        A list containing a possible configuration of the valence
        nucleons in the orbitals of the model space. Represented as
        indices of the orbitals. Ex.:

            (1, 1, 1), (2, 1, 0)

        which means that the first configuration has one nucleon in each
        of the three orbitals of the model space, while the sencond
        configuration has two nucleons in the first orbital, one nucleon
        in the second orbital and no nucleons in the third orbital.

    M_target:
        The target value of the sum of the total angular momentum z
        projections of the nucleons in the pair. `M = m0 + m1`. A pair
        will be kept only if the sum of the z projections of the two
        nucleons in the pair is equal to this value.
    """
    timing = time.perf_counter()
    j_orbital_values: list[int] = [3, 5, 1] # Hard-coded for d3/2, d5/2, s1/2.
    configuration_pair_permutation_indices: list[tuple[int, int]] = []
    n_occupations = len(configuration)

    for idx in range(n_occupations):
        """
        Deal with all the pairs within the same orbital.
        """
        occ = configuration[idx]
        if occ == 0: continue

        configuration_pair_permutation_indices.extend(
            [(idx, idx)]*math.comb(occ, 2)
        )
    
    for idx_1 in range(n_occupations):
        occ_1 = configuration[idx_1]
        if occ_1 == 0: continue
        
        j_1 = j_orbital_values[idx_1]

        for idx_2 in range(idx_1 + 1, n_occupations):
            occ_2 = configuration[idx_2]
            if occ_2 == 0: continue

            j_2 = j_orbital_values[idx_2]

            if idx_1 == idx_2:
                """
                This if is needed for the cases where the
                configuration occupies only a single orbital.
                """
                configuration_pair_permutation_indices.extend(
                    [(idx_1, idx_2)]*math.comb(occ_1, 2)
                )
            
            else:
                configuration_pair_permutation_indices.extend(
                    [(idx_1, idx_2)]*occ_1*occ_2
                )

    configuration_pair_permutation_indices.sort()
    timing = time.perf_counter() - timing
    timings.calculate_all_possible_pairs += timing

    return configuration_pair_permutation_indices