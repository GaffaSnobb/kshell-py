import time
from kshell_utilities.data_structures import (
    OrbitalParameters, Interaction
)
from data_structures import timings
from parameters import flags

def fill_orbitals(
    orbitals: list[OrbitalParameters],
    orbital_occupations: list[tuple[int]],
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
        orbital_occupations.append(tuple(current_orbital_occupation))    # tuple conversion was marginally faster than list.copy().
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
        
        fill_orbitals(
            orbitals = orbitals[1:],
            n_remaining_neutrons = n_remaining_neutrons - occupation,
            n_remaining_holes = n_remaining_holes - current_orbital.degeneracy,
            current_orbital_idx = current_orbital_idx + 1,
            orbital_occupations = orbital_occupations,
            current_orbital_occupation = current_orbital_occupation,
        )
        current_orbital_occupation[current_orbital_idx] -= occupation

    timing = time.perf_counter() - timing
    timings.fill_orbitals = timing  # The initial call of this function will overwrite all the timings from the recursive calls, giving the total time.

def calculate_all_possible_orbital_occupations(
    interaction: Interaction,
) -> list[tuple[int]]:
    """
    Given a number of valence nucleons and a model space, calculate the
    different possible orbital occupations. This is the same as the
    proton / neutron partition in the KSHELL .ptn files.
    
    Parameters
    ----------
    interaction:
        Custom dataclass containing all necessary parameters of the
        given interaction.

    Returns
    -------
    orbital_occupations:
        A list containing each allowed orbital occupation. Example:
            
            [(0, 1, 2), (0, 2, 1), ...]

        where the first allowed occupation tells us that the first
        orbital has zero nucleons, the second orbital has 1 nucleon, and
        the third orbital has 2 nucleons. The order of the orbitals is
        the same as the order they are listed in the interaction file.
    """
    timing = time.perf_counter()
    current_orbital_occupation: list[int] = [0]*interaction.model_space_neutron.n_orbitals
    orbital_occupations: list[tuple[int]] = []

    fill_orbitals(
        orbitals = interaction.model_space_neutron.orbitals,
        n_remaining_neutrons = interaction.model_space_neutron.n_valence_nucleons,
        n_remaining_holes = sum([orb.degeneracy for orb in interaction.model_space_neutron.orbitals]),
        current_orbital_idx = 0,
        orbital_occupations = orbital_occupations,
        current_orbital_occupation = current_orbital_occupation,
    )

    if flags["debug"]:
        """
        Should already be sorted lexicographically from the way the
        orbitals are traversed.
        """
        assert sorted(orbital_occupations) == orbital_occupations
        assert all(sum(occupation) == interaction.model_space_neutron.n_valence_nucleons for occupation in orbital_occupations)
    
    timing = time.perf_counter() - timing
    timings.calculate_all_possible_orbital_occupations = timing - timings.fill_orbitals

    return orbital_occupations