import time, sys, os
from functools import cache
from sympy.physics.quantum.cg import CG
from scipy.special import comb
from data_structures import Indices
from kshell_utilities.data_structures import Interaction
from data_structures import timings

@cache
def n_choose_k(n, k):
    """
    NOTE: This can be replaced by a lookup table to increase speed
    further.
    """
    return comb(n, k, exact=True)

def generate_clebsh_gordan_coefficients():
    """
    Generate the CG coefficients for a range of angular momenta and
    write the values to a file so that they may be copied into a Python
    dict, like in parameters.py. I use a dict lookup table solution
    instead of using sympy.physics.quantum.cg because the doit()
    function makes Pylance suuper slow.

    Note that all angular momenta values are saved to file as two times
    its value to avoid fractions.
    """
    cg_creation = CG(
        j1 = 3/2,
        m1 = 1/2,
        j2 = 3/2,
        m2 = -1/2,
        j3 = 0,
        m3 = 0,
    )
    m_vals = {
        3: [-3, -1, 1, 3],
        5: [-5, -3, -1, 1, 3, 5],
        1: [-1, 1],
    }
    with open("cg.txt", "w") as outfile:
        for j1 in [3, 5, 1]:
            for m1 in m_vals[j1]:

                for j2 in [3, 5, 1]:
                    for j3 in range(abs(j1 - j2), j1 + j2 + 2, 2):
                        for m3 in range(-j3, j3 + 2, 2):
                            for m2 in m_vals[j2]:
                                cg_creation = CG(
                                    j1 = j1/2,
                                    m1 = m1/2,
                                    j2 = j2/2,
                                    m2 = m2/2,
                                    j3 = j3/2,
                                    m3 = m3/2,
                                )
                                # outfile.write(f"({j1:2}, {m1:2}, {j2:2}, {m2:2}, {j3:2}, {m3:2}): {float(cg_creation.doit())},\n")

def generate_indices(interaction: Interaction) -> Indices:
    """
    Generate values for all the attributes of the Indices data
    structure.
    """
    timing = time.perf_counter()
    indices: Indices = Indices()
    indices.m_composite_idx_to_m_map = interaction.model_space_neutron.all_jz_values

    m_composite_idx_counter = 0
    for orb_idx, orbital in enumerate(interaction.model_space_neutron.orbitals):
        indices.orbital_idx_to_m_idx_map.append(tuple(range(orbital.degeneracy)))
        indices.orbital_idx_to_j_map.append(orbital.j)

        for m_idx in range(orbital.degeneracy):
            indices.orbital_m_pair_to_composite_m_idx_map[(orb_idx, m_idx)] = m_composite_idx_counter
            m_composite_idx_counter += 1

    timing = time.perf_counter() - timing
    timings.generate_indices_001 = timing
    return indices

class HidePrint:
    """
    Simple class for hiding prints to stdout when running unit tests.
    From: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    Usage:
    ```
    with HidePrint():
        # Code here will not show any prints.

    # Code here will show prints.
    ```
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout