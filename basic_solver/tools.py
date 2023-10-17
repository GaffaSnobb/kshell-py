import time, sys, os
from functools import cache
from sympy.physics.quantum.cg import CG
from scipy.special import comb
import numpy as np
from numpy.typing import NDArray
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
    arr = np.zeros(
        shape = (6, 6+5, 6, 6+5, 11, 11+10),
        dtype = np.float64
    )
    print(arr.nbytes/1000/1000)
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
                                # arr[j1, m1, j2, m2, j3, m3] = float(cg_creation.doit())
                                # outfile.write(f"({j1:2}, {m1:2}, {j2:2}, {m2:2}, {j3:2}, {m3:2}): {float(cg_creation.doit())},\n")

    # np.save(file="CG_coefficients.npy", arr=arr, allow_pickle=True)

def generate_indices(interaction: Interaction) -> Indices:
    """
    Generate values for all the attributes of the Indices data
    structure.
    """
    timing = time.perf_counter()

    orbital_idx_to_j_map: list[int] = []
    orbital_idx_to_composite_m_idx_map: list[tuple[int, ...]] = []

    previous_degeneracy = 0
    for orbital in interaction.model_space.orbitals:
        
        orbital_idx_to_j_map.append(orbital.j)
        orbital_idx_to_composite_m_idx_map.append(tuple(range(
            previous_degeneracy,
            previous_degeneracy + orbital.degeneracy
        )))

        previous_degeneracy = orbital.degeneracy + previous_degeneracy

    creation_orb_indices_0: list[int] = []
    creation_orb_indices_1: list[int] = []
    annihilation_orb_indices_0: list[int] = []
    annihilation_orb_indices_1: list[int] = []

    for creation_orb_idx_0 in range(interaction.model_space.n_orbitals):
        for creation_orb_idx_1 in range(creation_orb_idx_0, interaction.model_space.n_orbitals):
            
            for annihilation_orb_idx_0 in range(interaction.model_space.n_orbitals):
                for annihilation_orb_idx_1 in range(annihilation_orb_idx_0, interaction.model_space.n_orbitals):
                    
                    creation_orb_indices_0.append(creation_orb_idx_0)
                    creation_orb_indices_1.append(creation_orb_idx_1)
                    annihilation_orb_indices_0.append(annihilation_orb_idx_0)
                    annihilation_orb_indices_1.append(annihilation_orb_idx_1)

    indices: Indices = Indices(
        composite_m_idx_to_m_map = interaction.model_space.all_jz_values,
        orbital_idx_to_j_map = orbital_idx_to_j_map,
        orbital_idx_to_composite_m_idx_map = orbital_idx_to_composite_m_idx_map,
        creation_orb_indices_0 = creation_orb_indices_0,
        creation_orb_indices_1 = creation_orb_indices_1,
        annihilation_orb_indices_0 = annihilation_orb_indices_0,
        annihilation_orb_indices_1 = annihilation_orb_indices_1,
    )
    timing = time.perf_counter() - timing
    timings.generate_indices.time = timing
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

if __name__ == "__main__":
    generate_clebsh_gordan_coefficients()