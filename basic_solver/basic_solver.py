import time
import numpy as np
from numpy.linalg import eigh
from kshell_utilities.loaders import load_interaction
from kshell_utilities.data_structures import Interaction
import kshell_utilities as ksutil
from parameters import flags
from data_structures import timings
from hamiltonian import create_hamiltonian


def main():
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 2
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
    
    timing = time.perf_counter()
    H = create_hamiltonian(
        interaction = interaction,
    )
    timing = time.perf_counter() - timing
    timings.main = timing

    assert np.all(np.abs(H - H.T) < 1e-15)
    eigvalues, eigfunctions = eigh(H)
    print(eigvalues)

    # O18 = ksutil.loadtxt(path="O18_w/", load_and_save_to_file=True)
    # print(O18.levels)
    # print()

    # if flags["debug"]:
    #     test_calculate_all_possible_orbital_occupations()

if __name__ == "__main__":
    flags["debug"] = True
    main()
    # print(timings)