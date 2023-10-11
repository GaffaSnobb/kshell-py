import time
import numpy as np
from numpy.linalg import eigh
from kshell_utilities.loaders import load_interaction
from kshell_utilities.data_structures import Interaction
import kshell_utilities as ksutil
from parameters import flags
from data_structures import timings
from hamiltonian import create_hamiltonian

from test_tools import test_generate_indices
from test_hamiltonian import (
    test_O18_eigenvalues, test_O19_eigenvalues, test_O20_eigenvalues
)

def summary(interaction: Interaction):
    print(f"interaction: {interaction.name}")
    print(f"n_protons: {interaction.model_space_proton.n_valence_nucleons}")
    print(f"n_neutrons: {interaction.model_space_neutron.n_valence_nucleons}")
    print(timings)

    if flags["debug"]:
        tests = [
            test_generate_indices, test_O18_eigenvalues, test_O19_eigenvalues,
            test_O20_eigenvalues,
        ]
        print("Running tests...")
        for test in tests:
            test()
            print(f"    {test.__name__} passed!")

        print("All tests passed!")
        print()


def main():
    timing = time.perf_counter()
    
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 4
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons

    H = create_hamiltonian(
        interaction = interaction,
    )
    print(f"{H.nbytes = }")

    timing = time.perf_counter() - timing
    timings.main = timing

    assert np.all(np.abs(H - H.T) < 1e-10)
    eigvalues, eigfunctions = eigh(H)
    print(eigvalues)

    O18 = ksutil.loadtxt(path="O20_w/", load_and_save_to_file=True)
    print(O18.levels)
    print()
    summary(interaction=interaction)

if __name__ == "__main__":
    flags["debug"] = True
    main()
    