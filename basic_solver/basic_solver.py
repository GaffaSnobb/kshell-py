import time, cProfile, pstats
import numpy as np
from numpy.linalg import eigh
from kshell_utilities.loaders import load_interaction
from kshell_utilities.data_structures import Interaction
import kshell_utilities as ksutil
from parameters import flags
from data_structures import timings
from hamiltonian import create_hamiltonian
from tools import HidePrint

from test_tools import test_generate_indices
from test_hamiltonian import (
    test_O18_eigenvalues, test_O19_eigenvalues, test_O20_eigenvalues
)
from test_basis import (
    test_calculate_O18_m_basis_states, test_calculate_F19_m_basis_states,
    test_calculate_O19_m_basis_states, test_calculate_O20_m_basis_states,
)

def summary(interaction: Interaction):
    print(f"interaction: {interaction.name}")
    print(f"n_protons: {interaction.model_space_proton.n_valence_nucleons}")
    print(f"n_neutrons: {interaction.model_space_neutron.n_valence_nucleons}")
    print(timings)

    if flags["debug"]:
        tests = [
            test_generate_indices,
            test_O18_eigenvalues,
            test_O19_eigenvalues,
            test_O20_eigenvalues,
            test_calculate_O18_m_basis_states,
            test_calculate_F19_m_basis_states,
            test_calculate_O19_m_basis_states,
            test_calculate_O20_m_basis_states,
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
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons

    # for key in interaction.tbme:
    #     o0, o1, o2, o3, j = key
    #     if (o0 != o2) and (o1 != o3):
    #         assert interaction.tbme.get(key, 0) == 0, interaction.tbme.get(key, 0)

    H = create_hamiltonian(
        interaction = interaction,
    )
    print(f"{H.nbytes = }")

    assert np.all(np.abs(H - H.T) < 1e-10)

    eigensolve_time = time.perf_counter()
    eigvalues, eigfunctions = eigh(H)
    eigensolve_time = time.perf_counter() - eigensolve_time
    print(f"{eigensolve_time = :.4f}s")
    
    timing = time.perf_counter() - timing
    timings.total.time = timing

    with HidePrint():
        O18 = ksutil.loadtxt(path="O19_w/", load_and_save_to_file=True)
    print("KSHELL, this")
    print("------------")
    for l1, l2 in zip(O18.levels[:, 0], eigvalues):
        print(f"{l1:8.3f}, {l2:10.5f}, {abs(l1 - l2) < 1e-3}")
    
    print()
    
    summary(interaction=interaction)

if __name__ == "__main__":
    # flags["debug"] = True
    main()
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.dump_stats('profile.prof')

    # stats = pstats.Stats('profile.prof')
    # stats.sort_stats('cumulative')
    # stats.print_stats()
    