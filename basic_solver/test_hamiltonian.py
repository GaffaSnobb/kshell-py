import numpy as np
from numpy.linalg import eigh
# import kshell_utilities as ksutil
from kshell_utilities.data_structures import Interaction
from kshell_utilities.loaders import load_interaction
from hamiltonian import create_hamiltonian
from tools import HidePrint

EIG_TOL = 1e-3
MAT_TOL = 1e-10

def test_O18_eigenvalues():
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 2
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons

    expected_eigenvalues = [    # From KSHELL.
        -12.171, -9.991, -8.389, -7.851, -7.732,
    ]
    with HidePrint():
        """
        Hide all prints in the code.
        """
        H = create_hamiltonian(
            interaction = interaction,
        )
    
    assert np.all(np.abs(H - H.T) < MAT_TOL)
    calculated_eigenvalues, _ = eigh(H)

    for expected, calculated in zip(expected_eigenvalues, calculated_eigenvalues):
        assert np.abs(expected - calculated) < EIG_TOL, "18O eigenvalues are not within the tolerance!"

def test_O19_eigenvalues():
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 3
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons

    expected_eigenvalues = [    # From KSHELL.
        -16.064, -15.77, -14.594, -13.585, -13.103, -12.895, -12.317, -11.053,
        -10.889, -10.535,
    ]
    with HidePrint():
        """
        Hide all prints in the code.
        """
        H = create_hamiltonian(
            interaction = interaction,
        )
    
    assert np.all(np.abs(H - H.T) < MAT_TOL)
    calculated_eigenvalues, _ = eigh(H)

    for expected, calculated in zip(expected_eigenvalues, calculated_eigenvalues):
        assert np.abs(expected - calculated) < EIG_TOL, "19O eigenvalues are not within the tolerance!"

def test_O20_eigenvalues():
    interaction: Interaction = load_interaction(filename_interaction="../snt/w.snt")
    n_valence_protons = 0
    n_valence_neutrons = 4
    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons

    expected_eigenvalues = [    # From KSHELL.
        -23.827, -21.864, -20.056, -19.653, -18.788, -18.538, -18.495, -18.381,
        -18.323, -16.452,
    ]
    with HidePrint():
        """
        Hide all prints in the code.
        """
        H = create_hamiltonian(
            interaction = interaction,
        )
    
    assert np.all(np.abs(H - H.T) < MAT_TOL)
    calculated_eigenvalues, _ = eigh(H)

    for expected, calculated in zip(expected_eigenvalues, calculated_eigenvalues):
        assert np.abs(expected - calculated) < EIG_TOL, "20O eigenvalues are not within the tolerance!"

if __name__ == "__main__":
    test_O18_eigenvalues()
    test_O19_eigenvalues()
    test_O20_eigenvalues()

    # O18 = ksutil.loadtxt(path="O20_w/", load_and_save_to_file=True)

    # for e in O18.levels[:, 0]:
    #     print(f"{e}, ", end="")