from typing import Any
import numpy as np
import kshell_utilities as ksutil
from kshell_utilities.data_structures import Interaction
from kshell_utilities.loaders import load_interaction

def O18_w_manual_hamiltonian() -> np.ndarray[Any, float]:
    """
    Manual construction of the shell model hamiltonian for 18O (2
    valence neutrons) with the w interaction.

    TODO: I made this function thinking there were two valence protons
    while it it actually valence neutrons. It still works because the
    interaction and model space treats the protons and neutrons the same
    in this case, but it should be changed so that it is correct.
    """
    O18 = ksutil.loadtxt(path="O18_w/")
    interaction: Interaction = load_interaction(filename_interaction="O18_w/w.snt")
    spe = interaction.spe

    def tbme(a, b, c, d, j):
        return interaction.tbme.get((a, b, c, d, j), 0)
        return interaction.tbme[(a, b, c, d, j)]

    H = np.array([
        #              d3/2                          d5/2^2                           s1/2^2                           d3/2 d5/2                        d3/2 s1/2                        d5/2 s1/2
        [2*spe[0] + tbme(0, 0, 0, 0, 0),            tbme(0, 0, 1, 1, 0),            tbme(0, 0, 2, 2, 0),                   tbme(0, 0, 0, 1, 0),                   tbme(0, 0, 0, 2, 0),                   tbme(0, 0, 1, 2, 0)], # d3/2^2
        [           tbme(1, 1, 0, 0, 0), 2*spe[1] + tbme(1, 1, 1, 1, 0),            tbme(1, 1, 2, 2, 0),                   tbme(1, 1, 0, 1, 0),                   tbme(1, 1, 0, 2, 0),                   tbme(1, 1, 1, 2, 0)], # d5/2^2
        [           tbme(2, 2, 0, 0, 0),            tbme(2, 2, 1, 1, 0), 2*spe[2] + tbme(2, 2, 2, 2, 0),                   tbme(2, 2, 0, 1, 0),                   tbme(2, 2, 0, 2, 0),                   tbme(2, 2, 1, 2, 0)], # s1/2^2
        [           tbme(0, 1, 0, 0, 0),            tbme(0, 1, 1, 1, 0),            tbme(0, 1, 2, 2, 0), spe[0] + spe[1] + tbme(0, 1, 0, 1, 0),                   tbme(0, 1, 0, 2, 0),                   tbme(0, 1, 1, 2, 0)], # d3/2 d5/2
        [           tbme(0, 2, 0, 0, 0),            tbme(0, 2, 1, 1, 0),            tbme(0, 2, 2, 2, 0),                   tbme(0, 2, 0, 1, 0), spe[0] + spe[2] + tbme(0, 2, 0, 2, 0),                   tbme(0, 2, 1, 2, 0)], # d3/2 s1/2
        [           tbme(1, 2, 0, 0, 0),            tbme(1, 2, 1, 1, 0),            tbme(1, 2, 2, 2, 0),                   tbme(1, 2, 0, 1, 0),                   tbme(1, 2, 0, 2, 0), spe[1] + spe[2] + tbme(1, 2, 1, 2, 0)], # d5/2 s1/2
    ])

    H_alt = np.array([
        #              (0, 0, 2)                            (0, 1, 1)                            (0, 2, 0)                            (1, 0, 1)                            (1, 1, 0)                            (2, 0, 0)
        [2*spe[2] + tbme(2, 2, 2, 2, 0),                   tbme(2, 2, 1, 2, 0),            tbme(2, 2, 1, 1, 0),                   tbme(2, 2, 0, 2, 0),                   tbme(2, 2, 0, 1, 0),                   tbme(2, 2, 0, 0, 0)],
        [           tbme(1, 2, 2, 2, 0), spe[1] + spe[2] + tbme(1, 2, 1, 2, 0),            tbme(1, 2, 1, 1, 0),                   tbme(1, 2, 0, 2, 0),                   tbme(1, 2, 0, 1, 0),                   tbme(1, 2, 0, 0, 0)],
        [           tbme(1, 1, 2, 2, 0),                   tbme(1, 1, 1, 2, 0), 2*spe[1] + tbme(1, 1, 1, 1, 0),                   tbme(1, 1, 0, 2, 0),                   tbme(1, 1, 0, 1, 0),                   tbme(1, 1, 0, 0, 0)],
        [           tbme(0, 2, 2, 2, 0),                   tbme(0, 2, 1, 2, 0),            tbme(0, 2, 1, 1, 0), spe[0] + spe[2] + tbme(0, 2, 0, 2, 0),                   tbme(0, 2, 0, 1, 0),                   tbme(0, 2, 0, 0, 0)],
        [           tbme(0, 1, 2, 2, 0),                   tbme(0, 1, 1, 2, 0),            tbme(0, 1, 1, 1, 0),                   tbme(0, 1, 0, 2, 0), spe[0] + spe[1] + tbme(0, 1, 0, 1, 0),                   tbme(0, 1, 0, 0, 0)],
        [           tbme(0, 0, 2, 2, 0),                   tbme(0, 0, 1, 2, 0),            tbme(0, 0, 1, 1, 0),                   tbme(0, 0, 0, 2, 0),                   tbme(0, 0, 0, 1, 0),        2*spe[0] + tbme(0, 0, 0, 0, 0)],
    ])
    
    # print(H)
    # print()
    # print(H_alt)
    assert np.all(H == H.T)
    assert np.all(H_alt == H_alt.T)

    return H_alt

def O19_w_manual_hamiltonian() -> np.ndarray[Any, float]:
    O19 = ksutil.loadtxt(path="O19_w/")
    interaction: Interaction = load_interaction(filename_interaction="O19_w/w.snt")
    spe = interaction.spe

    def tbme(a, b, c, d, j):
        return interaction.tbme.get((a, b, c, d, j), 0)

    H = np.array([
        #              d3/2                          d5/2^2                           s1/2^2                           d3/2 d5/2                        d3/2 s1/2                        d5/2 s1/2
        [2*spe[0] + tbme(0, 0, 0, 0, 0),            tbme(0, 0, 1, 1, 0),            tbme(0, 0, 2, 2, 0),                   tbme(0, 0, 0, 1, 0),                   tbme(0, 0, 0, 2, 0),                   tbme(0, 0, 1, 2, 0)],
    ])

    assert np.all(H == H.T)

    return H