from math import sqrt
import kshell_utilities as ksutil
import numpy as np
from loaders import load_interaction
from data_structures import Interaction
from parameters import clebsch_gordan
import kshell_utilities.loaders
import kshell_utilities.data_structures

def main():
    """
             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2
    """
    interaction: Interaction = load_interaction(path="../snt/w/")
    # interaction: kshell_utilities.data_structures.Interaction = kshell_utilities.loaders.load_interaction(filename_interaction="../snt/w.snt")
    basis_states = (
        (0, 3), # c_0 c_3 | core > = - c_3 c_0 | core >
        (1, 2), # c_1 c_2 | core >
    )
    H = np.zeros((2, 2))
    # SPE term 1:

    for idx_left in range(2):
        for idx_right in range(2):
            
            
            for ma in [0, 1, 2, 3]:
                for mb in [0, 1, 2, 3]:
                    new_left_state = list(basis_states[idx_left])
                    new_right_state = list(basis_states[idx_right])
                    
                    try:
                        annihilation_idx = new_right_state.index(mb)
                    except ValueError:
                        continue
                    
                    annihilation_sign = (-1)**annihilation_idx
                    new_right_state.pop(annihilation_idx)

                    if ma in new_right_state: continue

                    new_right_state.insert(0, ma)

                    if new_left_state == new_right_state:
                        creation_sign = 1
                    elif new_left_state == new_right_state[::-1]:
                        creation_sign = -1
                    else:
                        continue

                    H[idx_left, idx_right] += annihilation_sign*creation_sign*interaction.spe[0]

    m_idx_to_m = {
        0: -3,
        1: -1,
        2: +1,
        3: +3,
    }
    for idx_left, left_state in enumerate(basis_states):
        for idx_right, right_state in enumerate(basis_states):

            for J in [0, 2]:#, 4, 6]:
                for M in [0]:#range(-J, J + 2, 2):
                    
                    try:
                        V = interaction.tbme[(0, 0, 0, 0, J)]
                        # print((0, 0, 0, 0, J), V)
                    except KeyError:
                        continue
                    
                    # Annihilation term:
                    annihilation_norm = 1/sqrt(2)
                    
                    annihilation_results = []
                    
                    for mc in [0, 1, 2, 3]:
                        for md in [0, 1, 2, 3]:
                            new_left_state = list(left_state)
                            new_right_state = list(right_state)

                            if mc not in new_right_state: continue

                            annihilation_idx = new_right_state.index(mc)
                            new_right_state.pop(annihilation_idx)
                            annihilation_sign = (-1)**annihilation_idx

                            if md not in new_right_state: continue

                            new_right_state.remove(md)

                            assert not new_right_state, new_right_state

                            cg_annihilation = clebsch_gordan[(
                                3,
                                m_idx_to_m[mc],
                                3,
                                m_idx_to_m[md],
                                J,
                                M,
                            )]

                            if cg_annihilation == 0: continue

                            annihilation_results.append(annihilation_sign*annihilation_norm*cg_annihilation)

                    # Creation term:
                    creation_norm = 1/sqrt(2)

                    for ma in [0, 1, 2, 3]:
                        for mb in [0, 1, 2, 3]:
                            cg_creation = clebsch_gordan[(
                                3,
                                m_idx_to_m[ma],
                                3,
                                m_idx_to_m[mb],
                                J,
                                M,
                            )]
                            if cg_creation == 0: continue

                            new_right_state = []
                            new_left_state = list(left_state)

                            new_right_state.insert(0, mb)
                            new_right_state.insert(0, ma)
                            # new_right_state.sort()
                            if new_left_state == new_right_state:
                                creation_sign = 1
                            elif new_left_state == new_right_state[::-1]:
                                creation_sign = -1
                            else:
                                continue
                            
                            for annihilation_result in annihilation_results:
                                H[idx_left, idx_right] += creation_sign*V*creation_norm*cg_creation*annihilation_result

            # input("lel:")
            # print()
            # print(H)

    O18 = ksutil.loadtxt(path="O18_w_no_d5s1/", load_and_save_to_file=True)
    print(O18.levels)
    print()

    print(H)
    print()
    assert np.all(np.abs(H - H.T) < 1e-15)
    from numpy.linalg import eigh
    eigenvals, eigenvecs = eigh(H)
    print(eigenvals)

if __name__ == "__main__":
    main()