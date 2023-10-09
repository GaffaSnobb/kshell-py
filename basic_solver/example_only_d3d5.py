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
       4     5     6     7     8     9    
    -  O  -  O  -  O  -  O  -  O  -  O    d5/2: 1
     -5/2  -3/2  -1/2   1/2   3/2   5/2

             0     1     2     3
          -  O  -  O  -  O  -  O  -       d3/2: 0
           -3/2  -1/2   1/2   3/2
    """
    interaction: Interaction = load_interaction(path="../snt/w/")
    # interaction: kshell_utilities.data_structures.Interaction = kshell_utilities.loaders.load_interaction(filename_interaction="../snt/w.snt")
    basis_states = (
        (0, 3),
        (0, 8),
        (1, 2),
        (1, 7),
        (2, 6),
        (3, 5),
        (4, 9),
        (5, 8),
        (6, 7),
    )
    jz_indices = [
        range(4),   # d3/2 has 4 m substates
        range(6),
    ]
    orbital_m_pair_to_m_idx = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 4,
        (1, 1): 5,
        (1, 2): 6,
        (1, 3): 7,
        (1, 4): 8,
        (1, 5): 9,
    }
    m_dim = len(basis_states)
    H = np.zeros((m_dim, m_dim))

    for idx_left in range(m_dim):
        for idx_right in range(m_dim):
            
            for a in range(2):  # There are 2 orbitals in the model space.
                b = a   # Most generally, should be for b in range(2), but the SPEs are only defined for a = b.
            
                for ma in jz_indices[a]:
                    ma = orbital_m_pair_to_m_idx[(a, ma)]
                    for mb in jz_indices[b]:
                        mb = orbital_m_pair_to_m_idx[(b, mb)]

                        left_state_copy = list(basis_states[idx_left])
                        new_right_state = list(basis_states[idx_right])
                        
                        try:
                            annihilation_idx = new_right_state.index(mb)
                        except ValueError:
                            continue
                        
                        annihilation_sign = (-1)**annihilation_idx
                        new_right_state.pop(annihilation_idx)

                        if ma in new_right_state: continue

                        new_right_state.insert(0, ma)

                        if left_state_copy == new_right_state:
                            creation_sign = 1
                        elif left_state_copy == new_right_state[::-1]:
                            creation_sign = -1
                        else:
                            continue

                        H[idx_left, idx_right] += annihilation_sign*creation_sign*interaction.spe[a]

    m_idx_to_m = {
        0: -3,
        1: -1,
        2: +1,
        3: +3,
        4: -5,
        5: -3,
        6: -1,
        7: +1,
        8: +3,
        9: +5,
    }
    j_idx_to_j = [3, 5] # 0 is 3/2, 1 is 5/2.
    for idx_left in range(m_dim):
        for idx_right in range(m_dim):

            for a in range(2):
                for b in range(a, 2):
                    for c in range(2):
                        for d in range(c, 2):

                            
                            j_min = max(
                                abs(j_idx_to_j[a] - j_idx_to_j[b]),
                                abs(j_idx_to_j[c] - j_idx_to_j[d]),
                            )
                            j_max = min(
                                j_idx_to_j[a] + j_idx_to_j[b],
                                j_idx_to_j[c] + j_idx_to_j[d],
                            )

                            for J in range(j_min, j_max + 2, 2):
                                for M in range(-J, J + 2, 2):
                                    
                                    try:
                                        V = interaction.tbme[(a, b, c, d, J)]
                                        # print((0, 0, 0, 0, J), V)
                                    except KeyError:
                                        continue
                                    
                                    # Annihilation term:
                                    annihilation_norm = 1/sqrt(1 + (c == d))
                                    
                                    annihilation_results = []
                                    
                                    for mc in jz_indices[c]:
                                        mc = orbital_m_pair_to_m_idx[(c, mc)]
                                        for md in jz_indices[d]:
                                            md = orbital_m_pair_to_m_idx[(d, md)]
                                            # left_state_copy = list(basis_states[idx_left])
                                            new_right_state = list(basis_states[idx_right])

                                            if mc not in new_right_state: continue

                                            annihilation_idx = new_right_state.index(mc)
                                            new_right_state.pop(annihilation_idx)
                                            annihilation_sign = (-1)**annihilation_idx

                                            if md not in new_right_state: continue

                                            new_right_state.remove(md)

                                            assert len(new_right_state) == 0    # Sanity check.

                                            cg_annihilation = clebsch_gordan[(
                                                j_idx_to_j[c],
                                                m_idx_to_m[mc],
                                                j_idx_to_j[d],
                                                m_idx_to_m[md],
                                                J,
                                                M,
                                            )]

                                            if cg_annihilation == 0: continue

                                            annihilation_results.append(annihilation_sign*annihilation_norm*cg_annihilation)

                                    # Creation term:
                                    creation_norm = 1/sqrt(1 + (a == b))

                                    for ma in jz_indices[a]:
                                        ma = orbital_m_pair_to_m_idx[(a, ma)]
                                        for mb in jz_indices[b]:
                                            mb = orbital_m_pair_to_m_idx[(b, mb)]

                                            cg_creation = clebsch_gordan[(
                                                j_idx_to_j[a],
                                                m_idx_to_m[ma],
                                                j_idx_to_j[b],
                                                m_idx_to_m[mb],
                                                J,
                                                M,
                                            )]
                                            if cg_creation == 0: continue

                                            new_right_state = []
                                            tmp_left_state = list(basis_states[idx_left])

                                            new_right_state.insert(0, mb)
                                            new_right_state.insert(0, ma)
                                            # new_right_state.sort()
                                            if tmp_left_state == new_right_state:
                                                creation_sign = 1
                                            elif tmp_left_state == new_right_state[::-1]:
                                                creation_sign = -1
                                            else:
                                                continue
                                            
                                            for annihilation_result in annihilation_results:
                                                H[idx_left, idx_right] += creation_sign*V*creation_norm*cg_creation*annihilation_result

            # input("lel:")
            # print()
            # print(H)

    O18 = ksutil.loadtxt(path="O18_w_only_d3d5/", load_and_save_to_file=True)
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