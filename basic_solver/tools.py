from functools import cache
from sympy.physics.quantum.cg import CG
from scipy.special import comb

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