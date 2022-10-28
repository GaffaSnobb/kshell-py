import math, sys, time
import numpy as np
from parameters import flags, debug
from loaders import read_interaction_file
from data_structures import (
    ModelSpace, Interaction, OneBody, TwoBody, OperatorJ, CouplingIndices
)

def initialise_operator_j_couplings(
    interaction: Interaction
) -> CouplingIndices:
    """
    Initialise the j-scheme operator.
    """
    initialise_operator_j_couplings_time = time.perf_counter()
    n_parities: int = 2
    n_proton_neutron: int = 3   # Proton, neutron, both.
    
    proton_ms: ModelSpace = interaction.proton_model_space
    neutron_ms: ModelSpace = interaction.neutron_model_space
    max_j_couple: int = interaction.model_space.max_j_couple
    
    j_couple: np.ndarray = np.zeros(
        shape = (max_j_couple, n_parities, n_proton_neutron),
        dtype = CouplingIndices
    )

    # Initialise the proton coupling indices.
    for i in range(max_j_couple):
        for j in range(n_parities):
            j_couple[i, j, 0]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((proton_ms.n_orbitals, proton_ms.n_orbitals), dtype=int)
            )

    # Initialise the neutron coupling indices.
    for i in range(max_j_couple):
        for j in range(n_parities):
            j_couple[i, j, 1]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((neutron_ms.n_orbitals, neutron_ms.n_orbitals), dtype=int)
            )

    # Initialise the proton-neutron coupling indices.
    for i in range(max_j_couple):
        for j in range(n_parities):
            j_couple[i, j, 2]: CouplingIndices = CouplingIndices(
                n_couplings = 0,
                idx = np.zeros((2, 0), dtype=int),
                idx_reverse = np.zeros((proton_ms.n_orbitals, neutron_ms.n_orbitals), dtype=int)
            )

    calculate_couplings(
        model_space_1 = proton_ms,
        model_space_2 = proton_ms,
        j_couple = j_couple,
        proton_neutron_idx = 0
    )

    calculate_couplings(
        model_space_1 = neutron_ms,
        model_space_2 = neutron_ms,
        j_couple = j_couple,
        proton_neutron_idx = 1
    )

    calculate_couplings(
        model_space_1 = proton_ms,
        model_space_2 = neutron_ms,
        j_couple = j_couple,
        proton_neutron_idx = 2
    )

    initialise_operator_j_couplings_time = time.perf_counter() - initialise_operator_j_couplings_time
    if flags.debug:
        print(f"{initialise_operator_j_couplings_time = :.4f} s")

    return j_couple
    
def calculate_couplings(
    model_space_1: ModelSpace,
    model_space_2: ModelSpace,
    j_couple: np.ndarray,
    proton_neutron_idx: int
):
    """
    Calculate the possible couplings of proton-proton, neutron-neutron,
    and proton-neutron orbitals. model_space_1 and model_space_2 are
    not edited in this function, but the coupling indices are added to
    the j_couple array.

    Parameters
    ----------
    model_space_1 : ModelSpace
        Either proton or neutron model space.

    model_space_2 : ModelSpace
        Either proton or neutron model space.

    j_couple : np.ndarray
        Array of CouplingIndices.
    """
    for orbital_1 in range(model_space_1.n_orbitals):
        
        if proton_neutron_idx == 2:
            """
            In proton-neutron coupling, the couplings (i, j) and (j, i)
            are different couplings and both must be counted. The first
            case is proton orbital i and neutron orbital j, and the
            second case is neutron orbital i and proton orbital j.
            """
            range_start = 0
        else:
            """
            In proton-proton and neutron-neutron coupling, the couplings
            (i, j) and (j, i) are the same coupling and only one must be
            counted.
            """
            range_start = orbital_1
            
        for orbital_2 in range(range_start, model_space_2.n_orbitals):
            
            if model_space_1.parity[orbital_1] == model_space_2.parity[orbital_2]:
                """
                The total parity is always +1 if each orbital have the
                same parity. parity_idx 0 means positive parity.
                """
                parity_idx = 0
            else:
                """
                The total parity is always -1 if each orbital have
                different parities. parity_idx 1 means negative parity.
                """
                parity_idx = 1

            if (orbital_1 == orbital_2) and (proton_neutron_idx != 2):
                """
                Two identical nucleons in the same orbital can only have
                even-numbered couplings.
                """
                j_step = 2
            else:
                """
                Two nucleons in different orbitals can have even- and
                odd-numbered couplings.
                """
                j_step = 1

            possible_couplings: np.ndarray = np.arange( # NOTE: model_space.j values are stored as 2*j to avoid fractions, but the couplings are never fractions and are stored as-is!
                start = abs(model_space_1.j[orbital_1] - model_space_2.j[orbital_2])/2,
                stop = (model_space_1.j[orbital_1] + model_space_2.j[orbital_2] + j_step)/2,    # + j_step to include the end point.
                step = j_step,
                dtype = int
            )

            for j in possible_couplings:
                """
                Count each possible coupling (n_couplings). Save the
                indices of the coupled orbitals (idx) as column vectors
                in a 2D array.

                idx_reverse is organised like this for j = 0, parity =
                +1 (using usda.snt as an example):

                    O1  O2  O3
                O1  1   0   0
                O2  0   2   0
                O3  0   0   3

                meaning that the zero angular momentum coupling only
                happens within the same orbital and that the coupling
                between orbital 1 and orbital 1 was counted first,
                orbital 2 to orbital 2 was counted second, etc.
                """
                j_couple[j, parity_idx, proton_neutron_idx].n_couplings += 1  # Count the coupling.
                
                j_couple[j, parity_idx, proton_neutron_idx].idx = np.concatenate( # Append the indices.
                    (j_couple[j, parity_idx, proton_neutron_idx].idx, np.array([orbital_1, orbital_2], dtype=int).reshape(-1, 1)),
                    axis = 1
                )
                
                j_couple[j, parity_idx, proton_neutron_idx].idx_reverse[orbital_1, orbital_2] = \
                    np.max(j_couple[j, parity_idx, proton_neutron_idx].idx_reverse) + 1   # Save the order of which the coupling was counted (se docstring for structure).

def operator_j_scheme(
    path: str,
    nucleus_mass: int
) -> OperatorJ:
    """
    In this function, the "raw" reduced matrix elements from the
    interaction file are multiplied by a few different factors and
    stored in an OperatorJ object. NOTE: At the moment I do not know
    what the purpose of changing the raw matrix elements is.

    Parameters
    ----------
    path : str
        Path to interaction file (.snt file).

    nucleus_mass : int
        Mass number of the nucleus. E.g. 20 for 20Ne.
    """
    n_parities: int = 2
    n_proton_neutron: int = 3
    interaction: Interaction = read_interaction_file(path=path)
    j_couple: CouplingIndices = initialise_operator_j_couplings(interaction=interaction)
    
    operator_j_scheme_time: float = time.perf_counter()

    operator_j: OperatorJ = OperatorJ(
        one_body_reduced_matrix_element = np.zeros(
            shape = interaction.one_body.n_elements,
            dtype = float
        ),
        two_body_reduced_matrix_element = np.zeros(
            shape = (interaction.model_space.max_j_couple, n_parities, n_proton_neutron),
            dtype = np.ndarray
        ) 
    )

    for i in range(interaction.model_space.max_j_couple):
        """
        Initialise the two-body matrix element arrays. From Fortran:
        https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L669
        """
        for j in range(n_parities):
            for k in range(n_proton_neutron):
                operator_j.two_body_reduced_matrix_element[i, j, k]: np.ndarray = np.zeros(
                    shape = (j_couple[i, j, k].n_couplings, j_couple[i, j, k].n_couplings),
                    dtype = float
                )

    # Aliases for shorter code.
    one_body: OneBody = interaction.one_body
    two_body: TwoBody = interaction.two_body
    model_space: ModelSpace = interaction.model_space

    # ONE-BODY
    if one_body.method == 0:
        one_body_mass_dependence = 1    # NOTE: Whats this?
    else:
        msg = f"one_body_mass_dependence only implemented for one-body method 0. Got {one_body.method}."
        raise NotImplementedError(msg)

    for i in range(one_body.n_elements):
        """
        One-body loop.

        NOTE: operator_jscheme.f90 stores the reduced matrix elements as
        2D matrices and stores an entry for [orbital_0, orbital_1] and
        [orbital_1, orbital_0]. Swapping the indices changes nothing
        since they are equal (at least for usda.snt, jun45.snt,
        and gs8.snt) so I don't do that here. read_interaction_file also
        raises an error if they are not equal, so they will always be
        equal here.

        From usda.snt:
        ! num, method=,  hbar_omega
        !  i  j     <i|H(1b)|j>
        6   0
        1   1      1.97980000
        2   2     -3.94360000
        3   3     -3.06120000
        4   4      1.97980000
        5   5     -3.94360000
        6   6     -3.06120000

        Note that the reduced matrix element for orbitals (i, j) is
        equal to the reduced matrix element for orbitals (j, i) since
        i = j.
        """
        factor = math.sqrt(model_space.j[i] + 1) # NOTE: Figure out the origin of this factor.
        operator_j.one_body_reduced_matrix_element[i] = \
            one_body.reduced_matrix_element[i]*one_body_mass_dependence*factor

    # TWO-BODY
    if two_body.method == 1:
        """
        This happens in subroutine read_firstline_2b in the Fortran
        code. https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L233
        """
        two_body_mass_dependence = (nucleus_mass/two_body.im0)**two_body.pwr
    else:
        msg = "Only method 1 is implemented for two-body."
        msg += f" Got: {two_body.method}."
        raise NotImplementedError(msg)

    for i in range(two_body.n_elements):
        """
        Two-body loop.
        """
        isign_01 = 1    # NOTE: Dont know what these are yet.
        isign_23 = 1

        if (
            (two_body.orbital_0[i] > two_body.orbital_1[i]) or
            (two_body.orbital_2[i] > two_body.orbital_3[i])
        ):
            """
            NOTE: Don't know why this has to happen (copied from
            operator_jscheme.f90). gs8.snt, jun45.snt and usda.snt
            do not fulfill this condition. For now this raises an error.

            See swap in the Fortran code:
            https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L636
            https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L154-L155

            if orbital_0 > orbital_1:
                orbital_0, orbital_1 = orbital_1, orbital_0
                isign_01 = (-1)**((model_space.j(orbital_0) + model_space.j(orbital_1))/2 - jj + 1)
            
            if orbital_2 > orbital_3:
                orbital_2, orbital_3 = orbital_3, orbital_2
                isign_23 = (-1)**((model_space.j(orbital_2) + model_space.j(orbital_3))/2 - jj + 1)
            """
            msg = "orbital_0 > orbital_1 or orbital_2 > orbital_3."
            msg += " Don't know why this has to be taken care of."
            raise NotImplementedError(msg)
        
        if (
            (two_body.orbital_1[i] < (model_space.n_proton_orbitals)) and
            (two_body.orbital_3[i] < (model_space.n_proton_orbitals))
        ):
            """
            If orbital idx 1 and 3 are smaller than the number of
            proton orbitals then they are proton orbitals.
            """
            proton_neutron_idx = 0  # 0 is proton.

        elif (
            (two_body.orbital_0[i] >= (model_space.n_proton_orbitals)) and
            (two_body.orbital_2[i] >= (model_space.n_proton_orbitals))
        ):
            """
            If orbital 0 and 2 are greater than or equal to the number
            of proton orbitals then they are neutron orbitals.
            """
            proton_neutron_idx = 1  # 1 is neutron.

        elif (
            (two_body.orbital_0[i] <  (model_space.n_proton_orbitals)) and
            (two_body.orbital_2[i] <  (model_space.n_proton_orbitals)) and
            (two_body.orbital_1[i] >= (model_space.n_proton_orbitals)) and
            (two_body.orbital_3[i] >= (model_space.n_proton_orbitals))
        ):
            proton_neutron_idx = 2  # 2 is proton-neutron.
        
        else:
            """
            Two-body orbital indices are not valid.
            """
            msg = "Two-body orbital indices are not valid."
            msg += f" Error in orbital index iteration {i} from interaction file: "
            msg += f"{path}.\n"
            msg += f"{two_body.orbital_0[i] = }\n"
            msg += f"{two_body.orbital_1[i] = }\n"
            msg += f"{two_body.orbital_2[i] = }\n"
            msg += f"{two_body.orbital_3[i] = }\n"
            msg += f"{model_space.n_proton_orbitals = }"
            raise RuntimeError(msg)

        debug.proton_neutron_idx[i] = proton_neutron_idx

        """
        Since idx_reverse always has the same number of rows and cols,
        be it for protons (proton_neutron_idx = 0), neutrons (= 1) or
        proton-neutron (= 2), and since the neutron orbital numbering
        continues after the proton orbital numbering, neutron orbital
        indices (indices >= n_proton_orbitals) has to be wrapped around
        to zero to correctly index the idx_reverse arrays.

        Note that this is only possible if n_proton_orbitals =
        n_neutron_orbitals (maybe also if n_proton_orbitals >=
        n_neutron_orbitals) which is a requirement for the function
        read_interaction_file to not raise an error. I'm not sure if any
        of the interaction files that comes with KSHELL has a different
        number of proton and neutron orbitals, certainly not usda and
        sdpf-mu.
        """
        orbital_idx_0 = two_body.orbital_0[i]%model_space.n_proton_orbitals
        orbital_idx_1 = two_body.orbital_1[i]%model_space.n_proton_orbitals
        orbital_idx_2 = two_body.orbital_2[i]%model_space.n_proton_orbitals
        orbital_idx_3 = two_body.orbital_3[i]%model_space.n_proton_orbitals

        ij_01 = j_couple[
            two_body.jj[i], two_body.parity[i], proton_neutron_idx
        ].idx_reverse[orbital_idx_0, orbital_idx_1]

        ij_23 = j_couple[
            two_body.jj[i], two_body.parity[i], proton_neutron_idx
        ].idx_reverse[orbital_idx_2, orbital_idx_3]

        if ij_01 == 0:
            msg = f"No coupling found for orb_0[{i}] = {two_body.orbital_0[i]}"
            msg += f" and orb_1[{i}] = {two_body.orbital_1[i]}."
            msg += " If this happens, the construction of idx_reverse might be incorrect."
            raise RuntimeError(msg)

        if ij_23 == 0:
            msg = f"No coupling found for orb_2[{i}] = {two_body.orbital_2[i]}"
            msg += f" and orb_3[{i}] = {two_body.orbital_3[i]}."
            msg += " If this happens, the construction of idx_reverse might be incorrect."
            raise RuntimeError(msg)

        ij_01 -= 1  # Python indices start from 0, not 1.
        ij_23 -= 1
        debug.ij_01[i] = ij_01
        debug.ij_23[i] = ij_23
        
        m_element_tmp = two_body.reduced_matrix_element[i]*two_body_mass_dependence*isign_01*isign_23
        operator_j.two_body_reduced_matrix_element[two_body.jj[i], two_body.parity[i], proton_neutron_idx][ij_01, ij_23] = m_element_tmp
        operator_j.two_body_reduced_matrix_element[two_body.jj[i], two_body.parity[i], proton_neutron_idx][ij_23, ij_01] = m_element_tmp

    operator_j_scheme_time = time.perf_counter() - operator_j_scheme_time
    if flags.debug:
        print(f"{operator_j_scheme_time = :.4f} s")

    return operator_j

if __name__ == "__main__":
    flags.debug = True
    # read_interaction_file(path="usda.snt")
    operator_j_scheme(
        path = "../snt/usda.snt",
        nucleus_mass = 20
    )