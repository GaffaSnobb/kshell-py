import ast, time
from typing import Union
import numpy as np
from data_structures import Interaction, ModelSpace, OneBody, TwoBody, Partition
from parameters import flags, debug, timing

def read_interaction_file(
    path: str
) -> Interaction:
    """
    Read an interaction file (.snt) and store all relevant parameters
    as an Interaction dataclass. Part of Python translation of
    operator_jscheme.f90 (subroutine read_intfile).

    Changes
    -------
    The two integer arrays kfin and kini are removed in favor of
    directly using the values they contain.
    kfin = (no. proton orbitals, no. proton + neutron orbitals)
    kini = (1, no. proton orbitals + 1)

    norb, jorb, etc. are saved as a dataclass (ModelSpace) of arrays
    instead of being individual arrays.

    Parameters
    ----------
    path : str
        Path to interaction file (.snt file).
    """
    read_interaction_time = time.perf_counter()

    with open(path, "r") as infile:
        for line in infile:
            """
            Read the number of j orbitals and the number of nucleons.

            ! ...
            ! model space 
            3   3     8   8                             <---- This line!
                1     0   2   3  -1  !   1 = p 0d_ 3/2
                2     0   2   5  -1  !   2 = p 0d_ 5/2
                3     1   0   1  -1  !   3 = p 1s_ 1/2
                4     0   2   3   1  !   4 = n 0d_ 3/2
                5     0   2   5   1  !   5 = n 0d_ 5/2
                6     1   0   1   1  !   6 = n 1s_ 1/2
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue

            line_split = [ast.literal_eval(elem) for elem in line.split()]

            if not all(isinstance(elem, int) for elem in line_split):
                msg = "The number of j orbitals and core nucleons should only be integers!"
                msg += " Might be due to unexpected snt file structure."
                raise TypeError(msg)

            n_proton_orbitals, n_neutron_orbitals = line_split[:2]
            n_orbitals = n_proton_orbitals + n_neutron_orbitals
            n_core_protons, n_core_neutrons = line_split[2:]
            
            model_space: ModelSpace = ModelSpace(
                n = np.zeros(n_orbitals, dtype=int),
                l = np.zeros(n_orbitals, dtype=int),
                j = np.zeros(n_orbitals, dtype=int),
                jz = np.zeros(n_orbitals, dtype=np.ndarray),
                isospin = np.zeros(n_orbitals, dtype=int),
                parity = np.zeros(n_orbitals, dtype=int),
                n_proton_orbitals = n_proton_orbitals,
                n_neutron_orbitals = n_neutron_orbitals,
                n_orbitals = n_orbitals,
                n_orbital_degeneracy = np.zeros(n_orbitals, dtype=int)
            )
            break
        
        # jz_max = -np.inf
        for i in range(model_space.n_orbitals):
            """
            ! ...
            ! model space 
            3   3     8   8
                1     0   2   3  -1  !   1 = p 0d_ 3/2  <---- These lines!
                2     0   2   5  -1  !   2 = p 0d_ 5/2  <----
                3     1   0   1  -1  !   3 = p 1s_ 1/2  <----
                4     0   2   3   1  !   4 = n 0d_ 3/2  <----
                5     0   2   5   1  !   5 = n 0d_ 5/2  <----
                6     1   0   1   1  !   6 = n 1s_ 1/2  <----
            """
            line_split = [ast.literal_eval(elem) for elem in infile.readline().split()[1:5]]
            
            if not all(isinstance(elem, int) for elem in line_split):
                msg = "Orbital parameters should only be integers!"
                msg += " Might be due to unexpected snt file structure."
                raise TypeError(msg)

            model_space.n[i] = line_split[0]
            model_space.l[i] = line_split[1]
            model_space.j[i] = line_split[2]
            model_space.isospin[i] = line_split[3]
            model_space.parity[i] = (-1)**model_space.l[i]

            model_space.n_orbital_degeneracy[i] = model_space.j[i] + 1  # Degeneracy is 2*j + 1, but the j values are stored as 2*j already.
            model_space.jz[i] = np.arange(-model_space.j[i], model_space.j[i] + 1, 2, dtype=int)    # From -j to j in integer steps.
            # model_space.jz_max = max(jz_max, max(model_space.jz[i]))

        for line in infile:
            """
            Read the number of one-body interactions and method(?).
            NOTE: Figure out what method means.
            
            ! ...
            ! num, method=,  hbar_omega
            !  i  j     <i|H(1b)|j>
            6   0                       <---- This line!
            1   1      1.97980000
            2   2     -3.94360000
            3   3     -3.06120000
            4   4      1.97980000
            5   5     -3.94360000
            6   6     -3.06120000
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue
            
            n_onebody_elements, method_onebody = [ast.literal_eval(elem) for elem in line.split()]
            break

        one_body: OneBody = OneBody(
            orbital_0 = np.zeros(n_onebody_elements, dtype=int),
            orbital_1 = np.zeros(n_onebody_elements, dtype=int),
            reduced_matrix_element = np.zeros(n_onebody_elements, dtype=float),
            method = method_onebody,
            n_elements = n_onebody_elements
        )
        
        for i in range(one_body.n_elements):
            """
            Read the one-body orbitals and reduced matrix elements.
            Example
            -------
            ! ...
            ! num, method=,  hbar_omega
            !  i  j     <i|H(1b)|j>
            6   0
            1   1      1.97980000   <---- These lines!
            2   2     -3.94360000   <---- 
            3   3     -3.06120000   <---- 
            4   4      1.97980000   <----
            5   5     -3.94360000   <----
            6   6     -3.06120000   <----
            """
            one_body.orbital_0[i], one_body.orbital_1[i], one_body.reduced_matrix_element[i] = \
                [ast.literal_eval(elem) for elem in infile.readline().split()]
            one_body.orbital_0[i] -= 1  # Indices start at 1 in the snt files.
            one_body.orbital_1[i] -= 1

            if one_body.orbital_0[i] != one_body.orbital_1[i]:
                msg = "One-body indices are not identical!"
                msg += f" Got {one_body.orbital_0[i] = }, {one_body.orbital_1[i] = }"
                msg += f" on line {i}. (Don't know how to treat this yet!)"
                raise NotImplementedError(msg)

        for line in infile:
            """
            Read the number of two-body interactions, method(?), im0(?)
            and pwr(?).
            NOTE: Figure out what im0 and pwr means.

            Example
            -------
            ! ...
            ! TBME
                    158   1  18 -0.300000           <---- This line!
            1   1   1   1    0       -1.50500000
            1   1   1   1    2       -0.15700000
            1   1   1   2    2        0.52080000
            1   1   1   3    2        0.13680000
            1   1   2   2    0       -3.56930000
            ...
            """
            if line.startswith("!"):
                """
                Skip comments.
                """
                continue
            
            try:
                n_twobody_elements, method_twobody, im0, pwr = [ast.literal_eval(elem) for elem in line.split()]
            except ValueError:
                msg = "Only method_twobody = 1 is implemented! For snt files"
                msg += " of method_twobody != 1, im0 and / or pwr might not"
                msg += " be present in the file, hence error when unpacking"
                msg += " the list comprehension."
                raise NotImplementedError(msg)
            break

        two_body: TwoBody = TwoBody(
            orbital_0 = np.zeros(n_twobody_elements, dtype=int),
            orbital_1 = np.zeros(n_twobody_elements, dtype=int),
            orbital_2 = np.zeros(n_twobody_elements, dtype=int),
            orbital_3 = np.zeros(n_twobody_elements, dtype=int),
            parity = np.zeros(n_twobody_elements, dtype=int),
            jj = np.zeros(n_twobody_elements, dtype=int),
            reduced_matrix_element = np.zeros(n_twobody_elements, dtype=float),
            method = method_twobody,
            im0 = im0,
            pwr = pwr,
            n_elements = n_twobody_elements
        )

        for i in range(two_body.n_elements):
            """
            I think jj is the coupled total angular momentum of two
            nucleons in the two-body interaction. However, consider the
            line
            (orb 0)  (orb 1)  (orb 2)  (orb 3)   (jj)      (mat elem)
               1        1        1        1       2       -0.15700000

            from usda.snt. Here jj = 2, but orbital 1 is 0d3/2. 3/2 +
            3/2 equals 3, not 2. Recall that d orbitals have l = 2.
            """
            two_body.orbital_0[i], two_body.orbital_1[i], two_body.orbital_2[i], \
            two_body.orbital_3[i], two_body.jj[i], two_body.reduced_matrix_element[i] = \
                [ast.literal_eval(elem) for elem in infile.readline().split()]
            
            two_body.orbital_0[i] -= 1     # snt indices start from 1.
            two_body.orbital_1[i] -= 1
            two_body.orbital_2[i] -= 1
            two_body.orbital_3[i] -= 1
            parity_twobody_01 = model_space.parity[two_body.orbital_0[i]]*model_space.parity[two_body.orbital_1[i]]
            parity_twobody_23 = model_space.parity[two_body.orbital_2[i]]*model_space.parity[two_body.orbital_3[i]]

            if parity_twobody_01 != parity_twobody_23:
                msg = "parity_twobody_01 != parity_twobody_23."
                msg += " This raises an error in the KSHELL Fortran code."
                msg += " I dont know why yet."
                raise RuntimeError(msg)

            two_body.parity[i] = 0 if (parity_twobody_01 == 1) else 1  # _01 arbitrary choice, can just as well use _23.

        max_proton_j_couple = np.max(np.concatenate(model_space.jz[:model_space.n_proton_orbitals]))
        max_neutron_j_couple = np.max(np.concatenate(model_space.jz[model_space.n_proton_orbitals:]))
        max_j_couple = max(max_proton_j_couple, max_neutron_j_couple) + 1

        model_space.max_proton_j_couple = max_proton_j_couple
        model_space.max_neutron_j_couple = max_neutron_j_couple
        model_space.max_j_couple = max_j_couple

        proton_model_space: ModelSpace = ModelSpace(
            n = model_space.n[:model_space.n_proton_orbitals],
            l = model_space.l[:model_space.n_proton_orbitals],
            j = model_space.j[:model_space.n_proton_orbitals],
            jz = model_space.jz[:model_space.n_proton_orbitals],
            isospin = model_space.isospin[:model_space.n_proton_orbitals], 
            parity = model_space.parity[:model_space.n_proton_orbitals],
            n_orbitals = model_space.n_proton_orbitals,
            n_orbital_degeneracy = model_space.n_orbital_degeneracy[:model_space.n_proton_orbitals],
            max_j_couple = max_proton_j_couple
        )

        neutron_model_space: ModelSpace = ModelSpace(
            n = model_space.n[model_space.n_proton_orbitals:],
            l = model_space.l[model_space.n_proton_orbitals:],
            j = model_space.j[model_space.n_proton_orbitals:],
            jz = model_space.jz[model_space.n_proton_orbitals:],
            isospin = model_space.isospin[model_space.n_proton_orbitals:],
            parity = model_space.parity[model_space.n_proton_orbitals:],
            n_orbitals = model_space.n_neutron_orbitals,
            n_orbital_degeneracy = model_space.n_orbital_degeneracy[model_space.n_proton_orbitals:],
            max_j_couple = max_neutron_j_couple
        )

        interaction: Interaction = Interaction(
            one_body = one_body,
            two_body = two_body,
            model_space = model_space,
            n_core_protons = n_core_protons,
            n_core_neutrons = n_core_neutrons,
            n_core_nucleons = n_core_protons + n_core_neutrons,
            proton_model_space = proton_model_space,
            neutron_model_space = neutron_model_space
        )

        if any(two_body.orbital_0 > two_body.orbital_1):
            """
            See https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L154
            """
            msg = "One or more two-body orbital 0 indices are larger"
            msg += " than the accompanying orbital 1 indices in the"
            msg += " interaction file. Special care must be taken which"
            msg += " has not yet been implemented."
            raise NotImplementedError(msg)

        if any(two_body.orbital_2 > two_body.orbital_3):
            """
            See https://github.com/GaffaSnobb/kshell/blob/088e6b98d273a4b59d0e2d6d6348ec1012f8780c/src/operator_jscheme.f90#L155
            """
            msg = "One or more two-body orbital 2 indices are larger"
            msg += " than the accompanying orbital 3 indices in the"
            msg += " interaction file. Special care must be taken which"
            msg += " has not yet been implemented."
            raise NotImplementedError(msg)

        if model_space.n_proton_orbitals != model_space.n_neutron_orbitals:
            msg = "There are not equally many proton and neutron orbitals in"
            msg += " the interaction file! This might cause problems with the"
            msg += " idx_reverse indices!"
            raise NotImplementedError(msg)

        debug.ij_01 = np.zeros(n_twobody_elements, dtype=int)
        debug.ij_23 = np.zeros(n_twobody_elements, dtype=int)
        debug.proton_neutron_idx = np.zeros(n_twobody_elements, dtype=int)

        read_interaction_time = time.perf_counter() - read_interaction_time
        timing.read_interaction_time = read_interaction_time
        timing.interaction_name = path.split("/")[-1]
        if flags.debug:
            print(f"{read_interaction_time = :.4f} s")

        return interaction

def read_partition_file(path: str) -> Partition:
    """
    Read a partition file and store all partition data in a Partition
    object.

    Parameters
    ----------
    path : str
        Path to the partition file.
    """
    read_partition_time: float = time.perf_counter()
    hw_min: Union[int, None] = None
    hw_max: Union[int, None] = None
    n_protons: Union[int, None] = None
    n_neutrons: Union[int, None] = None
    parity: Union[int, None] = None
    n_proton_configurations: Union[int, None] = None
    n_neutron_configurations: Union[int, None] = None
    n_proton_neutron_configurations: Union[int, None] = None
    line_counter: int = 0
    
    with open(path, "r") as infile:
        """
        Read the metadata of the partition file.
        """
        for line in infile:
            line_counter += 1
            if "hw" in line:
                """
                Example:
                # hw trucnation,  min hw = 0 ,   max hw = 1
                """
                hw_min, hw_max = line.split("=")[1:]
                hw_min = int(hw_min.split(",")[0])
                hw_max = int(hw_max)
            
            elif "partition file" in line:
                """
                Example:
                # partition file of sdpf-mu.snt  Z=15  N=19  parity=-1
                15 19 -1
                """
                line = infile.readline()
                line_counter += 1
                n_protons, n_neutrons, parity = [int(i) for i in line.split()]

            elif "proton partition, neutron partition" in line:
                """
                Example:
                # num. of  proton partition, neutron partition
                112 332
                """
                line = infile.readline()
                line_counter += 1
                n_proton_configurations, n_neutron_configurations = [int(i) for i in line.split()]

            elif "proton partition" in line:
                """
                Read the line number of where the proton configurations
                start.
                """
                proton_partition_line_start = line_counter
            
            elif "partition of proton and neutron" in line:
                """
                When this condition is met, all metadata of the
                partition file has been read. The rest of the file
                contains the proton and neutron configurations and will
                be read with np.loadtxt.
                """
                line = infile.readline()
                n_proton_neutron_configurations = int(line)
                break

            else:
                continue

        else:
            """
            A successful run should break the loop.
            """
            msg = "The partition file does not contain the required"
            msg += " metadata!\n"
            msg += f"{hw_min = }\n"
            msg += f"{hw_max = }\n"
            msg += f"{n_protons = }\n"
            msg += f"{n_neutrons = }\n"
            msg += f"{parity = }\n"
            msg += f"{n_proton_configurations = }\n"
            msg += f"{n_neutron_configurations = }\n"
            msg += f"{n_proton_neutron_configurations = }"
            raise ValueError(msg)

    proton_configurations: np.ndarray = np.loadtxt(
        fname = path,
        skiprows = proton_partition_line_start,
        max_rows = n_proton_configurations,
        dtype = int
    )
    proton_configurations[:, 0] -= 1    # Convert to 0-based indexing.

    neutron_configurations: np.ndarray = np.loadtxt(
        fname = path,
        skiprows = proton_partition_line_start + n_proton_configurations,
        max_rows = n_neutron_configurations,
        dtype = int
    )
    neutron_configurations[:, 0] -= 1    # Convert to 0-based indexing.

    proton_neutron_configurations: np.ndarray = np.loadtxt(
        fname = path,
        skiprows = proton_partition_line_start + n_proton_configurations + n_neutron_configurations + 3,
        max_rows = n_proton_neutron_configurations,
        dtype = int
    )
    proton_neutron_configurations -= 1    # Convert to 0-based indexing.

    partition: Partition = Partition(
        n_protons = n_protons,
        n_neutrons = n_neutrons,
        n_proton_configurations = n_proton_configurations,
        n_neutron_configurations = n_neutron_configurations,
        n_proton_neutron_configurations = n_proton_neutron_configurations,
        parity = parity,
        hw_min = hw_min,
        hw_max = hw_max,
        proton_configurations = proton_configurations,
        neutron_configurations = neutron_configurations,
        proton_neutron_configurations = proton_neutron_configurations,
        proton_configurations_max_j = np.zeros(n_proton_configurations, dtype=int),
        neutron_configurations_max_j = np.zeros(n_neutron_configurations, dtype=int),
        proton_configurations_parity = np.zeros(n_proton_configurations, dtype=int),
        neutron_configurations_parity = np.zeros(n_neutron_configurations, dtype=int),
        proton_configurations_jz = np.zeros(n_proton_configurations, dtype=np.ndarray),
        neutron_configurations_jz = np.zeros(n_neutron_configurations, dtype=np.ndarray),
    )

    read_partition_time = time.perf_counter() - read_partition_time
    timing.read_partition_time = read_partition_time
    timing.partition_name = path.split("/")[-1]
    if flags.debug:
        print(f"{read_partition_time = :.4f} s")

    return partition
