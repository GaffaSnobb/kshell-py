from dataclasses import dataclass, field

@dataclass(slots=True)
class Indices:
    """
    Attributes
    ----------
    orbital_idx_to_m_idx_map : list[tuple[int, ...]]
        Map the index of an orbital in a model space to the magnetic
        substate indices of that orbital. How the orbitals are ordered
        is defined by the interaction file (.snt). For example, in w.snt
        the indices are as follows:

            0 = p 0d_3/2
            1 = p 0d_5/2
            2 = p 1s_1/2
            3 = n 0d_3/2
            4 = n 0d_5/2
            5 = n 1s_1/2

        Please note that I have changed the indices to start at 0 while
        in the actual interaction file the indices start at 1. This
        means that orbital 0 (d3/2) has four magnetic substates and thus

            orbital_idx_to_m_idx_map[0] = (0, 1, 2, 3)
            orbital_idx_to_m_idx_map[1] = (0, 1, 2, 3, 4, 5)
            orbital_idx_to_m_idx_map[2] = (0, 1)
            ...

    orbital_m_pair_to_composite_m_idx_map : dict[tuple[int, int], int]
        Map a pair of (orbital index, m substate index) to a composite
        m substate index. Please consider the following scheme sd model
        space scheme:

                          10    11        
                        -  O  -  O  -             s1/2: 2
                         -1/2   1/2

               4     5     6     7     8     9    
            -  O  -  O  -  O  -  O  -  O  -  O    d5/2: 1
             -5/2  -3/2  -1/2   1/2   3/2   5/2

                     0     1     2     3
                  -  O  -  O  -  O  -  O  -       d3/2: 0
                   -3/2  -1/2   1/2   3/2

        The orbitals are indexed 0, 1, 2. The m substates are indexed by
        a composite / cumulative / whatever you wanna call it, m
        substate index, 0, 1, ..., 11. This map translates the orbital
        index and the orbital's 'local' m substate index into the
        composite m substate index. The 'local' m indices are for d3/2,
        d5/2, s1/2 respectively,

            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5],
            [0, 1].

        orbital_idx_to_j_map : list[int]
            Map the index of an orbital to its 2*j value. Considering
            the above scheme,

                orbital_idx_to_j_map = [3, 5, 1]

        m_composite_idx_to_m_map : list[int]
            Map a composite m index to the 2*m value it corresponds to.
            Considering the above scheme,

                m_composite_idx_to_m_map = [
                    -3, -1, +1, +3, -5, -3, -1, +1, +3, +5, -1, +1,
                ]

    """
    orbital_idx_to_m_idx_map: list[tuple[int, ...]] = field(default_factory=list)
    orbital_m_pair_to_composite_m_idx_map: dict[tuple[int, int], int] = field(default_factory=dict)
    orbital_idx_to_j_map: list[int] = field(default_factory=list)
    m_composite_idx_to_m_map: list[int] = field(default_factory=list)

@dataclass(slots=True)
class Interaction:
    spe: list[float]
    tbme: dict[tuple[int, int, int, int, int], float]

@dataclass(slots=True)
class Timing:
    """
    All timing values are in seconds and are named identically as the
    function they represent.
    """
    fill_orbitals: float = -1.0
    calculate_all_possible_orbital_occupations: float = -1.0
    calculate_all_possible_pairs: float = -1.0
    create_hamiltonian: float = -1.0
    main: float = -1.0

    def __str__(self):
        total = sum(getattr(self, attr) for attr in self.__slots__ if ((getattr(self, attr) != -1) and (attr != "main")))

        time_summary: str = "------------\n"

        for attr in self.__slots__:
            val = getattr(self, attr)
            if val == -1: continue
            if attr == "main": continue
            time_summary += f"{val:10.4f} s, {val*100/total:4.0f} %: {attr}\n"

        time_summary += f"{total:10.4f} s, {total/total*100:4.0f} %: total\n"
        time_summary += "------------"

        assert abs(total - self.main) < 1e-3, "Times do not add up! Get yo shit together!"

        return time_summary


timings: Timing = Timing()

@dataclass(slots=True)
class Configuration:
    """
    configuration : tuple[int, ...]
        A tuple containing an allowed orbital occupation. Example:
            
            (0, 1, 2)

        where the numbers tell us that the first orbital has zero
        nucleons, the second orbital has 1 nucleon, and the third
        orbital has 2 nucleons. The order of the orbitals is the same as
        the order they are listed in the interaction file.

    parity : int
        The parity of the allowed orbital occupation.
    """
    configuration: tuple[int, ...]
    parity: int

    def __lt__(self, other):
        if isinstance(other, Configuration):
            return self.configuration < other.configuration
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Configuration):
            return self.configuration <= other.configuration
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Configuration):
            return self.configuration > other.configuration
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Configuration):
            return self.configuration >= other.configuration
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Configuration):
            return self.configuration == other.configuration
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Configuration):
            return self.configuration != other.configuration
        return NotImplemented

@dataclass(slots=True)
class Partition:
    """
    configurations : list[Configuration]
        A list containing a configuration object with information about
        the allowed orbital occupations. Example of the configurations:
            
            [(0, 1, 2), (0, 2, 1), ...]

        where the first allowed occupation tells us that the first
        orbital has zero nucleons, the second orbital has 1 nucleon, and
        the third orbital has 2 nucleons. The order of the orbitals is
        the same as the order they are listed in the interaction file.
    """
    configurations: list[Configuration] = field(default_factory=list)