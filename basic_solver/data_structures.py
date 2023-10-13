import warnings
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
class TimingGroupID:
    time: float
    group_ids: list[int]
    attr_ids: list[int]

# @dataclass(slots=True)
class Timing:
    """
    All timing values are in seconds and are named identically as the
    function they represent.
    """
    def __init__(self) -> None:
        self.calculate_onebody_matrix_element = TimingGroupID(
            time = 0.0,
            group_ids = [0],
            attr_ids = [3],
        )
        self.calculate_twobody_matrix_element = TimingGroupID(
            time = 0.0,
            group_ids = [0, 0],
            attr_ids = [4, 0],
        )
        self.twobody_annihilation_term = TimingGroupID(
            time = 0.0,
            group_ids = [0, 0],
            attr_ids = [4, 1],
        )
        self.twobody_creation_term = TimingGroupID(
            time = 0.0,
            group_ids = [0, 0],
            attr_ids = [4, 2],
        )
        self.generate_indices = TimingGroupID(
            time = 0.0,
            group_ids = [0],
            attr_ids = [1],
        )
        self.calculate_m_basis_states = TimingGroupID(
            time = 0.0,
            group_ids = [0],
            attr_ids = [2],
        )
        self.create_hamiltonian = TimingGroupID(
            time = 0.0,
            group_ids = [0],
            attr_ids = [0],
        )
        self.total = TimingGroupID(
            time = 0.0,
            group_ids = [],
            attr_ids = [0],
        )

    def __str__(self):
        """
        Return a nicely formatted time table.
        """
        group_total_sum = 0.0   # The sum of all the group times (total program time).
        TIME_TOL = 1e-3         # Tolerance for the assertions.
        # total = sum(getattr(self, attr) for attr in self.__slots__ if ((getattr(self, attr) != 0) and (attr != "main")))
        defined_group_ids = [0]    # The group IDs to be included in the table.

        time_summary: str = "------------\n"

        group_timings: dict[int, list[tuple[int, str, float]]] = {}
        sub_group_timings: dict[tuple[int, int], list[tuple[str, float]]] = {}
        
        for defined_group in defined_group_ids:
            """
            Loop over all defined groups.
            """
            group_timings[defined_group] = []
            group_total: float = 0.0
            sub_group_total: float = 0.0
            
            for attr_name in sorted([attr_name for attr_name in dir(self) if not attr_name.startswith('__')]):
                """
                Loop over all non-special attributes.
                """
                if attr_name == "total":
                    """
                    The total time will be separately treated.
                    """
                    continue
                
                val: TimingGroupID = getattr(self, attr_name)

                time = val.time
                group_ids = val.group_ids
                attr_ids = val.attr_ids

                if len(group_ids) == 2:
                    try:
                        sub_group_timings[(0, 4)].append([attr_ids[1], attr_name, time])
                    except KeyError:
                        sub_group_timings[(0, 4)] = []
                        sub_group_timings[(0, 4)].append([attr_ids[1], attr_name, time])

                    if attr_ids[1] != 0:
                        """
                        We want to include the sub-group parents in the
                        main group overview. Sub-group children however,
                        are not included.
                        """
                        sub_group_total += time
                        continue

                if group_ids[0] != defined_group:
                    """
                    For each attribute, group_ids[0] is the ID of the
                    main group.
                    """
                    continue
                
                if attr_ids[0] != 0:
                    """
                    attr_ids[0] is the parent of the group which
                    contains function calls of functions which contain
                    their own timings. Dont include the parent in the
                    sum since the time sum of the children should equal
                    the time of the parent, minus the overhead of the
                    parent.
                    """
                    group_total += time

                # group_timings[defined_group].insert(attr_ids[0], (attr_name, time))
                group_timings[defined_group].append([attr_ids[0], attr_name, time])

            group_timings[defined_group].sort(key=lambda tup: tup[0])

            group_timings[defined_group].insert(    # This is the overhead time of the parent.
                1,
                (1, group_timings[defined_group][0][1], group_timings[defined_group][0][2] - group_total)
            )
            try:
                """
                Currently hard-coded for sub group 0.
                """
                sub_group_timings[(0, 4)].sort(key=lambda tup: tup[0])

                sub_group_timings[(0, 4)][0][2] -= sub_group_total
                # sub_group_timings[(0, 4)].insert(    # This is the overhead time of the parent.
                #     0,
                #     (1, sub_group_timings[(0, 4)] [0][1], sub_group_timings[(0, 4)] [0][2] - sub_group_total)
                # )
                assert sub_group_timings[(0, 4)] [1][2] >= 0  # Should never be negative.
            except KeyError:
                pass

            assert group_timings[defined_group][1][2] >= 0  # Should never be negative.

        for defined_group in defined_group_ids:
            for i, group_timing in enumerate(group_timings[defined_group]):
                attr_id, attr_name, time = group_timing

                if i != 0:
                    """
                    Make the sub-times indented.
                    """
                    time_summary += "    "

                time_summary += f"{time*100/group_total:4.0f}% {time:10.4f}s: {attr_name}\n"
                
                try:
                    sub_group_timings[(defined_group, attr_id)]
                except KeyError:
                    pass
                else:
                    for j, sub_group_timing in enumerate(sub_group_timings[(defined_group, attr_id)]):
                        sub_attr_id, sub_attr_name, sub_time = sub_group_timing
                        
                        if j != 0:
                            """
                            Make the sub-times indented.
                            """
                        time_summary += "        "

                        time_summary += f"{sub_time*100/sub_group_total:4.0f}% {sub_time:10.4f}s: {sub_attr_name}\n"







                

        # for group in groups:
        #     """
        #     Timings are collected in groups. The attr in a group with
        #     attr ID 0 is the 'head' of the group and all other attrs in
        #     the same group should sum up to attr ID 0's time.
        #     """
        #     group_timings: list[tuple[str, int]] = []
        #     sub_group_timings = {}
        #     for attr in sorted([attr for attr in dir(self) if not attr.startswith('__')]):
        #         """
        #         Fetch the attrs and their values for the current group.
        #         """
        #         if attr == "main":
        #             """
        #             Treat the main time separately.
        #             """
        #             continue
                
        #         val = getattr(self, attr)
        #         time = val.time
        #         group_id_main_group = val.group_id[0]
        #         attr_id_main_group = val.attr_id[0]
        #         if group_id_main_group != group: continue

        #         if len(val.group_id) == 2:
        #             attr_id_sub_group = val.attr_id[1]
        #             try:
        #                 sub_group_timings[(group_id_main_group, attr_id_main_group)].append()
        #             except KeyError:
        #                 sub_group_timings[(group_id_main_group, attr_id_main_group)] = []
        #                 sub_group_timings[(group_id_main_group, attr_id_main_group)].append()
                
        #         group_timings.insert(attr_id_main_group, (attr, time))

        #     group_total = 0.0
        #     for attr, val in group_timings[1:]:
        #         """
        #         Sum up all the times of a group, except for the head of
        #         the group which should be the total group time.
        #         """
        #         group_total += val

        #     group_total_sum += group_timings[0][1]
        #     group_timings.insert(1, (group_timings[0][0], abs(group_timings[0][1] - group_total)))  # Insert the overhead of the head of the group as the second element.

        #     for i in range(len(group_timings)):
        #         attr, val = group_timings[i]

        #         if i != 0:
        #             """
        #             Make the sub-times indented.
        #             """
        #             time_summary += "    "

        #         time_summary += f"{val*100/group_total:4.0f}% {val:10.4f}s: {attr}\n"

        time_summary += "------------"

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