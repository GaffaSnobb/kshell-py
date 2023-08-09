from dataclasses import dataclass

@dataclass(slots=True)
class Timing:
    """
    All timing values are in seconds and are named identically as the
    function they represent.
    """
    fill_orbitals: float = -1.0
    calculate_hamiltonian_orbital_occupation: float = -1.0
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