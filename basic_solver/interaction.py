from kshell_utilities.data_structures import Interaction

def fill_orbital_m_substates(
    interaction: Interaction,
):
    # m_substates: list[int] = []

    m_substates: tuple[int] = (1,)

    for orb in interaction.model_space_neutron.orbitals:
        print(f"{orb.jz = }")

    print(m_substates)

    # print(interaction.model_space_neutron.orbitals[2].jz)