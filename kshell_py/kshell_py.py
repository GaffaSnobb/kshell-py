from parameters import flags, timing
from operator_j_scheme import operator_j_scheme
from data_structures import Interaction, OperatorJ, Partition
from loaders import read_interaction_file
from partition import initialise_partition
flags.debug = True
flags.timing_summary = False

def main(path_interaction: str, path_partition: str):
    interaction: Interaction = read_interaction_file(path=path_interaction)
    partition: Partition = initialise_partition(path=path_partition, interaction=interaction)
    # operator_j: OperatorJ = operator_j_scheme(
    #     interaction = interaction,
    #     nucleus_mass = 20
    # )
    
    if flags.timing_summary:
        print(timing)

if __name__ == "__main__":
    path_interaction_usda = "../snt/usda.snt"
    path_partition_usda = "../ptn/Ne20_usda_p.ptn"
    
    path_interaction_sdpfmu = "../snt/sdpf-mu.snt"
    path_partition_sdpfmu = "../ptn/V50_sdpf-mu_n.ptn"
    
    main(
        path_interaction = path_interaction_usda,
        path_partition = path_partition_usda
    )

    # main(
    #     path_interaction = path_interaction_sdpfmu,
    #     path_partition = path_partition_sdpfmu
    # )