from parameters import flags, timing
from operator_j_scheme import operator_j_scheme
from data_structures import Interaction, OperatorJ
from loaders import read_interaction_file
from partition import initialise_partition
flags.debug = True
flags.timing_summary = False

def main(path="../snt/usda.snt"):
    interaction: Interaction = read_interaction_file(path=path)
    operator_j: OperatorJ = operator_j_scheme(
        interaction = interaction,
        nucleus_mass = 20
    )
    initialise_partition(path="../ptn/Ne20_usda_p.ptn")
    
    if flags.timing_summary:
        print(timing)

if __name__ == "__main__":
    main()