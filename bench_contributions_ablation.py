import os
import ase
import time
import torch
import csv
import statistics
import sys
import warnings
import argparse

from model_maker import make_model

from hippynn.interfaces.ase_interface.calculator import calculator_from_model

from ase.io.proteindatabank import read_proteindatabank

from mace.calculators import mace_off
from ase import units

import os

# This file evaluates HIP-HOP-NN evaluation times (per atom) of different model sizes on a chosen molecule benchmark.
# Simply set data_folder equal to a file containing the .pdb files to benchmark on.

# Uncomment the line below in order to print out the autotuning results (can be nice for knowing how far into autotuning you are)
# os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():

    # list of (sensitivites, features) combinations that we want to test out.
    # nu_and_b_list = [(20,128), (20, 200), (20,256), (20,300), (20,400), (20,512), (40,128), (40,200), (40,256), (40,300)]
    nu_and_b_list = [(20,128)]

    if len(sys.argv) > 1:

        # I find it easier to set parameters as variables than to pass them in as command line arguments.

        # l_max and n_max are as they are in HopInvariantsLayer.py
        l_max=3
        n_max=4

        num_spin_up = 3 # How many times we run the calculator on each molecule before evaluating to get spin-up times (compiling etc.) out of the way.
        num_repeat = 10 # how many times we evaluate the timing on each molecule.
        size_thresh = 10000 # only time molecules that have less atoms than this threshold. Set this equal to -1 to time all molecules in the data folder.

        out_file = "../results/time_all.csv" # where should we write the results
        data_folder = "../data/large_molecule_pdb" # folder containing the molecules that we are testing (in .pdb format)

        suppress_model_creation_prints = False # Whether we want to suppress the print statements from creating models that say things like "determined inputs" etc.
        warnings.filterwarnings("ignore") # uncomment this line in order to ignore warnings that are coming from HIP-HOP-NN (the main warning that gets printed out is saying that it is in a beta stage)
    
    else:
        parser = argparse.ArgumentParser(prog="HIP-HOP-NN ablation timing with ASE calculator")

        parser.add_argument("-l_max", help="The maximum tensor order used in invariants", type=int, default=3)
        parser.add_argument("-n_max", help="The maximum number of tensors used in an invariant", type=int, default=4)

        parser.add_argument("-n_spinup", help="How many times do you spin up each evaluator before it runs.", type=int, default=3)
        parser.add_argument("-n_repeat", help="How many times do you time each molecule.")
        parser.add_argument("-thresh", help="Only benchmark on molecules with less atoms than this", type=float, default=float('inf'))

        parser.add_argument("-csv", help="The location of the output csv file", required=True)
        parser.add_argument("-i", help="The location of the large molecule files (in npz format).", required=True)

        parser.add_argument("-creation_prints", help="Display the model creaiton prints?", action=argparse.BooleanOptionalAction)
        parser.add_argument("-warnings", help="Display warnings?", action=argparse.BooleanOptionalAction)

        args = parser.parse_args(sys.argv[1:])

        l_max = args.l_max
        n_max = args.n_max

        num_spin_up = args.n_spinup
        num_repeat = args.n_repeat
        size_thresh = args.thresh

        out_file = args.csv
        data_folder = args.i

        suppress_model_creation_prints = not args.creation_prints
        if not args.warnings:
            warnings.filterwarnings("ignore")

    # END OF PARAMETERS #

    stdout = sys.stdout

    # open the results file and write the header
    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["l_max", "n_max", "nu", "b", "molecule", "time default", "time polys", "time grad fused", "time polys grad fused", "time env+grads fused", "time polys env+grads fused"])

    # load in the molecules from the data folder and load them in using ase.
    # This creates an array, test_molecules. Each entry is a tuple of the
    # molecule's filename, and then an object of the molecule.
    test_molecules = []
    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            inf = open(f"{data_folder}/{f}","r")
            a = read_proteindatabank(inf,index=0)
            test_molecules.append((f,a))

    # iterate over different model sizes.
    for nu, b in nu_and_b_list:

        # time each molecule.
        for m in test_molecules:

            if size_thresh == -1 or len(m[1]) < size_thresh:
                times = {} # stores the times that we will write to file

                # Ablation testing:
                # Each tuple specifies which of my optimizations are implemented comapred to the default.
                # The first entry is set to True to use polynomial invariants, and False to use default inariants.
                # The second entry is set to True to use the tensor message passing implementation for the forward pass, and False otherwise.
                # The third entry is set to True to use the tensor message passing implementation for the gradient computation, and False otherwise.

                # Throughout these trials, we run m[1].rattle(1e-6). This slightly perturbs the molecules with a standard deviation of 1e-6.
                # In doing so, the molecules are slightly shifted, which means that the results of the calculator will not be cached. On the other
                # hand, because the perturbation is so small, the graph will not need to be rebuilt. This allows us to actually re-run the forward pass
                # (just without the graph being rebuilt).
                for config in [(False,False,False), (True,False,False), (False,False,True), (True,False,True), (False,True,True), (True,True,True)]:
                    times_for_this_config = [] # only used for printing to console.

                    # Create the model, obtain the calculator, and set it as the molecule's calculator.
                    if suppress_model_creation_prints:
                        sys.stdout = open(os.devnull, "w")
                    model, _ = make_model(nu, b, l_max, n_max, *config)
                    sys.stdout = stdout

                    c = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')
                    m[1].calc = c

                    # run the calculator to get spin-up times out of the way.
                    print(f"{m[0]} ({len(m[1])}) [{config}]: spinning up...",end="\r")
                    for repeat in range(num_spin_up):
                        m[1].rattle(1e-6)
                        e = m[1].get_potential_energy()
                        f = m[1].get_forces()

                    # Run the calculator the number of times that 
                    for x in range(num_repeat):
                        print(f"{m[0]} ({len(m[1])}) [{config}]: {x+1} / {num_repeat}                                                ",end="\r")

                        m[1].rattle(1e-6)

                        torch.cuda.synchronize()
                        t1 = time.time()

                        e = m[1].get_potential_energy()
                        f = m[1].get_forces()

                        torch.cuda.synchronize()
                        t2 = time.time()

                        t_peratom = (t2-t1) / len(m[1])

                        times[ (x, *config) ] = t_peratom
                        times_for_this_config.append( t_peratom )

                    med_time = statistics.median(times_for_this_config)
                    print(f"{m[0]} ({len(m[1])}) [{config}]: {med_time}                                          ")

                # write times to the file:
                for x in range(num_repeat):
                    writer.writerow([ l_max, n_max, nu, b, m[0], times[(x,False,False,False)], times[(x,True,False,False)], times[(x,False,False,True)], times[(x,True,False,True)], times[(x,False,True,True)], times[(x,True,True,True)]] )

                outf.flush()

if __name__ == "__main__":
    main()