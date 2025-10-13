import os
import ase
import time
import statistics
import torch
import csv
import warnings
import sys
import argparse

from model_maker import make_model

import hippynn
from hippynn.interfaces.ase_interface.calculator import calculator_from_model

from ase.io.proteindatabank import read_proteindatabank

from mace.calculators import mace_off
from ase import units

import os

import numpy as np

# This file evaluates HIP-HOP-NN evaluation times (per atom) of different model sizes on a chosen molecule benchmark.
# It does the same thing as bench_contributions_ablation.py, except that it uses the evaluator included in hippynn, rather
# than using the ASE calculator.

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():

    # This cannot be changed by command line arguments.
    nu_and_b_list = [(20,128), (20,200)]

    if len(sys.argv) == 1:
        # I find it easier to set parameters as variables than to pass them in as command line arguments.

        # l_max and n_max are as they are in HopInvariantsLayer.py
        l_max=3
        n_max=4

        num_spin_up = 3 # How many times we run the calculator on each molecule before evaluating to get spin-up times (compiling etc.) out of the way.
        num_repeat = 10 # how many times we evaluate the timing on each molecule.
        size_thresh = 10000 # only time molecules that have less atoms than this threshold. Set this equal to -1 to time all molecules in the data folder.

        # list of (sensitivites, features) combinations that we want to test out.
        # nu_and_b_list = [(20,128), (20, 200), (20,256), (20,300), (20,400), (20,512), (40,128), (40,200), (40,256), (40,300)]

        out_file = "../results/time_all_evaluator.csv" # where should we write the results
        data_folder = "../data/large_molecule_npz" # folder containing the molecules that we are testing (in .pdb format)

        suppress_model_creation_prints = False # Whether we want to suppress the print statements from creating models that say things like "determined inputs" etc.
        warnings.filterwarnings("ignore") # uncomment this line in order to ignore warnings that are coming from HIP-HOP-NN (the main warning that gets printed out is saying that it is in a beta stage)
    else:
        parser = argparse.ArgumentParser(prog="HIP-HOP-NN large molecule benchmark with evaluator.", description="Runs ablation testing on" \
        "the various contributions that have been made to HIP-HOP-NN using the evaluator.")
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

    # iterate over different model sizes
    for nu,b in nu_and_b_list:

        # iterate over different molecules in the data folder
        for f in os.listdir(data_folder):

            if not os.path.isfile(f"{data_folder}/{f}"):
                continue

            # Load in the numpy arrays needed for the evaluator, and 
            print(f"{data_folder}/{f}")
            npz = np.load(f"{data_folder}/{f}")
            arr_dict_input = {}
            for k in npz:
                if npz[k].dtype == np.float64:
                    arr_dict_input[k] = npz[k].astype(np.float32)
                else:
                    arr_dict_input[k] = npz[k]

            # get the number of atoms in the molecule and check that it is below our current size threshold.
            size = arr_dict_input['atomic_numbers'].shape[1]
            if size_thresh == -1 or size < size_thresh:
                times = {} # stores all of the times that we measure

                # set up the db_info needed for the evaluator
                db_info = {'inputs': ['R', 'Z'], 'targets': ['F', 'T', 'Z']}

                db = hippynn.databases.Database(
                    arr_dict_input,
                    seed = 0,
                    num_workers=0,
                    pin_memory=True,
                    allow_unfound=True,
                    **db_info,
                )
                db.arr_dict['Z'] = db.arr_dict['atomic_numbers']
                db.arr_dict['coordinates'] = db.arr_dict['R']
                db.arr_dict['F'] = np.zeros_like(db.arr_dict['R'])
                db.split_the_rest("test")

                split_arrdict = db.splits['test']

                # Ablation testing:
                # Each tuple specifies which of my optimizations are implemented comapred to the default.
                # The first entry is set to True to use polynomial invariants, and False to use default inariants.
                # The second entry is set to True to use the tensor message passing implementation for the forward pass, and False otherwise.
                # The third entry is set to True to use the tensor message passing implementation for the gradient computation, and False otherwise.

                # Throughout these trials, we run m[1].rattle(1e-6). This slightly perturbs the molecules with a standard deviation of 1e-6.
                # In doing so, the molecules are slightly shifted, which means that the results of the calculator will not be cached. On the other
                # hand, because the perturbation is so small, the graph will not need to be rebuilt. This allows us to actually re-run the forward pass
                # (just without the graph being rebuilt).

                for config in [(False,False,False),(True,False,False),(False,False,True),(True,False,True),(False,True,True),(True,True,True)]:
                    times_for_this_config = []

                    if suppress_model_creation_prints:
                        sys.stdout = open(os.devnull, "w")
                    model, _ = make_model(nu, b, l_max, n_max, *config)
                    sys.stdout = stdout

                    predictor = hippynn.graphs.Predictor.from_graph(model,model_device='cuda')

                    input_names = [x.db_name for x in predictor.graph.input_nodes]
                    dict_inputs = {k: split_arrdict[k] for k in input_names}

                    # spin-up timing (including compilation)
                    for x in range(num_spin_up):
                        predictor(**dict_inputs, batch_size=1)

                    # run the real evaluation
                    for x in range(num_repeat):
                        torch.cuda.synchronize()
                        t1 = time.time()
                        predictor(**dict_inputs, batch_size=1)
                        torch.cuda.synchronize()
                        t2 = time.time()

                        times[(x,*config)] = t2-t1
                        times_for_this_config.append(t2-t1)
                    
                    med_time = statistics.median(times_for_this_config)
                    print(f"{f} ({size}) [{config}]: {med_time}                                          ")

                # write results of the current trials to the file.
                for x in range(num_repeat):
                    writer.writerow([ l_max, n_max, nu, b, f, times[(x,False,False,False)], times[(x,True,False,False)], times[(x,False,False,True)], times[(x,True,False,True)], times[(x,False,True,True)], times[(x,True,True,True)]] )

                outf.flush()




if __name__ == "__main__":
    main()