import os
import time
import statistics
import torch
import csv
import numpy as np
import warnings
import sys
import argparse

from model_maker import make_model

from ase.io.proteindatabank import read_proteindatabank

from ase import units

import os

# This evaluates the computation time of MACE-OFF small, medium, and large, along with HIP-HOP-NN.
# HIP-HOP-NN is evaluated both with and without evlauations implemented.

# Uncomment the line below in order to print out the autotuning results (can be nice for knowing how far into autotuning you are)
# os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():

    # fill out this list with pairs of a calculator and a molecule to prevent that calculator from being
    # evaluated on that molecule. This may be necessary if a certain calculator will run out of memory
    # on certain molecules.
    exclude_evaluation = [
        ("large", "dhfr"),
        ("large", "factorIX"),
        ("medium", "factorIX"),
    ]

    if len(sys.argv) == 1:
        # l_max and n_max are as they are in HopInvariantsLayer.py
        l_max=3
        n_max=4

        # Name of calculator:
        # Set this equal to 'mace-small,' 'mace-medium,' or 'mace-large' to evaluate MACE-OFF
        # Set this equal to 'PET-MAD' to evaluate PET-MAD
        # Set this equal to the matPES model type (either M3GNet, CHGNet, or TensorNet)
        # small, medium, or large, respectively. Otherwise, give a tuple of (sense,feat,optimized (bool)) for hippynn.
        calc_name = (20,300,True)

        # NOTE: for MACE-OFF the number for spinning up needs to be a bit higher than for HIP-HOP-NN (5 should be ok)
        num_spin_up = 5 # How many times we run the calculator on each molecule before evaluating to get spin-up times (compiling etc.) out of the way.
        num_repeat = 10 # how many times we evaluate the timing on each molecule.
        size_thresh = 10000 # only time molecules that have less atoms than this threshold. Set this equal to -1 to time all molecules in the data folder.

        out_file = "../time_mace.csv" # where should we write the results.
        data_folder = "../data/large_molecule_pdb" # folder containing the molecules that we are testing (in .pdb format)
    else:
        parser = argparse.ArgumentParser(prog="HIP-HOP-NN ablation timing with ASE calculator")

        parser.add_argument("-l_max", help="The maximum tensor order used in invariants", type=int, default=3)
        parser.add_argument("-n_max", help="The maximum number of tensors used in an invariant", type=int, default=4)

        parser.add_argument("-n_spinup", help="How many times do you spin up each evaluator before it runs.", type=int, default=3)
        parser.add_argument("-n_repeat", help="How many times do you time each molecule.")
        parser.add_argument("-thresh", help="Only benchmark on molecules with less atoms than this", type=float, default=float('inf'))

        parser.add_argument("-calc_name", help="Enter the name of the calculator (see code comments for a list).", required=True)

        parser.add_argument("-csv", help="The location of the output csv file", required=True)
        parser.add_argument("-i", help="The location of the large molecule files (in npz format).", required=True)

        args = parser.parse_args(sys.argv[1:])

        l_max = args.l_max
        n_max = args.n_max

        num_spin_up = args.n_spinup
        num_repeat = args.n_repeat
        size_thresh = args.thresh

        out_file = args.csv
        data_folder = args.i

        calc_name = args.calc_name

        if "(" in args.calc_name:
            num_comma = calc_name.count(",")
            assert num_comma == 2, "invalid calculator name " + calc_name
            comma1 = calc_name.find(",")
            comma2 = calc_name.rfind(",")
            nu = int(calc_name[1:comma1])
            b = int(calc_name[comma1+1:comma2])
            optimized = bool(calc_name[comma2+1:])
            calc_name = (nu,b,optimized)

    # END OF PARAMETERS #

    # open output file and write header.
    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["molecule", "calculator", "time"])

    # load in the molecules from the data folder and load them in using ase.
    # This creates an array, test_molecules. Each entry is a tuple of the
    # molecule's filename, and then an object of the molecule.
    test_molecules = []
    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            inf = open(f"{data_folder}/{f}","r")
            a = read_proteindatabank(inf,index=0)
            a.positions = a.positions.astype(np.float32)
            test_molecules.append((f,a))

    if calc_name in ["mace-small", "mace-medium", "mace-large"]:
        from mace.calculators import mace_off
        calc = mace_off(model=calc_name[5:],dispersion=False, dtype=torch.float32)
    elif calc_name == "PET-MAD":
        from pet_mad.calculator import PETMADCalculator
        calc = PETMADCalculator(version="latest", device="cuda")
    elif calc_name in ["M3GNet", "CHGNet", "TensorNet"]:
        import matgl
        from matgl.ext.ase import PESCalculator       
        # populate this with the latest matPES version:
        matpes_version = "MatPES-PBE-v2025.1"
        model = matgl.load_model(f"{calc_name}-{matpes_version}-PES")
        calc = PESCalculator(model)
    else:
        from hippynn.interfaces.ase_interface.calculator import calculator_from_model
        model, _ = make_model(calc_name[0], calc_name[1], l_max, n_max, calc_name[2], calc_name[2], calc_name[2])
        calc = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')

    # Iterate through molecules and calculators.
    # To see why we use .rattle, see the comments for bench_all.py.
    for m in test_molecules:

        if size_thresh == -1 or len(m[1]) < size_thresh:
            times = {}

            calc_times = []

            # skip excluded evaluations.
            skip_this = False
            for calc_name_, mol in exclude_evaluation:
                if calc_name_ == calc_name and mol in m[0]:
                    skip_this = True
            
            if skip_this:
                continue

            m[1].calc = calc

            # Run the calculator a certain number of times without timing it to spin up.
            print(f"{m[0]} ({len(m[1])}) [{calc_name}]: spinning up...",end="\r")
            for repeat in range(num_spin_up):
                m[1].rattle(1e-6)
                e = m[1].get_total_energy()
                f = m[1].get_forces()

            # Time the calculator evaluation
            for x in range(num_repeat):
                print(f"{m[0]} ({len(m[1])}) [{calc_name}]: {x+1} / {num_repeat}                                                     ",end="\r")

                m[1].rattle(1e-6)

                torch.cuda.synchronize()
                t1 = time.time()

                e = m[1].get_total_energy()
                f = m[1].get_forces()

                torch.cuda.synchronize()
                t2 = time.time()

                t_peratom = (t2-t1) / len(m[1])

                calc_times.append( t_peratom )

            med_time = statistics.median(calc_times)
            print(f"{m[0]} ({len(m[1])}) [{calc_name}]: {med_time}                                          ")
            for t in calc_times:
                writer.writerow([m[0], calc_name, t])

        outf.flush()






if __name__ == "__main__":
    main()