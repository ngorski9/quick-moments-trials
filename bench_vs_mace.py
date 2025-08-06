import os
import ase
import time
import statistics
import torch
import csv

from model_maker import make_model

from hippynn.interfaces.ase_interface.calculator import calculator_from_model

from ase.io.proteindatabank import read_proteindatabank

from mace.calculators import mace_off
from ase import units

import os

# This evaluates the computation time of MACE-OFF small, medium, and large, along with HIP-HOP-NN.
# HIP-HOP-NN is evaluated both with and without evlauations implemented.

# Uncomment the line below in order to print out the autotuning results (can be nice for knowing how far into autotuning you are)
# os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():

    # l_max and n_max are as they are in HopInvariantsLayer.py
    l_max=3
    n_max=4

    # number of sensitivities and features used when evuating HIP-HOP-NN
    nu = 20
    b = 300

    # NOTE: for MACE-OFF the number for spinning up needs to be a bit higher than for HIP-HOP-NN (5 should be ok)
    num_spin_up = 5 # How many times we run the calculator on each molecule before evaluating to get spin-up times (compiling etc.) out of the way.
    num_repeat = 10 # how many times we evaluate the timing on each molecule.
    size_thresh = 10000 # only time molecules that have less atoms than this threshold. Set this equal to -1 to time all molecules in the data folder.

    out_file = "../results/time_mace.csv" # where should we write the results
    data_folder = "../bench_data" # folder containing the molecules that we are testing (in .pdb format)

    # calculator names: gt: HIP-HOP-NN PYPI implementation. optimized: HIP-HOP-NN optimized implementation.
    #                   small: MACE-OFF small. medium: MACE-OFF medium. large: MACE-OFF large.

    # fill out this list with pairs of a calculator and a molecule to prevent that calculator from being
    # evaluated on that molecule. This may be necessary if a certain calculator will run out of memory
    # on certain molecules.
    exclude_evaluation = [
        ("large", "dhfr"),
        ("large", "factorIX"),
        ("medium", "factorIX"),
    ]

    # END OF PARAMETERS #

    # open output file and write header.
    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["molecule", "time gt", "time optimized", "time small", "time medium", "time large"])

    # load in the molecules from the data folder and load them in using ase.
    # This creates an array, test_molecules. Each entry is a tuple of the
    # molecule's filename, and then an object of the molecule.
    test_molecules = []
    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            inf = open(f"{data_folder}/{f}","r")
            a = read_proteindatabank(inf,index=0)
            test_molecules.append((f,a))

    # Stores calculators as tuples of (calc object, name)
    calcs = []

    model, _ = make_model(20, 300, l_max, n_max, True, True, True)
    c = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')
    calcs.append((c,"optimized"))

    model, _ = make_model(20, 300, l_max, n_max, False, False, False)
    c = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')
    calcs.append((c,"gt"))

    calcs.append((mace_off(model="small",dispersion=False, dtype=torch.float32), "small"))
    calcs.append((mace_off(model="medium",dispersion=False, dtype=torch.float32), "medium"))
    calcs.append((mace_off(model="large",dispersion=False, dtype=torch.float32), "large"))

    # Iterate through molecules and calculators.
    # To see why we use .rattle, see the comments for bench_all.py.
    for m in test_molecules:

        if size_thresh == -1 or len(m[1]) < size_thresh:
            times = {}

            for c in calcs:

                calc_times = [] # used only for printing.

                m[1].calc = c[0]

                # Run the calculator a certain number of times without timing it to spin up.
                print(f"{m[0]} ({len(m[1])}) [{c[1]}]: spinning up...",end="\r")
                for repeat in range(num_spin_up):
                    m[1].rattle(1e-6)
                    e = m[1].get_potential_energy()
                    f = m[1].get_forces()

                # Time the calculator evaluation
                for x in range(num_repeat):
                    print(f"{m[0]} ({len(m[1])}) [{c[1]}]: {x+1} / {num_repeat}",end="\r")

                    m[1].rattle(1e-6)

                    torch.cuda.synchronize()
                    t1 = time.time()

                    e = m[1].get_potential_energy()
                    f = m[1].get_forces()

                    torch.cuda.synchronize()
                    t2 = time.time()

                    t_peratom = (t2-t1) / len(m[1])

                    calc_times.append( t_peratom )
                    times[(x,c[1])] = t_peratom

                med_time = statistics.median(calc_times)
                print(f"{m[0]} ({len(m[1])}) [{c[1]}]: {med_time}                                          ")
            
            for x in range(num_repeat):
                writer.writerow([ m[0], times[(x,"gt")], times[(x,"optimized")], times[(x,"small")], times[(x,"medium")], times[(x,"large")] ] )

            outf.flush()






if __name__ == "__main__":
    main()