import os
import ase
import time
import torch
import csv

from model_maker import make_model

from hippynn.interfaces.ase_interface.calculator import calculator_from_model

from ase.io.proteindatabank import read_proteindatabank

from mace.calculators import mace_off
from ase import units

import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():
    out_file = "./results/time_all.csv"
    data_folder = "bench_data"

    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["l_max", "n_max", "nu", "b", "molecule", "time default", "time polys", "time grad fused", "time polys grad fused", "time env+grads fused", "time polys env+grads fused"])

    l_max=3
    n_max=4

    num_repeat = 10
    size_thresh = 10000

    test_molecules = []
    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            inf = open(f"{data_folder}/{f}","r")
            a = read_proteindatabank(inf,index=0)
            test_molecules.append((f,a))

    for nu in [20,40]:
        for b in [128,200,256,300,400,512]:

            for m in test_molecules:

                if size_thresh == -1 or len(m[1]) < size_thresh:
                    config_times = {}

                    for config in [(False,False,False)]:#,(True,False,False),(False,False,True),(True,False,True),(False,True,True),(True,True,True)]:

                        times = []

                        model, _ = make_model(nu, b, l_max, n_max, *config)
                        c = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')
                        m[1].calc = c

                        print(f"{m[0]} ({len(m[1])}) [{config}]: 0 / {num_repeat}",end="\r")
                        for repeat in range(3):
                            m[1].rattle(1e-6)
                            e = m[1].get_potential_energy()
                            f = m[1].get_forces()

                        for x in range(num_repeat):
                            print(f"{m[0]} ({len(m[1])}) [{config}]: {x+1} / {num_repeat}",end="\r")

                            m[1].rattle(1e-6)

                            torch.cuda.synchronize()
                            t1 = time.time()

                            e = m[1].get_potential_energy()
                            f = m[1].get_forces()

                            torch.cuda.synchronize()
                            t2 = time.time()
                            print(t2-t1)

                            times.append(t2-t1)

                        med_time = statistics.median(times) / len(m[1])
                        print(f"{m[0]} ({len(m[1])}) [{config}]: {med_time}                                          ")
                        config_times[config] = med_time
                    
                    writer.writerow([ l_max, n_max, nu, b, m[0], config_times[(False,False,False)], config_times[(True,False,False)], config_times[(False,False,True)], config_times[(True,False,True)], config_times[(False,True,True)], config_times[(True,True,True)]] )
                    outf.flush()

if __name__ == "__main__":
    main()