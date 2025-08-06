import os
import ase
import time
import statistics
import torch
import csv

from model_maker import make_model

import hippynn
from hippynn.interfaces.ase_interface.calculator import calculator_from_model

from ase.io.proteindatabank import read_proteindatabank

from mace.calculators import mace_off
from ase import units

import os

import numpy as np


# THIS SCRIPT IS CURRENTLY UNDERGOING REVISION #
# DO NOT USE THIS FOR NOW #


os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

def main():
    out_file = "./results/time_all_alice.csv"
    data_folder = "bench_data_alice"

    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["l_max", "n_max", "nu", "b", "molecule", "run", "time default", "time polys", "time grad fused", "time polys grad fused", "time env+grads fused", "time polys env+grads fused"])

    l_max=3
    n_max=4

    num_repeat = 100
    num_compile = 1

    for nu in [20]:#[20,40]:
        for b in [128]:#[128,200,256,300,400,512]:

            model, _ = make_model(nu, b, l_max, n_max, True, False, False)
            model = model.to('cuda')
            weights = model.state_dict()

            for f in os.listdir(data_folder):
                # outf2 = open("debug.txt", "a")
                # outf2.write(str(t2-t1) + "\n")
                # outf2.close()

                if "testosterone" not in f:
                    continue
                if "stmv" in f:
                    continue
                if not os.path.isfile(f"{data_folder}/{f}"):
                    continue

                npz = np.load(f"{data_folder}/{f}")
                arr_dict_input = {}
                for k in npz:
                    print(k)
                    print(npz[k].shape)
                    if npz[k].dtype == np.float64:
                        arr_dict_input[k] = npz[k].astype(np.float32)
                    else:
                        arr_dict_input[k] = npz[k]

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
                split_name = 'test'

                models = {}

                for x in range(num_repeat):

                    config_times = {}

                    for config in [(True,False,False)]:#[(False,False,False)],(True,False,False),(False,False,True),(True,False,True),(False,True,True),(True,True,True)]:
                        print(config)
                        model, _ = make_model(nu, b, l_max, n_max, *config)
                        # weights2 = model.state_dict()
                        # for k in weights:
                        #     if k in weights2:
                        #         weights2[k] = weights[k]

                        model.load_state_dict(weights)

                        predictor = hippynn.graphs.Predictor.from_graph(model,model_device='cuda')
                        predictor = hippynn.graphs.Predictor(predictor.inputs,predictor.outputs[:2],model_device=hippynn.tools.device_fallback())

                        input_names = [x.db_name for x in predictor.graph.input_nodes]
                        dict_inputs = {k: split_arrdict[k] for k in input_names}

                        times = {}

                        # compile before timing (if necessary)
                        predictor(**dict_inputs, batch_size=1)

                        # run the real evalaution
                        torch.cuda.synchronize()
                        t1 = time.time()
                        predictor(**dict_inputs, batch_size=1)
                        torch.cuda.synchronize()
                        t2 = time.time()

                    config_times[config] = t2-t1
                    print(t2-t1)
                    # outf2 = open("debug.txt", "a")
                    # outf2.write(str(t2-t1) + "\n")
                    # outf2.close()
                
                    # writer.writerow([ l_max, n_max, nu, b, f, x, config_times[(False,False,False)], config_times[(True,False,False)], config_times[(False,False,True)], config_times[(True,False,True)], config_times[(False,True,True)], config_times[(True,True,True)] ] )
                    outf.flush()






if __name__ == "__main__":
    main()

# alanine_dipeptide.pdb (22): 0.0011365305293690074                                          
# chignolin.pdb (166): 0.0001635364739291639                                          
# dhfr.pdb (2489): 8.887466823015409e-05                                          
# factorIX.pdb (5807): 8.627240128580378e-05                                          
# testosterone.pdb (49): 0.0005140085609591737 