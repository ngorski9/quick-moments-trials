import os
import hippynn
import numpy as np
import torch
import csv
import time
import sys
import argparse

# this file benchmarks the individual time savings associated with the polynomial kernel
# for computing the invariants. It runs the tests using the same molecule sizes that are
# in the large molecule benchmark.

if __name__ == "__main__":

    n_max_l_max_combinations = [(1,0), (2,1), (2,2), (2,3), (3,2), (3,3), (4,2), (4,3)]
    feats = [128]

    if len(sys.argv) == 1:
        size_thresh = 10000
        num_repeat = 10

        out_file = "../results/polynomial_ablation.csv"
        data_folder = "../data/large_molecule_npz" # you must point this to the same data used by the evaluator (in NPZ format)
    else:
        parser = argparse.ArgumentParser(prog="HIP-HOP-NN polynomial kernel time testing")

        parser.add_argument("-thresh", help="The size of the largest molecule to simulate", type=float, default=float('inf'))
        parser.add_argument("-n_repeat", help="How many times to time each test", type=int, default=10)

        parser.add_argument("-csv", help="the csv file where the results should be written", required=True)
        parser.add_argument("-i", help="The folder containing the molecules (in npz format)", required=True)

        args = parser.parse_args(sys.argv[1:])

        size_thresh = args.thresh
        num_repeat = args.n_repeat

        out_file = args.csv
        data_folder = args.i

    # END OF PARAMETERS

    cols_per_l_max = {
        0 : 1,
        1 : 4,
        2 : 9,
        3 : 16
    }

    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["molecule", "# atoms", "# feat", "n_max", "l_max", "time forward torch", "time forward triton", "time backward torch", "time backward triton"])

    molecules = []
    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            npz = np.load(f"{data_folder}/{f}")
            molecules.append((f,npz['atomic_numbers'].shape[1]))

    molecules.sort(key = lambda i : i[1])

    for m in molecules:
        for b in feats:
            for n_max, l_max in n_max_l_max_combinations:
                for x in range(num_repeat):
                    torch.manual_seed(0)

                    n_cols = cols_per_l_max[l_max]

                    warmup_matrix = torch.randn(( m[1]*b, n_cols ), requires_grad=True, device='cuda')
                    real_matrix = torch.randn(( m[1]*b, n_cols ), requires_grad=True, device='cuda')

                    layer = hippynn.layers.hiplayers.invariants.HopInvariantLayer(n_max, l_max).to('cuda')

                    sample_out = layer(warmup_matrix)

                    warmup_grad_out = torch.randn(sample_out.shape, requires_grad=True, device='cuda')
                    real_grad_out = torch.randn(sample_out.shape, requires_grad=True, device='cuda')

                    results = {} # key will be a tuple of (use_polynomials,forward), value is the time.

                    for use_polynomials in [True, False]:

                        hippynn.settings.USE_POLYNOMIAL_INVARIANTS = use_polynomials

                        # warmup
                        warmup_out = layer(warmup_matrix)
                        torch.autograd.grad(warmup_out, warmup_matrix, warmup_grad_out)

                        # real evaluation

                        torch.cuda.synchronize()
                        t1 = time.time()
                        real_out = layer(real_matrix)
                        torch.cuda.synchronize()
                        t2 = time.time()

                        results[use_polynomials,True] = t2-t1

                        torch.cuda.synchronize()
                        t1 = time.time()
                        torch.autograd.grad(real_out, real_matrix, real_grad_out)
                        torch.cuda.synchronize()
                        t2 = time.time()

                        results[use_polynomials,False] = t2-t1

                    writer.writerow([m[0], m[1], b, n_max, l_max, results[False,True], results[True,True], results[False,False], results[True,False]])
                    outf.flush()
                    print([m[0], m[1], b, n_max, l_max, results[False,True], results[True,True], results[False,False], results[True,False]])