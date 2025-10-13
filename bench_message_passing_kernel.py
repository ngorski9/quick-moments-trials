import os
import hippynn
import numpy as np
import torch
import csv
import time

if __name__ == "__main__":

    t_nu_b_combinations = [(16,20,128)]
    size_thresh = 10000
    num_repeat = 10

    # you must point this to a folder containing numpy arrays where the first row is pair_first and the second is pair_second
    data_folder = "../data/large_molecule_pairs" 
    out_file = "../results/message_passing_ablation.csv"

    # end of parameters

    outf = open(out_file, "a")
    writer = csv.writer(outf)
    writer.writerow(["molecule", "# atoms", "# pairs", "# tensor comp.", "# sense", "# feat", "time forward torch", "time forward triton", "time backward torch", "time backward triton"])

    molecules = []

    for f in os.listdir(data_folder):
        if os.path.isfile(f"{data_folder}/{f}"):
            array = np.load(f"{data_folder}/{f}")
            underscore_idx = f.rfind("_")
            num_atoms = int(f[underscore_idx+1:-4])
            molecules.append((f,array,num_atoms))
    
    for m in molecules:
        for t, nu, b in t_nu_b_combinations:
            for x in range(num_repeat):
                torch.manual_seed(0)

                pair_first = torch.Tensor(m[1][0]).to('cuda').to(torch.int64)
                pair_second = torch.Tensor(m[1][1]).to('cuda').to(torch.int64)

                n_pairs = pair_first.shape[0]
                n_atoms = m[2]

                T_warmup = torch.randn((n_pairs,t), device='cuda', requires_grad=True)
                s_warmup = torch.randn((n_pairs,nu), device='cuda', requires_grad=True)
                z_warmup = torch.randn((n_atoms,b), device='cuda', requires_grad=True)
                grad_output_warmup = torch.randn((n_atoms,t*nu,b), device='cuda', requires_grad=True)

                T_real = torch.randn((n_pairs,t), device='cuda', requires_grad=True)
                s_real = torch.randn((n_pairs,nu), device='cuda', requires_grad=True)
                z_real = torch.randn((n_atoms,b), device='cuda', requires_grad=True)
                grad_output_real = torch.randn((n_atoms,t*nu,b), device='cuda', requires_grad=True)

                results = {} # key will be a tuple of (use_kernel,forward), value is the time.

                for use_kernel in [True, False]:

                    hippynn.settings.USE_TENSOR_MESSAGE_PASSING = use_kernel

                    # warm up
                    output_warmup = hippynn.custom_kernels.hopMessagePassing(T_warmup, s_warmup, z_warmup, pair_first, pair_second)
                    grad_warmup = torch.autograd.grad(output_warmup, (T_warmup, s_warmup, z_warmup), grad_output_warmup)

                    # real timing:
                    torch.cuda.synchronize()
                    t1 = time.time()
                    output_real = hippynn.custom_kernels.hopMessagePassing(T_real, s_real, z_real, pair_first, pair_second)
                    torch.cuda.synchronize()
                    t2 = time.time()

                    results[(use_kernel,True)] = t2-t1

                    torch.cuda.synchronize()
                    t1 = time.time()
                    grad_real = torch.autograd.grad(output_real, (T_real, s_real, z_real), grad_output_real)
                    torch.cuda.synchronize()
                    t2 = time.time()

                    results[(use_kernel,False)] = t2-t1

                writer.writerow([m[0], n_atoms, n_pairs, t, nu, b, results[False,True], results[True,True], results[False,False], results[True,False]])
                outf.flush()
                print([m[0], n_atoms, n_pairs, t, nu, b, results[False,True], results[True,True], results[False,False], results[True,False]])