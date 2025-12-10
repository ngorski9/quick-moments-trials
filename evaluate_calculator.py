import ase
from ase import units
import numpy as np
import math
import csv
import os
import sys
import argparse

from ase import units

import torch

# This file evaluates the accuracy of an ASE calculator (either MACE-OFF or ours)
# on the test data that is from MACE. It compute the RMSE of both energy and force
# computation on the mace test data, and disaggregates the results based on
# the molecule type (as is done in the MACE-OFF paper). It also computes the average
# molecule size per category.
if __name__ == "__main__":

    # populate this with the latest matPES version:
    matpes_version = "MatPES-PBE-v2025.1"

    if len(sys.argv) == 1:
        # set to an integer to stop evaluation once that many molecules are reached.
        # For example, if this is set to 100, only the first 100 molecules will be
        # evaluated. Primarily used for debugging.
        # Otherwise, set this equal to float('inf).
        kill_early = float('inf')
        
        # Set this equal to 'mace-small,' 'mace-medium,' or 'mace-large' to evaluate MACE-OFF
        # Set this equal to 'PET-MAD' to evaluate PET-MAD
        # Set this equal to the matPES model type (either M3GNet, CHGNet, or TensorNet)

        # corresponding to a hippynn calculator that you would like to evaluate.
        calc_name = "PET-MAD"

        # What file should we store the trial's results to. If the file already exists, it will
        # append the results as another row of the file. Otherwise, it will write the header and data.
        out_file = "../results.csv"

        # The data should all be formatted using the npy format for hippynn 
        # (see the documentation) in one folder. The prefix should specify
        # the folder and names of the files. In the examples given here,
        # the data files are '../testData/mace-test-data-Z.npy" etc.

        # The categories are identified by a number (in a numpy array)
        # The names of the categories should be stored in (data_prefix)-Category-Names.txt
        data_prefix = "../data/mace_test_data/mace"
    else:

        parser = argparse.ArgumentParser(prog="HIP-HOP-NN ablation timing with ASE calculator")

        parser.add_argument("-early_stop", help="should you stop evaluating after a certain number of molecules", type=float, default=float('inf'))
        parser.add_argument("-calc_name", help="Name of the calculator to evaluate. See the comments for more details.", required=True)
        parser.add_argument("-csv", help="csv file specifying where to write the results.", required=True)
        parser.add_argument("-i", help="data prefix that specifies the folder and filename patterns of the input data. See the comments for more details.", required=True)

        args = parser.parse_args(sys.argv[1:])

        kill_early = args.early_stop
        if kill_early != float('inf'):
            kill_early = int(kill_early)
        calc_name = args.calc_name
        out_file = args.csv
        data_prefix = args.i

    # END OF PARAMETERS #

    species = np.load(f"{data_prefix}-Z.npy")
    location = np.load(f"{data_prefix}-R.npy")
    energy = np.load(f"{data_prefix}-T.npy")
    forces = np.load(f"{data_prefix}-F.npy")
    categories = np.load(f"{data_prefix}-C.npy")
    if os.path.isfile(f"{data_prefix}-CELL.npy"):
        cells = np.load(f"{data_prefix}-CELL.npy")
    else:
        cells = None

    num_categories = np.max(categories)+1

    # Load in the calculator:
    if calc_name in ["mace-small", "mace-medium", "mace-large"]:
        from mace.calculators import mace_off
        calc = mace_off(model=calc_name[5:],dispersion=False, dtype=torch.float32)
    elif calc_name == "PET-MAD":
        from pet_mad.calculator import PETMADCalculator
        calc = PETMADCalculator(version="latest", device="cuda")
    elif calc_name in ["M3GNet", "CHGNet", "TensorNet"]:
        import matgl
        from matgl.ext.ase import PESCalculator
        model = matgl.load_model(f"{calc_name}-{matpes_version}-PES")
        calc = PESCalculator(model)
    else:
        from hippynn.experiment.serialization import load_model_from_cwd
        from hippynn.interfaces.ase_interface.calculator import calculator_from_model

        # change directory to the hippynn model that we are loading.
        # then load from cwd and change the directory back.
        cwd = os.getcwd()
        os.chdir(calc_name)
        model = load_model_from_cwd()
        calc = calculator_from_model(model, en_unit = units.eV, dist_unit = units.Ang).to('cuda')
        os.chdir(cwd)

    # Store results by category.
    mol_sizes = [0] * num_categories
    e_se_peratom = [0] * num_categories
    e_ae_peratom = [0] * num_categories
    f_se_peratom = [0] * num_categories
    f_ae_peratom = [0] * num_categories
    n_mols_eval = [0] * num_categories
    total_e_se_peratom = 0
    total_e_ae_peratom = 0
    total_f_se_peratom = 0
    total_f_ae_peratom = 0
    total_n_mols = 0
    total_n_atoms = 0

    num_mols_to_test = min(species.shape[0], kill_early)

    # iterate through molecules.
    for i in range(num_mols_to_test):
        print(f"{i+1} / {num_mols_to_test}                      ", end="\r")
        # pull relevant information
        gt_energy = energy[i]
        gt_forces = forces[i]
        category = categories[i]

        # Get species and locations in list form in order to create ASE object.
        species_list = list(species[i])
        if 0.0 in species_list:
            num_atoms = species_list.index(0.0)
        else:
            num_atoms = len(species_list)
        species_list = species_list[:num_atoms]

        location_list = location[i,:num_atoms,:]

        # Make ase object and set calculator
        if cells is not None:
            atoms = ase.Atoms(species_list, location_list, pbc=[True,True,True], cell=cells[i])
        else:
            atoms = ase.Atoms(species_list, location_list)

        atoms.calc = calc

        # Predict force and energy
        pred_energy = atoms.get_total_energy()
        pred_forces = atoms.get_forces()

        # Remove padding from gt_forces
        gt_forces = gt_forces[:num_atoms,:]

        # Update information to lists.
        mol_sizes[category] += num_atoms
        e_se_peratom[category] += ((pred_energy-gt_energy)/num_atoms)**2
        e_ae_peratom[category] += abs(pred_energy-gt_energy)/num_atoms
        total_e_se_peratom += ((pred_energy-gt_energy)/num_atoms)**2
        total_e_ae_peratom += abs(pred_energy-gt_energy)/num_atoms
        total_n_mols += 1

        f_se_peratom[category] += np.sum(((pred_forces-gt_forces)/(num_atoms))**2)
        total_f_se_peratom += np.sum(((pred_forces-gt_forces)/(num_atoms))**2)
        f_ae_peratom[category] += np.sum(np.abs(pred_forces-gt_forces))/(3*num_atoms)
        total_f_ae_peratom += np.sum(np.abs(pred_forces-gt_forces))/(3*num_atoms)
        n_mols_eval[category] += 1

    # Load in the category names for evaluation.
    category_names = []
    category_names_file = open(f"{data_prefix}-Category-Names.txt", "r")
    for name in category_names_file:
        category_names.append(name.strip())

    # Compute the RMSEs and print results.
    file_header = ["model name"]
    if calc_name in ["small", "medium", "large"]:
        file_results = ["MACE-OFF " + calc_name]
    else:
        file_results = [calc_name]
    if kill_early != float('inf'):
        file_results[0] += f" ({kill_early})"

    for i in range(num_categories):
        if n_mols_eval[i] == 0:
            continue

        avg_mol_size = mol_sizes[i] / n_mols_eval[i]
        e_mae_peratom = e_ae_peratom[i] / n_mols_eval[i]
        e_rmse_peratom = math.sqrt(e_se_peratom[i] / n_mols_eval[i])
        f_mae_peratom = f_ae_peratom[i] / n_mols_eval[i]
        f_rmse_peratom = math.sqrt(f_se_peratom[i] / (3*n_mols_eval[i]))

        print(f"{category_names[i]} ({n_mols_eval[0]} molecules @ {avg_mol_size} atoms (on avg))")
        print(f"E RMSE (eV): {e_rmse_peratom}")
        print(f"E MAE (eV): {e_mae_peratom}")
        print(f"F RMSE (eV/Å): {f_rmse_peratom}")
        print(f"F MAE (eV/Å): {f_mae_peratom}")

        file_header.append(f"{category_names[i]} E RMSE (eV)")
        file_header.append(f"{category_names[i]} E MAE (eV)")
        file_header.append(f"{category_names[i]} F RMSE (eV/Å)")
        file_header.append(f"{category_names[i]} F MAE (eV/Å)")
        file_header.append(f"{category_names[i]} Avg Mol. Size")

        file_results.append(e_rmse_peratom)
        file_results.append(e_mae_peratom)
        file_results.append(f_rmse_peratom)
        file_results.append(f_mae_peratom)
        file_results.append(avg_mol_size)

        print()

    e_mae = total_e_ae_peratom / total_n_mols
    e_rmse = math.sqrt(total_e_se_peratom / total_n_mols)
    f_mae = total_f_ae_peratom / total_n_mols
    f_rmse = math.sqrt(total_f_se_peratom / total_n_mols)

    print(f"Overall E RMSE (eV) : {e_rmse}")
    print(f"Overall E MAE (eV) : {e_mae}")
    print(f"Overall F RMSE (eV/Å) : {f_rmse}")
    print(f"Overall F MAE (eV/Å) : {f_mae}")

    # Write results to file.
    write_header = not os.path.isfile(out_file)
    outf = open(out_file, "a")
    writer = csv.writer(outf)
    if write_header:
        writer.writerow(file_header)
    writer.writerow(file_results)
    outf.close()    
