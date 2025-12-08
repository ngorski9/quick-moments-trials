import argparse
import torch
import hippynn
import numpy as np
import os

# This file is used for creating and training HIP-HOP-NN models. It only works with a hippynn.DirectoryDatabase
# for the data, although this could be expanded to other types of databases as needed.

# Returns the henergy, force and species nodes from a given tensor model type with given n_max and l_max.
def get_primary_nodes(network_params, tensor_model, l_max, n_max, atomization_consistent, db_dir, db_name):
    from hippynn.graphs import inputs, networks, targets, physics

    if l_max == 0 or tensor_model == "NONE":
        net_class = networks.Hipnn
    else:
        if tensor_model == "TS":
            net_class = {
                1: networks.HipnnVec,
                2: networks.HipnnQuad,
            }[l_max]
        elif tensor_model == "HOP":
            net_class = networks.HipHopnn
            network_params["l_max"] = l_max
            network_params["n_max"] = n_max

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    if os.path.isfile(f"{db_dir}/{db_name}-CELL.npy"):
        cells = inputs.CellNode(db_name="CELL")
        network = net_class("hipnn_model", (species,positions,cells), periodic=True, module_kwargs=network_params)
    else:
        network = net_class("hipnn_model", (species, positions), module_kwargs=network_params)

    if not atomization_consistent:
        henergy = targets.HEnergyNode("HEnergy", network)
    else:
        henergy = targets.AtomizationEnergyNode("HEnergy", network)

    force = physics.GradientNode("forces", (henergy, positions), sign=-1)

    return henergy, force, species


# Creates the loss nodes for training from the given henergy, force, and species node.
# force_percent is a number from 0-1 that states how much force should be weighted in
# the loss function (set force_percent = 0.0 to not train on forces at all).
def make_loss(henergy, force, species, force_percent):
    from hippynn.graphs.nodes.loss import MSELoss, MAELoss, Rsq, Mean
    from hippynn.graphs import physics

    E_pred_per_atom = physics.PerAtom("E/atomPred",(henergy.mol_energy.pred,species.true))
    E_true_per_atom = physics.PerAtom("E/atomTrue",(henergy.mol_energy.true,species.true))

    losses = {
        "T-RMSE": MSELoss.of_node(henergy) ** (1 / 2),
        "T-RMSE/atom": MSELoss(E_pred_per_atom, E_true_per_atom) ** (1 / 2),
        "T-MAE": MAELoss.of_node(henergy),
        "T-RSQ": Rsq.of_node(henergy),
        "T-Hier": Mean.of_node(henergy.hierarchicality),
    }

    force_losses = {
        "F-RMSE": MSELoss.of_node(force) ** (1 / 2),
        "F-MAE": MAELoss.of_node(force),
        "F-RSQ": Rsq.of_node(force),
    }

    losses["EnergyTotal"] = losses["T-RMSE"] + losses["T-MAE"]
    losses["LossTotal"] = losses["EnergyTotal"] + losses["T-Hier"]
    if force_percent != 0.0:
        losses.update(force_losses)
        losses["ForceTotal"] = losses["F-RMSE"] + losses["F-MAE"]
        losses["LossTotal"] = 2*( (1 - force_percent) * losses["LossTotal"] + force_percent * losses["ForceTotal"] )

    return losses

# creates a database given relevant information about the database.
def load_db(db_info, name, directory, quiet, test_size, valid_size, seed):

    from hippynn.databases import DirectoryDatabase

    database_params = {
        "name": name,  # Prefix for arrays in folder
        "directory": directory,
        "quiet": quiet,
        "test_size": test_size,
        "valid_size": valid_size,
        "seed": seed,
        **db_info,  # Adds the inputs and targets names from the model as things to load
    }

    database = DirectoryDatabase(**database_params)

    return database

# Sets up objects that are required for the experiment to run.
def setup_experiment(training_modules, device, batch_size, init_lr, patience, max_epochs, stopping_key):
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=init_lr)
    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=batch_size,
        patience=patience,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_epochs=max_epochs,
        stopping_key=stopping_key,
        termination_patience=2 * patience,
    )

    setup_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=device,
    )
    return setup_params

def main(args):
    # Configure GPU settings
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    torch.set_default_dtype(torch.float32)

    hippynn.settings.WARN_LOW_DISTANCES = False

    # compute possible species:
    species = np.load(f"{args.db_dir}/{args.db_name}-Z.npy")
    possible_species = np.unique(species)

    # set up network parameters
    netname = f"{args.tag}_GPU{args.gpu}"
    network_parameters = {
        "possible_species": possible_species,
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
    }

    with hippynn.tools.active_directory(netname):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):

            if os.path.isfile("./best_model.pt") and not args.force_restart:
                # If the model has already been trained, then resume training.

                from hippynn.experiment.serialization import load_checkpoint_from_cwd
                from hippynn.experiment import train_model
                print("loading from checkpoint...")
                check = load_checkpoint_from_cwd(restart_db = True)
                train_model(**check, callbacks=None, batch_callbacks=None)

            else:
                # If the model has not yet been trained, then create a new model and traing it.

                # Create model and loss functions
                henergy, force, species = get_primary_nodes(
                    network_parameters,
                    l_max=args.tensor_order,
                    n_max=args.tensor_factors,
                    atomization_consistent=args.atomization_consistent,
                    db_dir=args.db_dir,
                    db_name=args.db_name,
                )

                henergy.mol_energy.db_name = "T"
                force.db_name = "F"

                validation_losses = make_loss(henergy, force, species, force_percent=args.force_percent)

                train_loss = validation_losses["LossTotal"]

                from hippynn.experiment import assemble_for_training

                training_modules, db_info = assemble_for_training(train_loss, validation_losses)

                # Load data
                database = load_db(
                    db_info,
                    args.db_name,
                    args.db_dir,
                    args.db_quiet,
                    args.test_size,
                    args.valid_size,
                    args.seed
                )

                from hippynn.pretraining import hierarchical_energy_initialization

                hierarchical_energy_initialization(henergy, database, trainable_after=False)

                # Setup training apparatus and begin training.

                patience = args.patience
                if args.use_ccx_subset:
                    patience *= 4

                setup_params = setup_experiment(
                    training_modules,
                    device=args.gpu,
                    batch_size=args.batch_size,
                    init_lr=args.init_lr,
                    patience=patience,
                    max_epochs=args.max_epochs,
                    stopping_key=args.stopping_key,
                )

                from hippynn.experiment import setup_and_train

                setup_and_train(
                    training_modules=training_modules,
                    database=database,
                    setup_params=setup_params,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction

    parser.add_argument("--tag", type=str, default="TEST_MODEL", help="name for run")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to run on, if any")
    parser.add_argument(
        "--use-gpu",
        action=BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Whether to use GPU. Defaults to torch.cuda.is_available()",
    )

    parser.add_argument("--seed", type=int, default=0, help="random seed for init and split")

    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=128)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.5)
    parser.add_argument("--lower_cutoff", type=float, default=0.55, help="Where to initialize the shortest distance sensitivity")
    parser.add_argument(
        "--tensor_model",
        type=str.upper,
        default="HOP",
        choices=["HOP", "TS", "NONE"],
        help="Which tensor architecture to use.  Choices are 'HOP' for HIP-HOP-NN, "
        "'TS' for HIP-NN-TS, and 'NONE' for vanilla HIP-NN'. "
        "If tensor_order==0 then vanilla HIP-NN will "
        "be used regardless.",
    )
    parser.add_argument("--tensor_order", type=int, default=3, help="tensor order $\ell$")
    parser.add_argument("--tensor_factors", type=int, default=4, help="number of factors used (in HIP-HOP-NN only)")
    parser.add_argument("--atomization_consistent", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--stopping_key", type=str, default="T-RMSE/atom")

    parser.add_argument(
        "--use_ccx_subset",
        type=bool,
        default=False,
        help="Train only to configurations from the ANI-1ccx subset."
        " Note that this will still use the energies using the `qm_method` argument."
        " *Note!* This argument will multiply the patience by a factor of 4.",
    )

    parser.add_argument("--progress", action=BooleanOptionalAction, default=True, help="Whether to use progress bars.")
    parser.add_argument("--n_workers", type=int, default=2, help="workers for pytorch dataloaders")

    parser.add_argument("--db_name", type=str, default="mace-", help="prefix in front of data files.")
    parser.add_argument("--db_dir", type=str, default="../../../data/mace", help="directory containing data files")
    parser.add_argument("--db_quiet", action=BooleanOptionalAction, default=False, help="should the database be quiet")
    parser.add_argument("--test_size", type=float, default=0.1, help="Percentage of the dataset used for testing (decimal between 0 and 1)")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Percentage of dataset used for validation (decimal between 0 and 1)")
    parser.add_argument("--force_restart", action=BooleanOptionalAction, default=False, help="Restart training from beginning.")
    parser.add_argument("--force_percent", type=float, default=0.5, help="What percentage of the loss is due to the force.")

    args = parser.parse_args()

    main(args)
