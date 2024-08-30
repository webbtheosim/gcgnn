import os
import torch
import random
import pickle
import itertools

import numpy as np

from torch_geometric.data import DataLoader

from gcgnn.model_utils import train
from gcgnn.model_data import load_data

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Args:
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.001
        self.epochs = 2000
        self.patience = int(self.epochs / 1)
        self.dim = 128
        self.device = DEVICE
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dim = 3
        self.output_dim = 1
        self.model_type = None
        self.split_type = 0
        self.pure_type = 0
        self.print_freq = 10
        self.y_type = "mean"
        self.hyper_name = None
        self.if_log = 1
        self.loss = "mae"
        self.MODEL_PATH = "/scratch/gpfs/sj0161/delta_learning/model/"
        self.HIST_PATH = "/scratch/gpfs/sj0161/delta_learning/history/"
        self.DATA_DIR = "/scratch/gpfs/sj0161/delta_pattern/"


def process_data(args):
    """Process the data"""
    (data_train, data_val, topo_train, topo_val), data_test, topo_test = load_data(args)

    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    args.train_loader = train_loader
    args.val_loader = val_loader
    args.test_loader = test_loader

    hyper_name = f"{args.model_type}_{args.split_type}_{args.pure_type}_{args.batch_size}_{args.lr}_{args.epochs}_{args.dim}_{args.y_type}_{args.loss}_{args.if_log}"
    args.hyper_name = hyper_name

    hist_file = os.path.join(args.HIST_PATH, hyper_name + ".pickle")

    print()
    print(hyper_name)
    print(f"Train size: {len(data_train)}")
    print(f"Val size:   {len(data_val)}")
    print(f"Test size:  {len(data_test)}")

    return hist_file


def get_hist_name(args):
    """Get the history file name"""
    hyper_name = f"{args.model_type}_{args.split_type}_{args.pure_type}_{args.batch_size}_{args.lr}_{args.epochs}_{args.dim}_{args.y_type}_{args.loss}_{args.if_log}"
    args.hyper_name = hyper_name
    hist_file = os.path.join(args.HIST_PATH, hyper_name + ".pickle")

    print()
    print(hyper_name)

    return hist_file


def train_model(args, hist_file):
    """Train the model"""
    (
        train_loss,
        val_loss,
        train_metric,
        val_metric,
        y_pred,
        y_true,
        y_base,
        v_pred,
        v_true,
        v_base,
        model,
    ) = train(args)

    if not os.path.exists(args.HIST_PATH):
        os.makedirs(args.HIST_PATH)

    with open(hist_file, "wb") as handle:
        pickle.dump(y_true, handle)
        pickle.dump(y_pred, handle)
        pickle.dump(y_base, handle)
        pickle.dump(v_true, handle)
        pickle.dump(v_pred, handle)
        pickle.dump(v_base, handle)
        pickle.dump(train_loss, handle)
        pickle.dump(val_loss, handle)
        pickle.dump(train_metric, handle)
        pickle.dump(val_metric, handle)


def main(args):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    hist_file = get_hist_name(args)

    if not os.path.exists(hist_file):
        hist_file = process_data(args)
        train_model(args, hist_file)
    else:
        print(f"# {hist_file} done ...")


def job_array(idx, max_idx):
    split_types = [0, 1, 2]
    pure_types = [0, 1, 2]
    y_types = ["std", "mean"]
    model_types = ["GNN_Guided_Baseline_Simple", "GNN", "Baseline"]
    losses = ["mse", "mae"]
    batch_sizes = [64, 128, 256, 512]
    dims = [32, 64, 128, 256]
    lrs = [0.001, 0.005, 0.0005]
    if_logs = [0, 1]

    combs = itertools.product(
        model_types,
        split_types,
        pure_types,
        y_types,
        losses,
        batch_sizes,
        dims,
        lrs,
        if_logs,
    )
    combs = list(combs)

    size = len(combs) // max_idx
    start = idx * size
    end = min((idx + 1) * size, len(combs))

    return combs[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

    combs = job_array(idx, max_idx)

    for comb in combs:
        model_type, split_type, pure_type, y_type, loss, batch_size, dim, lr, if_log = (
            comb
        )

        args = Args()
        args.split_type = split_type
        args.pure_type = pure_type
        args.model_type = model_type
        args.y_type = y_type
        args.loss = loss
        args.batch_size = batch_size
        args.dim = dim
        args.lr = lr
        args.epochs = 1000
        args.if_log = if_log

        main(args)
