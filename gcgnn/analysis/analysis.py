import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr


def get_metrics(y_true, y_pred):
    """Get the metrics for the model"""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr = pearsonr(y_true, y_pred)[0]
    mp = mean_absolute_percentage_error(y_true, y_pred)
    return (rmse, mae, r2, pr, mp)


def hyper_select(
    TRAIN_RESULT_DIR, method, split_type, pattern, y_type, if_log, select_on
):
    """Select the hyperparameters for each model"""

    file_pattern = os.path.join(
        TRAIN_RESULT_DIR,
        f"{method}_{split_type}_{pattern}_*_*_*_*_{y_type}_mse*_{if_log}.pickle",
    )
    files = glob.glob(file_pattern)

    files = sorted(files)
    v_metrics = []
    y_metrics = []

    for file in files:

        with open(file, "rb") as handle:
            y_true = pickle.load(handle)
            y_pred = pickle.load(handle)
            y_base = pickle.load(handle)
            v_true = pickle.load(handle)
            v_pred = pickle.load(handle)
            v_base = pickle.load(handle)

        if "_1.pickle" in file:
            y_true = 10**y_true
            y_pred = 10**y_pred
            y_base = 10**y_base
            v_true = 10**v_true
            v_pred = 10**v_pred
            v_base = 10**v_base

        if select_on == "log":
            y_true = np.log10(y_true)
            y_pred = np.log10(y_pred)
            y_base = np.log10(y_base)
            v_true = np.log10(v_true)
            v_pred = np.log10(v_pred)
            v_base = np.log10(v_base)

        try:
            v_rmse, v_mae, v_r2, v_pr, v_mp = get_metrics(v_true, v_pred)
            y_rmse, y_mae, y_r2, y_pr, y_mp = get_metrics(y_true, y_pred)
        except Exception:
            v_rmse = v_mae = y_rmse = y_mae = v_mp = y_mp = np.inf
            v_r2 = v_pr = y_r2 = y_pr = -np.inf

        v_metrics.append([v_rmse, v_mae, v_r2, v_pr, v_mp])
        y_metrics.append([y_rmse, y_mae, y_r2, y_pr, y_mp])

    v_metrics = np.array(v_metrics)
    y_metrics = np.array(y_metrics)

    return v_metrics, y_metrics, files


def hyper_best(TRAIN_RESULT_DIR,
    methods, split_type, pattern, y_type, scoring="RMSE", if_log="1", select_on="normal"
):
    """Select the best hyperparameters for each model"""
    v_metrics = []
    y_metrics = []
    files = []

    for i, method in enumerate(methods):
        v_metric, y_metric, file = hyper_select(TRAIN_RESULT_DIR,
            method, split_type, pattern, y_type, if_log, select_on
        )
        v_metrics.append(v_metric)
        y_metrics.append(y_metric)
        files.append(file)

    if scoring == "RMSE":
        j = 0
    elif scoring == "MAE":
        j = 1
    elif scoring == "R2":
        j = 2
    elif scoring == "R":
        j = 3

    y_metrics_out = []
    v_metrics_out = []
    files_out = []

    for i, v_metric in enumerate(v_metrics):
        if j <= 1:
            k = np.argmin(v_metric[:, j])
        else:
            k = np.argmax(v_metric[:, j])
        v_metrics_out.append(v_metrics[i][k])
        y_metrics_out.append(y_metrics[i][k])
        files_out.append(files[i][k])

    return v_metrics_out, y_metrics_out, files_out


def metric_csv(TRAIN_RESULT_DIR, RESULT_DIR, y_type, split_type, pattern, scoring):
    """Generate csv file for the metrics of the models"""

    methods_short = ["GC-GNN", "GNN", "GC"]
    methods = ["GNN_Guided_Baseline_Simple", "GNN", "Baseline"]

    v_metric, y_metric, file = hyper_best(TRAIN_RESULT_DIR,
        methods, split_type, pattern, y_type, scoring, if_log="1", select_on="normal"
    )

    print("Best hyperparameters (file):")
    for f in file:
        print_name = os.path.basename(f).split(".pickle")[0]
        if "GNN_Guided_Baseline_Simple" in print_name:
            print(print_name.replace("GNN_Guided_Baseline_Simple", "GC-GNN"))
        elif "Baseline" in print_name:
            print(print_name.replace("Baseline", "GC"))
        else:
            print(print_name)

    arr = []

    for i, method in enumerate(methods):
        arr.append([methods_short[i]] + ["Test"] + list(y_metric[i]))

    for i, method in enumerate(methods):
        arr.append([methods_short[i]] + ["Val"] + list(v_metric[i]))

    df = pd.DataFrame(
        arr, columns=["Model", "Split", "RMSE", "MAE", "R2", "R", "MAPE"]
    )

    out_file = os.path.join(RESULT_DIR, f"metric_{y_type}_{split_type}_{pattern}.csv")

    df.to_csv(out_file, index=False)

    return df


class Args:
    """Class for the hyperparameters of the model"""
    def __init__(self):
        self.batch_size = 256
        self.lr = 0.005
        self.epochs = 2000
        self.patience = int(self.epochs / 1)
        self.dim = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def get_args(file):
    """Get the hyperparameters for the model"""
    args = Args()
    args.split_type = int(file.split("_")[-9])
    args.pure_type = int(file.split("_")[-8])
    args.model_type = "_".join(file.split("_")[:-9])
    args.y_type = file.split("_")[-3]
    args.batch_size = int(file.split("_")[-7])
    args.dim = int(file.split("_")[-4])
    args.lr = float(file.split("_")[-6])
    args.epochs = int(file.split("_")[-5])
    args.if_log = int(file.split("_")[-1])
    args.hyper_name = file
    return args
