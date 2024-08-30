import os
from gcgnn.models.early_stopping import EarlyStopping
from gcgnn.models.gnn_models import (
    GNN_Guided_Baseline,
    GNN,
    Baseline,
    GNN_Guided_Baseline_Simple,
)
import numpy as np
from sklearn.metrics import r2_score
import warnings
import torch
from sklearn.metrics import mean_squared_error, r2_score

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.nn.functional as F


def train(args):
    """Train the model"""
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    hyper_name = args.hyper_name
    model_types = {
        "GNN": GNN(args.input_dim, args.dim, args.output_dim),
        "Baseline": Baseline(args.input_dim, args.dim, args.output_dim),
        "GNN_Guided_Baseline_Simple": GNN_Guided_Baseline_Simple(
            args.input_dim, args.dim, args.output_dim
        ),
    }

    model = model_types.get(args.model_type, None)
    if model is None:
        raise ValueError("Invalid model_type")
    model.to(args.device)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_metric = []

        # =========================train=======================
        for _, batch in enumerate(args.train_loader):
            batch = batch.to(args.device)
            outputs = model(batch)
            optimizer.zero_grad()

            loss = criterion(outputs[0], batch.y)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            y_true = batch.y.cpu().numpy()
            y_pred = outputs[0].cpu().detach().numpy()
            r2 = r2_score(y_true, y_pred)
            train_metric.append(r2)

        train_losses.append(np.average(train_loss))
        train_metrics.append(np.average(train_metric))

        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_loss = []
            val_metric = []

            for _, batch in enumerate(args.val_loader):
                batch = batch.to(args.device)
                outputs = model(batch)
                loss = criterion(outputs[0], batch.y)
                val_loss.append(loss.item())

                y_true = batch.y.cpu().numpy()
                y_pred = outputs[0].cpu().detach().numpy()
                r2 = r2_score(y_true, y_pred)
                val_metric.append(r2)

            val_losses.append(np.average(val_loss))
            val_metrics.append(np.average(val_metric))

        early_stopping(
            val_losses[-1], model=model, path=args.MODEL_PATH, name=hyper_name, epoch=epoch
        )

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # =========================test=========================
    model_file = os.path.join(args.MODEL_PATH, f"{hyper_name}.pt")
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

    model.eval()

    y_pred = []
    y_true = []
    y_base = []

    with torch.no_grad():
        for _, batch in enumerate(args.test_loader):
            batch = batch.to(args.device)
            output = model(batch)
            y_pred.append(output[0].cpu().numpy())
            y_true.append(batch.y.cpu().numpy())
            y_base.append(batch.base.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0).squeeze()
    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_base = np.concatenate(y_base, axis=0).squeeze()

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print("Testing")
    print(f"{hyper_name} RMSE:{rmse:<5.2f}  R2:{r2:<5.2f}")

    # validation
    v_pred = []
    v_true = []
    v_base = []

    with torch.no_grad():
        for _, batch in enumerate(args.val_loader):
            batch = batch.to(args.device)
            output = model(batch)
            v_pred.append(output[0].cpu().numpy())
            v_true.append(batch.y.cpu().numpy())
            v_base.append(batch.base.cpu().numpy())

    v_pred = np.concatenate(v_pred, axis=0).squeeze()
    v_true = np.concatenate(v_true, axis=0).squeeze()
    v_base = np.concatenate(v_base, axis=0).squeeze()

    rmse = mean_squared_error(v_true, v_pred) ** 0.5
    r2 = r2_score(v_true, v_pred)
    print("Validation")
    print(f"{hyper_name} RMSE:{rmse:<5.2f}  R2:{r2:<5.2f}")

    return (
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        y_pred,
        y_true,
        y_base,
        v_pred,
        v_true,
        v_base,
        model,
    )


def evaluate_model(args):
    """Evaluate the model"""
    hyper_name = args.hyper_name
    print(hyper_name)

    model_types = {
        "GNN_Guided_Baseline": GNN_Guided_Baseline(
            args.input_dim, args.dim, args.output_dim
        ),
        "GNN": GNN(args.input_dim, args.dim, args.output_dim),
        "Baseline": Baseline(args.input_dim, args.dim, args.output_dim),
        "GNN_Guided_Baseline_Simple": GNN_Guided_Baseline_Simple(
            args.input_dim, args.dim, args.output_dim
        ),
    }
    model = model_types.get(args.model_type, None)

    if model is None:
        raise ValueError("Invalid model_type")

    model.to(args.device)
    model_file = os.path.join(args.MODEL_PATH, f"{hyper_name}.pt")
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()

    return model
