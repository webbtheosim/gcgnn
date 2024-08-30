import os
import torch
import numpy as np


class EarlyStopping:
    """Early stopping and model checkpoint"""

    def __init__(self, patience=200, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, name, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, name, epoch):
        if self.verbose:
            print(
                f"Epoch {epoch:<4}: val loss ({self.val_loss_min:.4f} --> {val_loss:.4f})."
            )
        torch.save(model.state_dict(), os.path.join(path, f"{name}.pt"))
        self.val_loss_min = val_loss
