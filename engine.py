import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses_and_metrics import mae, mape, rmse


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    use_cpu: bool = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def get_device(use_cpu: bool = False):
    return torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")


def make_loaders(bundle, batch_size: int):
    train_loader = DataLoader(bundle.train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(bundle.val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(bundle.test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_count = 0.0, 0
    last_parts = {"main_loss": 0.0, "trend_loss": 0.0, "seasonal_loss": 0.0}
    for x_hist, x_future_known, y in loader:
        x_hist = x_hist.to(device)
        x_future_known = x_future_known.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_hist, x_future_known)
        loss, parts = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = x_hist.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        last_parts = parts
    return total_loss / max(total_count, 1), last_parts


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    for x_hist, x_future_known, y in loader:
        x_hist = x_hist.to(device)
        x_future_known = x_future_known.to(device)
        y = y.to(device)
        pred = model(x_hist, x_future_known)
        loss, _ = criterion(pred, y)
        bs = x_hist.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x_hist, x_future_known, y in loader:
        x_hist = x_hist.to(device)
        x_future_known = x_future_known.to(device)
        preds.append(model(x_hist, x_future_known).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def compute_step_metrics(true, pred):
    rows = []
    for t in range(true.shape[1]):
        rows.append({
            "step": t + 1,
            "mae": mae(true[:, t], pred[:, t]),
            "rmse": rmse(true[:, t], pred[:, t]),
            "mape": mape(true[:, t], pred[:, t]),
        })
    return pd.DataFrame(rows)
