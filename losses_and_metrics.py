import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


class PeriodicityConstrainedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2, gamma=0.2, trend_kernel=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trend_kernel = trend_kernel

    def moving_average(self, x):
        pad = self.trend_kernel // 2
        trend = F.avg_pool1d(x.unsqueeze(1), kernel_size=self.trend_kernel, stride=1, padding=pad)
        return trend.squeeze(1)

    def forward(self, pred, true):
        main_loss = F.mse_loss(pred, true)
        pred_trend = self.moving_average(pred)
        true_trend = self.moving_average(true)
        trend_loss = F.l1_loss(pred_trend, true_trend)
        pred_seasonal = pred - pred_trend
        true_seasonal = true - true_trend
        seasonal_loss = F.l1_loss(pred_seasonal, true_seasonal)
        total = self.alpha * main_loss + self.beta * trend_loss + self.gamma * seasonal_loss
        return total, {"main_loss": float(main_loss.detach().item()), "trend_loss": float(trend_loss.detach().item()), "seasonal_loss": float(seasonal_loss.detach().item())}
