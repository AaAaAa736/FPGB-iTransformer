"""FPGB-iTransformer：超参数搜索 + 多 horizon / 多 seed 正式训练。

FPGB-iTransformer 是一种面向电力负荷预测的 iTransformer 变体，在标准
iTransformer 的基础上引入了三个关键组件：

1. 未知历史重构分支（Unknown History Reconstructor）
2. 已知未来特征编码分支（Future Known Encoder）
3. 周期上下文门控（Periodic Context Gate）

并配合 RevIN 归一化与三支路融合（Three-Branch Fusion），主干使用标准
Scaled Dot-Product Softmax 自注意力。

本脚本把超参数搜索与正式训练整合为一个完整流程：对每个 horizon 先自动
搜索最佳超参数（默认随机抽样 100 组超参数，每组训练 100 轮，可早停，按
验证集损失 best_val_loss 选出最佳超参数），随后直接用该最佳超参数训练
5 个随机种子，输出压缩预测结果（predictions.npz）与评估指标（metrics.json）。

数据：Ontario IESO 小时级负荷（2002-2024），默认位于 data/ 目录。

示例：
    python run_fpgb_itransformer.py                         # 完整流程（3 horizon × 5 seed）
    python run_fpgb_itransformer.py --horizons 24 --cpu     # 只跑 24 步 horizon，使用 CPU
    python run_fpgb_itransformer.py --smoke_only            # 冒烟测试，只验证数据与模型前向
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import build_multi_file_datasets
from engine import (
    TrainConfig,
    compute_step_metrics,
    evaluate_loss,
    get_device,
    make_loaders,
    predict,
    set_seed,
    train_one_epoch,
)
from losses_and_metrics import PeriodicityConstrainedLoss, mae, mape, rmse
from modeling_softmax import ITransformerWithKnownFuture


HERE = Path(__file__).resolve().parent
MODEL_NAME = "FPGB-iTransformer"

# ---------------------------------------------------------------------------
# 数据特征定义（Ontario IESO）
# ---------------------------------------------------------------------------
TARGET_NAME = "ontario_demand_mw"
UNKNOWN_FUTURE_NAMES = [
    "weather_site_mean_relative_humidity_pct",
    "weather_site_mean_snowfall_cm",
    "weather_site_mean_cloud_cover_pct",
    "weather_site_mean_shortwave_radiation_wm2",
]
TEMPERATURE_NAMES = [
    "temperature_2m_toronto_c",
    "temperature_2m_london_c",
    "weather_site_mean_apparent_temperature_c",
]
PERIODIC_NAMES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
DAYTYPE_NAMES = [
    "daytype_workday",
    "daytype_saturday",
    "daytype_sunday",
    "daytype_holiday",
]
KNOWN_FUTURE_NAMES = TEMPERATURE_NAMES + PERIODIC_NAMES + DAYTYPE_NAMES
FEATURE_NAMES = [TARGET_NAME] + UNKNOWN_FUTURE_NAMES + KNOWN_FUTURE_NAMES

HORIZONS = [24, 48, 96]
SEEDS = [42, 123, 456, 789, 135]


def parse_args():
    parser = argparse.ArgumentParser(description="FPGB-iTransformer 超参数搜索与正式训练")
    parser.add_argument(
        "--source_csv",
        default=str(HERE / "data" / "ieso_ontario_hourly_demand_2002_2024.csv"),
    )
    parser.add_argument("--save_root", default=str(HERE / "outputs"))
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--search_seed", type=int, default=42, help="超参数搜索随机抽样的种子")
    parser.add_argument("--search_trials", type=int, default=100, help="超参数搜索试验组数")
    parser.add_argument("--search_epochs", type=int, default=100, help="每组搜索试验的训练轮数")
    parser.add_argument("--search_patience", type=int, default=10, help="搜索阶段的早停 patience")
    parser.add_argument("--formal_epochs", type=int, default=30, help="正式训练的轮数")
    parser.add_argument("--formal_patience", type=int, default=6, help="正式训练的早停 patience")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    return parser.parse_args()


def atomic_json_dump(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def prepare_ontario_features(source_csv: Path, prepared_dir: Path, train_ratio: float, val_ratio: float):
    """读取原始 IESO CSV，构造特征表并缓存到 prepared_dir。"""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = prepared_dir / "ontario_hourly_features.csv"
    metadata_path = prepared_dir / "dataset_metadata.json"
    source_stat = source_csv.stat()
    source_signature = {
        "source_csv": str(source_csv),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }

    if feature_csv.exists() and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source_signature") == source_signature and cached.get("feature_names") == FEATURE_NAMES:
            timestamps = pd.to_datetime(
                pd.read_csv(source_csv, usecols=["timestamp_start_est"])["timestamp_start_est"]
            )
            return feature_csv, timestamps, cached

    required = [
        "timestamp_start_est",
        TARGET_NAME,
        *UNKNOWN_FUTURE_NAMES,
        *TEMPERATURE_NAMES,
        "daytype_workday",
        "daytype_saturday",
        "daytype_sunday",
    ]
    df = pd.read_csv(source_csv, usecols=required)
    timestamps = pd.to_datetime(df["timestamp_start_est"])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamp_start_est must be monotonically increasing")
    if timestamps.duplicated().any():
        raise ValueError("timestamp_start_est contains duplicates")
    gaps = timestamps.diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError("timestamp_start_est is not a continuous hourly series")

    hour_angle = 2.0 * math.pi * timestamps.dt.hour.to_numpy() / 24.0
    days_in_year = np.where(timestamps.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    doy_angle = 2.0 * math.pi * (timestamps.dt.dayofyear.to_numpy() - 1.0) / days_in_year
    dow_angle = 2.0 * math.pi * timestamps.dt.dayofweek.to_numpy() / 7.0

    features = pd.DataFrame(index=df.index)
    for name in [TARGET_NAME] + UNKNOWN_FUTURE_NAMES + TEMPERATURE_NAMES:
        features[name] = pd.to_numeric(df[name], errors="raise")
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["doy_sin"] = np.sin(doy_angle)
    features["doy_cos"] = np.cos(doy_angle)
    features["dow_sin"] = np.sin(dow_angle)
    features["dow_cos"] = np.cos(dow_angle)
    for name in DAYTYPE_NAMES[:-1]:
        features[name] = pd.to_numeric(df[name], errors="raise")
    features["daytype_holiday"] = 1.0 - features[DAYTYPE_NAMES[:-1]].sum(axis=1)
    features = features[FEATURE_NAMES]

    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Prepared feature table contains NaN or infinity")
    if not features[DAYTYPE_NAMES].isin([0, 1]).all().all():
        raise ValueError("Day-type columns must be binary")
    if not (features[DAYTYPE_NAMES].sum(axis=1) == 1).all():
        raise ValueError("Every row must have exactly one day type")

    features.to_csv(feature_csv, index=False, float_format="%.9g")
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    metadata = {
        "source_signature": source_signature,
        "rows": n,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "known_future_names": KNOWN_FUTURE_NAMES,
        "unknown_future_names": UNKNOWN_FUTURE_NAMES,
        "excluded_columns": ["sequence_index", "timestamp_start_est", "hour_of_day"],
        "split": {
            "train": {"start": str(timestamps.iloc[0]), "end": str(timestamps.iloc[train_end - 1]), "rows": train_end},
            "val": {"start": str(timestamps.iloc[train_end]), "end": str(timestamps.iloc[val_end - 1]), "rows": val_end - train_end},
            "test": {"start": str(timestamps.iloc[val_end]), "end": str(timestamps.iloc[-1]), "rows": n - val_end},
        },
    }
    atomic_json_dump(metadata, metadata_path)
    return feature_csv, timestamps, metadata


def build_bundle(prepared_dir: Path, args, pred_len: int):
    bundle = build_multi_file_datasets(
        data_dir=str(prepared_dir),
        target_name=TARGET_NAME,
        known_future_names=KNOWN_FUTURE_NAMES,
        seq_len=args.seq_len,
        pred_len=pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if bundle.feature_names != FEATURE_NAMES:
        raise RuntimeError(f"Unexpected feature order: {bundle.feature_names}")
    bundle.periodic_known_idx = [
        idx for idx, name in enumerate(KNOWN_FUTURE_NAMES) if name in PERIODIC_NAMES + DAYTYPE_NAMES
    ]
    return bundle


def main_model(bundle, seq_len: int, pred_len: int, cfg: dict):
    """构建 FPGB-iTransformer 模型。"""
    return ITransformerWithKnownFuture(
        seq_len=seq_len,
        pred_len=pred_len,
        hist_input_dim=len(bundle.feature_names),
        target_idx=bundle.target_idx,
        known_future_dim=len(bundle.known_future_idx),
        periodic_known_idx=bundle.periodic_known_idx,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        e_layers=cfg["e_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )


def search_candidates(seed: int, count: int):
    """生成随机抽样的超参数候选。

    搜索范围覆盖 d_model / n_heads / e_layers / d_ff / dropout / lr / weight_decay，
    并剔除 d_model 不能被 n_heads 整除的非法组合。
    """
    grid = {
        "d_model": [64, 128, 256, 512, 1024],
        "n_heads": [4, 8, 16, 32],
        "e_layers": [2, 3, 4, 5],
        "d_ff": [128, 256, 512, 1024, 2048],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-4, 5e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    }
    keys = list(grid)
    candidates = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, values))
        if cfg["d_model"] % cfg["n_heads"] == 0:
            candidates.append(cfg)
    random.Random(seed).shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def _search_best_config(args, bundle, pred_len: int, search_dir: Path):
    """在验证集上搜索最佳超参数（完整训练流程的自动前置步骤）。"""
    search_dir.mkdir(parents=True, exist_ok=True)
    summary_path = search_dir / "search_summary.csv"
    best_path = search_dir / "best_hyperparameters.json"
    if summary_path.exists() and best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            return json.load(f), pd.read_csv(summary_path)

    rows = []
    device = get_device(args.cpu)
    train_loader, val_loader, _ = make_loaders(bundle, args.batch_size)
    criterion = PeriodicityConstrainedLoss(alpha=1.0, beta=0.2, gamma=0.2, trend_kernel=3)

    candidates = search_candidates(args.search_seed, args.search_trials)
    for trial_idx, cfg in enumerate(candidates, start=1):
        trial_dir = search_dir / f"trial_{trial_idx:03d}"
        trial_metrics = trial_dir / "metrics.json"
        if trial_metrics.exists():
            with open(trial_metrics, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
            continue

        trial_dir.mkdir(parents=True, exist_ok=True)
        set_seed(args.search_seed + trial_idx)
        model = main_model(bundle, args.seq_len, pred_len, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        best_val = float("inf")
        best_state = None
        patience_counter = 0
        epoch_rows = []
        started = time.time()
        print(f"\n[SEARCH {trial_idx}/{len(candidates)}] {json.dumps(cfg)}", flush=True)

        for epoch in range(1, args.search_epochs + 1):
            train_loss, parts = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            epoch_rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "main_loss": parts["main_loss"],
                "trend_loss": parts["trend_loss"],
                "seasonal_loss": parts["seasonal_loss"],
                "lr": optimizer.param_groups[0]["lr"],
            })
            print(f"[SEARCH {trial_idx}] epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.search_patience:
                break

        elapsed = time.time() - started
        torch.save({"model_state_dict": best_state, "hyperparameters": cfg}, trial_dir / "best.pt")
        pd.DataFrame(epoch_rows).to_csv(trial_dir / "train_log.csv", index=False)
        row = {"trial_idx": trial_idx, **cfg, "best_val_loss": float(best_val), "time_s": elapsed}
        atomic_json_dump(row, trial_metrics)
        rows.append(row)
        del model, optimizer, scheduler, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    best = summary.iloc[0]
    best_cfg = {
        "d_model": int(best["d_model"]),
        "n_heads": int(best["n_heads"]),
        "e_layers": int(best["e_layers"]),
        "d_ff": int(best["d_ff"]),
        "dropout": float(best["dropout"]),
        "lr": float(best["lr"]),
        "weight_decay": float(best["weight_decay"]),
        "attn_type": "softmax",
    }
    atomic_json_dump(best_cfg, best_path)
    return best_cfg, summary


def _train_seed(args, bundle, best_cfg: dict, pred_len: int, seed: int, timestamps, val_end: int, save_dir: Path):
    """用最佳超参数对单个 seed 进行正式训练，保存 predictions.npz 与 metrics.json。"""
    metrics_path = save_dir / "metrics.json"
    prediction_path = save_dir / "predictions.npz"
    checkpoint_path = save_dir / "best.pt"
    if metrics_path.exists() and prediction_path.exists() and checkpoint_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = get_device(args.cpu)
    model = main_model(bundle, args.seq_len, pred_len, best_cfg).to(device)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.formal_epochs,
        patience=args.formal_patience,
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
        seed=seed,
        use_cpu=args.cpu,
    )
    criterion = PeriodicityConstrainedLoss(alpha=1.0, beta=0.2, gamma=0.2, trend_kernel=3)
    train_loader, val_loader, test_loader = make_loaders(bundle, train_cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    epoch_rows = []
    started = time.time()
    run_name = f"{MODEL_NAME}_pred{pred_len}_seed{seed}"

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, parts = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "main_loss": parts["main_loss"],
            "trend_loss": parts["trend_loss"],
            "seasonal_loss": parts["seasonal_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{run_name}] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= train_cfg.patience:
            print(f"[{run_name}] early_stop={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {run_name}")
    model.load_state_dict(best_state)
    pred_scaled, true_scaled = predict(model, test_loader, device)
    pred = bundle.scaler.inverse_transform_target(pred_scaled, bundle.target_idx)
    true = bundle.scaler.inverse_transform_target(true_scaled, bundle.target_idx)
    elapsed = time.time() - started
    starts = timestamps.iloc[val_end : val_end + len(pred)].to_numpy(dtype="datetime64[ns]").astype("int64")
    np.savez_compressed(
        prediction_path,
        pred=pred.astype(np.float32),
        true=true.astype(np.float32),
        forecast_start_ns=starts,
    )
    compute_step_metrics(true, pred).to_csv(save_dir / "step_metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(save_dir / "train_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": best_state,
            "model": MODEL_NAME,
            "pred_len": pred_len,
            "seed": seed,
            "feature_names": bundle.feature_names,
            "target_idx": bundle.target_idx,
            "known_future_idx": bundle.known_future_idx,
            "periodic_known_idx": bundle.periodic_known_idx,
            "hyperparameters": best_cfg,
        },
        checkpoint_path,
    )
    metrics = {
        "model": MODEL_NAME,
        "pred_len": pred_len,
        "seed": seed,
        "mae": mae(true, pred),
        "rmse": rmse(true, pred),
        "mape": mape(true, pred),
        "best_val_loss": float(best_val),
        "epochs_ran": len(epoch_rows),
        "time_s": elapsed,
        "prediction_npz": str(prediction_path),
        "checkpoint": str(checkpoint_path),
    }
    atomic_json_dump(metrics, metrics_path)
    del model, optimizer, scheduler, best_state, pred_scaled, true_scaled, pred, true
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def scan_completed_metrics(save_root: Path):
    rows = []
    for path in save_root.glob("pred_*/models/*/seed_*/metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def write_summaries(save_root: Path):
    metrics = scan_completed_metrics(save_root)
    if metrics.empty:
        return
    metrics = metrics.sort_values(["pred_len", "seed"]).reset_index(drop=True)
    metrics.to_csv(save_root / "all_seed_metrics.csv", index=False)
    summary = (
        metrics.groupby(["pred_len"], as_index=False)
        .agg(
            seeds_completed=("seed", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            time_mean_s=("time_s", "mean"),
        )
        .sort_values("pred_len")
        .reset_index(drop=True)
    )
    summary.to_csv(save_root / "model_summary_by_horizon.csv", index=False)
    prediction_index = metrics[["pred_len", "seed", "prediction_npz", "checkpoint"]]
    prediction_index.to_csv(save_root / "prediction_index.csv", index=False)


def run_horizon(args, bundle, pred_len, timestamps, val_end, save_root):
    """单个 horizon 的完整训练流程：自动搜索最佳超参数，然后正式训练全部种子。"""
    horizon_dir = save_root / f"pred_{pred_len}"
    print(f"\n{'=' * 80}\nHORIZON {pred_len}\n{'=' * 80}", flush=True)
    print(
        f"windows={len(bundle.train_ds)}/{len(bundle.val_ds)}/{len(bundle.test_ds)} "
        f"features={len(bundle.feature_names)} known={len(bundle.known_future_idx)}",
        flush=True,
    )

    if args.smoke_only:
        cfg = search_candidates(args.search_seed, 1)[0]
        model = main_model(bundle, args.seq_len, pred_len, cfg).to(get_device(args.cpu))
        x_hist, x_future, _ = next(iter(make_loaders(bundle, 2)[0]))
        with torch.no_grad():
            out = model(x_hist.to(get_device(args.cpu)), x_future.to(get_device(args.cpu)))
        print(f"smoke {MODEL_NAME} pred={pred_len} -> {tuple(out.shape)}", flush=True)
        del model
        return

    # 自动搜索最佳超参数
    best_cfg, _ = _search_best_config(args, bundle, pred_len, horizon_dir / "main_model_search")
    print(f"HORIZON {pred_len} BEST={json.dumps(best_cfg)}", flush=True)

    # 用最佳超参数正式训练全部种子
    for seed in args.seeds:
        seed_dir = horizon_dir / "models" / MODEL_NAME / f"seed_{seed}"
        metrics = _train_seed(args, bundle, best_cfg, pred_len, seed, timestamps, val_end, seed_dir)
        print(
            f"DONE model={MODEL_NAME} pred={pred_len} seed={seed} "
            f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} MAPE={metrics['mape']:.3f}%",
            flush=True,
        )
        write_summaries(save_root)


def main():
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 - args.train_ratio:
        raise ValueError("train_ratio and val_ratio must leave a positive test split")
    source_csv = Path(args.source_csv).resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"数据集不存在: {source_csv}")
    save_root = Path(args.save_root).resolve()
    prepared_dir = save_root / "prepared_data"

    _, timestamps, metadata = prepare_ontario_features(
        source_csv, prepared_dir, args.train_ratio, args.val_ratio
    )
    val_end = int(len(timestamps) * (args.train_ratio + args.val_ratio))
    atomic_json_dump(
        {
            "arguments": vars(args),
            "model": MODEL_NAME,
            "device": str(get_device(args.cpu)),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else None,
            "dataset_metadata": metadata,
        },
        save_root / "experiment_manifest.json",
    )

    for pred_len in args.horizons:
        bundle = build_bundle(prepared_dir, args, pred_len)
        run_horizon(args, bundle, pred_len, timestamps, val_end, save_root)

    if args.smoke_only:
        print("\nSMOKE TESTS COMPLETE", flush=True)
        return

    write_summaries(save_root)
    print("\nALL EXPERIMENTS COMPLETE", flush=True)
    summary_csv = save_root / "model_summary_by_horizon.csv"
    if summary_csv.exists():
        print(pd.read_csv(summary_csv).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
