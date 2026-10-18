import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def inverse_transform_target(self, x: np.ndarray, target_idx: int):
        return x * self.std[0, target_idx] + self.mean[0, target_idx]


class MultiFileSlidingWindowDataset(Dataset):
    def __init__(self, data_list, target_idx, known_future_idx, seq_len=168, pred_len=24):
        self.data_list = [arr.astype(np.float32) for arr in data_list]
        self.target_idx = target_idx
        self.known_future_idx = list(known_future_idx)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.index_map = []
        for file_idx, arr in enumerate(self.data_list):
            length = len(arr) - seq_len - pred_len + 1
            if length > 0:
                for start_idx in range(length):
                    self.index_map.append((file_idx, start_idx))
        if len(self.index_map) == 0:
            raise ValueError(f"没有可用窗口。请检查 seq_len={seq_len}, pred_len={pred_len}，以及各 CSV 文件长度是否足够。")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, start_idx = self.index_map[idx]
        arr = self.data_list[file_idx]
        x_hist = arr[start_idx:start_idx + self.seq_len]
        y = arr[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len, self.target_idx]
        x_future_known = arr[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len, self.known_future_idx]
        return (
            torch.tensor(x_hist, dtype=torch.float32),
            torch.tensor(x_future_known, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


@dataclass
class DatasetBundle:
    train_ds: MultiFileSlidingWindowDataset
    val_ds: MultiFileSlidingWindowDataset
    test_ds: MultiFileSlidingWindowDataset
    scaler: StandardScaler
    feature_names: List[str]
    target_idx: int
    known_future_idx: List[int]
    periodic_known_idx: List[int]
    used_files: Dict[str, List[str]]


def read_one_csv(csv_path: str, target_name: str):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number]).copy()
    if target_name not in df.columns:
        raise ValueError(f"{csv_path} 中找不到目标列 {target_name}")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()
    return df


def load_all_csvs(data_dir: str, target_name: str = "MW", min_rows: int = 1):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if len(csv_files) == 0:
        raise ValueError(f"{data_dir} 下没有找到任何 .csv 文件")
    dfs = []
    base_columns = None
    for path in csv_files:
        df = read_one_csv(path, target_name)
        if len(df) < min_rows:
            continue
        if base_columns is None:
            base_columns = list(df.columns)
        elif list(df.columns) != base_columns:
            raise ValueError(f"列不一致:\n基准列: {base_columns}\n当前文件: {path}\n当前列: {list(df.columns)}")
        dfs.append((path, df))
    if len(dfs) == 0:
        raise ValueError("没有可用的 CSV 文件")
    feature_names = list(dfs[0][1].columns)
    target_idx = feature_names.index(target_name)
    return dfs, feature_names, target_idx


def split_single_series(arr: np.ndarray, seq_len: int, pred_len: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = len(arr)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_raw = arr[:train_end]
    val_raw = arr[max(0, train_end - seq_len):val_end]
    test_raw = arr[max(0, val_end - seq_len):]
    def is_usable(x):
        return len(x) >= seq_len + pred_len
    return (
        train_raw if is_usable(train_raw) else None,
        val_raw if is_usable(val_raw) else None,
        test_raw if is_usable(test_raw) else None,
    )


def build_feature_groups(feature_names: Sequence[str], target_name: str, known_future_names: Sequence[str]):
    if target_name not in feature_names:
        raise ValueError(f"目标列 {target_name} 不在特征中")
    for name in known_future_names:
        if name not in feature_names:
            raise ValueError(f"已知未来特征 {name} 不在数据列中")
    target_idx = feature_names.index(target_name)
    known_future_idx = [feature_names.index(x) for x in known_future_names]
    return target_idx, known_future_idx


def build_periodic_known_idx(known_future_names: Sequence[str]):
    periodic_keywords = {"DayOfYearSin", "DayOfYearCos", "HourOfDaySin", "HourOfDayCos", "DayType_0", "DayType_1", "DayType_2"}
    return [i for i, name in enumerate(known_future_names) if name in periodic_keywords]


def build_multi_file_datasets(data_dir: str, target_name: str, known_future_names: Sequence[str], seq_len: int, pred_len: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> DatasetBundle:
    min_rows = seq_len + pred_len
    dfs, feature_names, target_idx_from_data = load_all_csvs(data_dir=data_dir, target_name=target_name, min_rows=min_rows)
    target_idx, known_future_idx = build_feature_groups(feature_names=feature_names, target_name=target_name, known_future_names=known_future_names)
    if target_idx != target_idx_from_data:
        raise RuntimeError("target_idx 校验失败")
    periodic_known_idx = build_periodic_known_idx(known_future_names)

    train_raw_list, val_raw_list, test_raw_list = [], [], []
    used_files = {"train": [], "val": [], "test": []}
    for path, df in dfs:
        arr = df.values.astype(np.float32)
        train_raw, val_raw, test_raw = split_single_series(arr=arr, seq_len=seq_len, pred_len=pred_len, train_ratio=train_ratio, val_ratio=val_ratio)
        if train_raw is not None:
            train_raw_list.append(train_raw)
            used_files["train"].append(path)
        if val_raw is not None:
            val_raw_list.append(val_raw)
            used_files["val"].append(path)
        if test_raw is not None:
            test_raw_list.append(test_raw)
            used_files["test"].append(path)

    if len(train_raw_list) == 0:
        raise ValueError("没有任何文件能产生训练窗口")
    if len(val_raw_list) == 0:
        raise ValueError("没有任何文件能产生验证窗口")
    if len(test_raw_list) == 0:
        raise ValueError("没有任何文件能产生测试窗口")

    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_raw_list, axis=0))
    train_scaled_list = [scaler.transform(x) for x in train_raw_list]
    val_scaled_list = [scaler.transform(x) for x in val_raw_list]
    test_scaled_list = [scaler.transform(x) for x in test_raw_list]

    train_ds = MultiFileSlidingWindowDataset(train_scaled_list, target_idx, known_future_idx, seq_len, pred_len)
    val_ds = MultiFileSlidingWindowDataset(val_scaled_list, target_idx, known_future_idx, seq_len, pred_len)
    test_ds = MultiFileSlidingWindowDataset(test_scaled_list, target_idx, known_future_idx, seq_len, pred_len)

    return DatasetBundle(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, scaler=scaler, feature_names=feature_names, target_idx=target_idx, known_future_idx=list(known_future_idx), periodic_known_idx=list(periodic_known_idx), used_files=used_files)
