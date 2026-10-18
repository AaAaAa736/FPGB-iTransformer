# FPGB-iTransformer

面向电力负荷预测的 iTransformer 变体。在标准 iTransformer 的基础上引入了三个关键组件，并使用标准 Scaled Dot-Product Softmax 自注意力作为主干。

## 模型简介

FPGB-iTransformer 在标准 iTransformer 的倒置编码（inverted encoding）框架上，额外引入：

1. **未知历史重构分支（Unknown History Reconstructor）**：对历史未知特征做线性重构，增强历史输入的表示能力。
2. **已知未来特征编码分支（Future Known Encoder）**：将预测区间内的已知未来特征（温度、星期/小时周期、节假日类型等）编码后与历史特征对齐融合。
3. **周期上下文门控（Periodic Context Gate）**：以周期型已知未来特征为上下文，对目标、历史、未来三支路做门控加权。

此外还使用了 RevIN 归一化与三支路融合（Three-Branch Fusion），主干为 Softmax 多头自注意力编码器。

## 目录结构

```
FPGB-iTransformer/
├── data/
│   └── ieso_ontario_hourly_demand_2002_2024.csv   # Ontario IESO 小时级负荷数据
├── data_utils.py                                  # 数据加载与标准化
├── engine.py                                      # 训练引擎
├── losses_and_metrics.py                          # 损失函数与评估指标
├── modeling_softmax.py                            # FPGB-iTransformer 模型定义
├── run_fpgb_itransformer.py                       # 主入口：完整训练流程
└── README.md
```

## 环境依赖

- Python 3.10+
- PyTorch
- NumPy
- pandas

```bash
pip install torch numpy pandas
```

## 数据

由于PJM数据集文件过大，在该项目中只公开Ontario IESO数据集，PJM数据集另行公开。数据为 Ontario IESO 小时级电力负荷（2002–2024），存放在 `data/ieso_ontario_hourly_demand_2002_2024.csv`。

原始 CSV 的列包括：

- `timestamp_start_est`：整点时间戳（东部标准时间）
- `ontario_demand_mw`：负荷目标列（MW）
- 天气特征（湿度、降雪、云量、短波辐射、温度等）
- 节假日类型（工作日 / 周六 / 周日）

脚本会读取原始 CSV，构造小时/星期/年日周期特征与节假日独热编码，并按 0.70 / 0.15 / 0.15 切分训练 / 验证 / 测试集。

## 快速开始

```bash
# 完整流程：3 个 horizon（24/48/96）× 5 个种子，先搜索后正式训练
python run_fpgb_itransformer.py

# 只跑 24 步 horizon，使用 CPU
python run_fpgb_itransformer.py --horizons 24 --cpu

# 冒烟测试：只验证数据加载与模型前向，不训练
python run_fpgb_itransformer.py --smoke_only
```

## 实验流程

脚本是一个完整的训练流程：对每个 horizon，自动先搜索最佳超参数，随后直接用该超参数正式训练。

### 超参数搜索（自动完成）

默认随机抽样 **100 组**超参数，每组训练 **100 轮**（可早停），按验证集损失 `best_val_loss` 选出最佳超参数。

搜索范围覆盖：

| 超参数 | 搜索范围 |
| --- | --- |
| `d_model` | 64, 128, 256, 512 |
| `n_heads` | 4, 8, 16 |
| `e_layers` | 2, 3, 4 |
| `d_ff` | 128, 256, 512, 1024 |
| `dropout` | 0.1, 0.2, 0.3 |
| `lr` | 1e-4, 5e-4, 1e-3 |
| `weight_decay` | 1e-5, 1e-4 |

（自动剔除 `d_model` 不能被 `n_heads` 整除的非法组合。）

### 正式训练

用搜索得到的最佳超参数，对每个 horizon 分别训练 **5 个随机种子**（42, 123, 456, 789, 135），默认 30 轮、早停 patience 6。

## 输出结果

所有结果保存在 `outputs/` 目录：

```
outputs/
├── prepared_data/                       # 预处理后的特征表与元数据
├── experiment_manifest.json             # 实验配置
├── pred_24/
│   ├── main_model_search/               # horizon 24 的超参数搜索结果
│   │   ├── search_summary.csv
│   │   └── best_hyperparameters.json
│   └── models/FPGB-iTransformer/
│       └── seed_42/
│           ├── predictions.npz          # 压缩预测结果（pred / true / forecast_start_ns）
│           ├── metrics.json             # MAE / RMSE / MAPE / best_val_loss
│           ├── best.pt                  # 最佳模型权重
│           ├── train_log.csv
│           └── step_metrics.csv
├── pred_48/ ...
├── pred_96/ ...
├── all_seed_metrics.csv                 # 所有 seed 的指标汇总
├── model_summary_by_horizon.csv         # 按 horizon 汇总
└── prediction_index.csv
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--source_csv` | `data/ieso_ontario_hourly_demand_2002_2024.csv` | 原始数据路径 |
| `--save_root` | `outputs` | 结果输出目录 |
| `--horizons` | 24 48 96 | 预测 horizon 列表 |
| `--seeds` | 40 41 42 43 44 45 | 正式训练种子列表 |
| `--seq_len` | 168 | 历史窗口长度 |
| `--train_ratio` | 0.7 | 训练集比例 |
| `--val_ratio` | 0.15 | 验证集比例 |
| `--search_seed` | 42 | 搜索随机抽样种子 |
| `--search_trials` | 100 | 超参数搜索试验组数 |
| `--search_epochs` | 100 | 每组搜索试验的训练轮数 |
| `--search_patience` | 10 | 搜索阶段早停 patience |
| `--formal_epochs` | 30 | 正式训练轮数 |
| `--formal_patience` | 6 | 正式训练早停 patience |
| `--batch_size` | 256 | 批大小 |
| `--cpu` | False | 使用 CPU 训练 |
| `--smoke_only` | False | 冒烟测试 |

## License

MIT License
