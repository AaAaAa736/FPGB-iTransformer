"""FPGB-iTransformer 主模型。

使用标准 Scaled Dot-Product Softmax 自注意力（O(N²)）作为编码器主干，
配合以下组件：

1. RevIN 归一化（Reversible Instance Normalization）
2. 三支路融合（历史重构 / 已知未来编码 / 周期上下文门控）
3. 倒置编码（iTransformer 的 Inverted Embedding）

模型固定使用完整三支路结构，不包含消融开关或消融变体。
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def norm(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, mean, std

    def denorm_target(self, y, mean, std, target_channel=0):
        if self.affine:
            weight = self.affine_weight[:, :, target_channel]
            bias = self.affine_bias[:, :, target_channel]
            y = (y - bias) / (weight + self.eps)
        target_mean = mean[:, :, target_channel]
        target_std = std[:, :, target_channel]
        return y * target_std + target_mean


class UnknownHistoryReconstructor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        return self.proj(x)


class FutureKnownEncoder(nn.Module):
    def __init__(self, pred_len, known_dim, out_len):
        super().__init__()
        if known_dim <= 0:
            raise ValueError("已知未来特征数 known_dim 必须 > 0")
        self.time_proj = nn.Linear(pred_len, out_len)
        self.act = nn.GELU()

    def forward(self, x_future_known):
        x = x_future_known.transpose(1, 2)
        x = self.time_proj(x)
        x = self.act(x)
        return x.transpose(1, 2)


class GenericPeriodicContextGate(nn.Module):
    def __init__(self, pred_len, periodic_known_idx, branch_dims: Sequence[int], dropout=0.1):
        super().__init__()
        self.periodic_known_idx = list(periodic_known_idx)
        if len(self.periodic_known_idx) == 0:
            raise ValueError("periodic_known_idx 不能为空")
        in_dim = pred_len * len(self.periodic_known_idx)
        self.context_encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(64, dim), nn.Sigmoid()) for dim in branch_dims
        ])

    def forward(self, x_future_known, features):
        periodic_part = x_future_known[:, :, self.periodic_known_idx]
        bsz = periodic_part.size(0)
        ctx = self.context_encoder(periodic_part.reshape(bsz, -1))
        gated = []
        for feat, gate in zip(features, self.gates):
            g = gate(ctx).unsqueeze(1)
            gated.append(feat * g)
        return gated


class DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class SoftmaxSelfAttention(nn.Module):
    """标准 Scaled Dot-Product Multi-Head Self-Attention（O(N²)）。"""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, n_tok, dim = x.shape
        h = self.n_heads
        dh = self.head_dim

        q = self.q_proj(x).view(bsz, n_tok, h, dh).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(x).view(bsz, n_tok, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_tok, h, dh).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                                 # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(bsz, n_tok, dim)       # (B, N, D_model)
        return self.out_proj(out)


class SoftmaxAttentionEncoderLayer(nn.Module):
    """单层 Softmax 自注意力编码器层（Self-Attention + FFN + LayerNorm）。"""

    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = SoftmaxSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class SoftmaxAttentionEncoder(nn.Module):
    """多层 Softmax 自注意力编码器。"""

    def __init__(self, d_model, n_heads, e_layers, d_ff=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SoftmaxAttentionEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KnownFutureUnknownHistoryFusion(nn.Module):
    """三支路融合：目标特征 + 未知历史重构 + 已知未来编码，并用周期门控加权。"""

    def __init__(
        self,
        seq_len,
        pred_len,
        hist_input_dim,
        target_idx,
        known_future_dim,
        periodic_known_idx,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hist_input_dim = hist_input_dim
        self.target_idx = target_idx
        self.known_future_dim = known_future_dim
        self.periodic_known_idx = list(periodic_known_idx)
        self.unknown_encoder = UnknownHistoryReconstructor(in_dim=hist_input_dim)
        self.future_encoder = FutureKnownEncoder(pred_len=pred_len, known_dim=known_future_dim, out_len=seq_len)
        self.periodic_gate = None
        if len(self.periodic_known_idx) > 0:
            self.periodic_gate = GenericPeriodicContextGate(
                pred_len=pred_len,
                periodic_known_idx=periodic_known_idx,
                branch_dims=[1, hist_input_dim, known_future_dim],
                dropout=dropout,
            )
        self.out_dim = 1 + hist_input_dim + known_future_dim

    def forward(self, x_hist, x_future_known):
        target_feat = x_hist[:, :, self.target_idx:self.target_idx + 1]
        unknown_feat = self.unknown_encoder(x_hist)
        future_feat = self.future_encoder(x_future_known)
        if self.periodic_gate is not None:
            target_feat, unknown_feat, future_feat = self.periodic_gate(
                x_future_known,
                [target_feat, unknown_feat, future_feat],
            )
        return torch.cat([target_feat, unknown_feat, future_feat], dim=-1)


class ITransformerWithKnownFuture(nn.Module):
    """FPGB-iTransformer 主模型。"""

    def __init__(
        self,
        seq_len,
        pred_len,
        hist_input_dim,
        target_idx,
        known_future_dim,
        periodic_known_idx: Sequence[int],
        d_model=128,
        n_heads=4,
        e_layers=3,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hist_input_dim = hist_input_dim
        self.target_idx = target_idx
        self.revin = RevIN(num_features=hist_input_dim, affine=True)

        self.input_fusion = KnownFutureUnknownHistoryFusion(
            seq_len=seq_len,
            pred_len=pred_len,
            hist_input_dim=hist_input_dim,
            target_idx=target_idx,
            known_future_dim=known_future_dim,
            periodic_known_idx=periodic_known_idx,
            dropout=dropout,
        )
        self.concat_dim = self.input_fusion.out_dim
        self.embedding = DataEmbeddingInverted(seq_len=seq_len, d_model=d_model, dropout=dropout)
        self.encoder = SoftmaxAttentionEncoder(
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.concat_dim * d_model),
            nn.Linear(self.concat_dim * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x_hist, x_future_known):
        x_hist, mean, std = self.revin.norm(x_hist)
        z = self.input_fusion(x_hist, x_future_known)
        z = self.embedding(z)
        z = self.encoder(z)
        z = z.reshape(z.size(0), -1)
        y = self.head(z)
        return self.revin.denorm_target(y, mean, std, target_channel=self.target_idx)
