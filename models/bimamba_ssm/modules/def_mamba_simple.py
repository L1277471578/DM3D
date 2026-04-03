

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

# try:
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# except ImportError:
#     causal_conv1d_fn, causal_conv1d_update = None

import sys

sys.path.append(r"../../models/bimamba_ssm")
from models.bimamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
    mamba_inner_fn_no_out_proj
# except ImportError:
#     selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

# try:
from models.bimamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None

# try:
from models.bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# from models.utils.structure import Point
import pointnet2_ops.pointnet2_utils as pointnet2_utils
from torch_scatter import scatter_mean
from torchsort import soft_sort
import faiss
from torch_cluster import knn
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, length = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch, channels, length)

class AttentiveAdaptiveFusion(nn.Module):
    def __init__(self, dim: int, bias: bool = False,groups=32) -> None:
        super().__init__()

        self.fuse_proj = nn.Conv1d(dim * 3, dim, kernel_size=1, bias=bias,groups=groups)
        # self.freq_enhance = FrequencyEnhanceBlock(in_channels=dim,groups=groups)
        self.ChannelShuffle =ChannelShuffle(groups)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        w_x = torch.sigmoid(x)
        w_y = torch.sigmoid(y)
        w_z = torch.sigmoid(z)
        x_att = x * (w_y + w_z) / 2
        y_att = y * (w_x + w_z) / 2
        z_att = z * (w_x + w_y) / 2

        fused = torch.cat([x_att, y_att, z_att], dim=1)  # [B, 3D, L]
        fused = self.fuse_proj(fused)  # [B, D, L]
        out = self.ChannelShuffle(fused)

        return out


def _build_local_window(base_idx, radius, N):
    # base_idx: [B, N]
    device = base_idx.device
    offsets = torch.arange(-radius, radius + 1, device=device).view(1, 1, -1)  # [1,1,K]
    local_index = base_idx.unsqueeze(-1) + offsets
    local_index = local_index.clamp(0, N - 1).long()  # [B,N,K]
    return local_index


def _gather_local_feat(features, local_index):
    # features: [B, N, C]
    B, N, C = features.shape
    K = local_index.size(-1)
    batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(B, N, K)
    local_feat = features[batch_idx, local_index]  # [B,N,K,C]
    return local_feat

def sinkhorn_normalization(log_alpha, n_iters=5):
    """
    log_alpha: [..., K, K]
    returns doubly-stochastic matrix approximation
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  # row norm
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  # col norm
    return log_alpha.exp()


def sorting_local_sinkhorn(features, delta_t, base_idx=None, radius=4, tau=0.2, n_iters=5, eps=1e-8):
    B, N, C = features.shape
    device = features.device
    dtype = features.dtype

    if delta_t.shape != (B, N):
        raise ValueError(f"delta_t shape {delta_t.shape} != {(B, N)}")

    if base_idx is None:
        base_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    else:
        if base_idx.shape != (B, N):
            raise ValueError(f"base_idx shape {base_idx.shape} != {(B, N)}")

    base_idx = base_idx.long()
    new_order = base_idx.to(dtype) + delta_t  # [B,N]

    local_index = _build_local_window(base_idx, radius, N)   # [B,N,K]
    local_feat = _gather_local_feat(features, local_index)    # [B,N,K,C]
    local_order = local_index.to(dtype)                       # [B,N,K]

    # shifted positions of all local tokens
    B_, N_, K = local_index.shape
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    local_shift = new_order[batch_idx, local_index]          # [B,N,K]

    # cost[a,b] = distance between local token a's shifted position and local target index b
    cost = (local_shift.unsqueeze(-1) - local_order.unsqueeze(-2)) ** 2  # [B,N,K,K]

    log_alpha = -cost / (tau + 1e-12)
    P = sinkhorn_normalization(log_alpha, n_iters=n_iters)   # [B,N,K,K]

    # center token is always at offset 0, i.e. row = radius
    center_row = radius
    W = P[:, :, center_row, :]                               # [B,N,K]
    W = W / (W.sum(dim=-1, keepdim=True) + eps)

    reordered = (local_feat * W.unsqueeze(-1)).sum(dim=2)
    return reordered, new_order

def sorting_local_gaussian(features, delta_t, base_idx=None, radius=4, sigma=1.0, eps=1e-8):
    B, N, C = features.shape
    device = features.device
    dtype = features.dtype

    if delta_t.shape != (B, N):
        raise ValueError(f"delta_t shape {delta_t.shape} != {(B, N)}")

    if base_idx is None:
        base_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    else:
        if base_idx.shape != (B, N):
            raise ValueError(f"base_idx shape {base_idx.shape} != {(B, N)}")

    base_idx = base_idx.long()
    new_order = base_idx.to(dtype) + delta_t

    offsets = torch.arange(-radius, radius + 1, device=device).view(1, 1, -1)
    local_index = base_idx.unsqueeze(-1) + offsets
    local_index = local_index.clamp(0, N - 1).long()
    K = local_index.size(-1)

    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    local_feat = features[batch_idx, local_index]
    local_order = local_index.to(dtype)

    diff = new_order.unsqueeze(-1) - local_order
    W = torch.exp(-0.5 * (diff / (sigma + 1e-12)) ** 2)
    W = W / (W.sum(dim=-1, keepdim=True) + eps)

    permuted = (local_feat * W.unsqueeze(-1)).sum(dim=2)
    return permuted, new_order


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class DeformableScan3DForPointCloud(nn.Module):
    def __init__(
        self,
        in_channels,
        num_neighbors=4,
        R_t=4,
        dp_scale=1.0,
        enable_dp=True,
        enable_dt=True,
        semantic_ratio=16,   # ??
        use_ca=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_neighbors = num_neighbors
        self.R_t = R_t
        self.dp_scale = dp_scale
        self.enable_dp = enable_dp
        self.enable_dt = enable_dt

        self.semantic_dim = max(4, in_channels // semantic_ratio)
        self.semantic_proj = nn.Linear(in_channels, self.semantic_dim, bias=False)
        offset_in_dim = in_channels + self.semantic_dim

        self.offset_net = nn.Sequential(
            nn.Conv1d(offset_in_dim, offset_in_dim, kernel_size=5, padding=2, groups=offset_in_dim),  # kernel 5 ? 3
            nn.Conv1d(offset_in_dim, in_channels // 4, kernel_size=1),  #  channel
            ChannelAttention(in_channels // 4) if use_ca else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, 4, kernel_size=1),
        )

        self.sigma_t = nn.Parameter(torch.tensor(0.2))
        self.sigma_s = nn.Parameter(torch.tensor(1.0))
        self._save_counter = 0
        self.save_interval = 12

    def forward(self, coords, feats, base_idx=None):
        B, _, C = feats.shape
        device = feats.device

        # ---- CLS ----
        cls_token = feats[:, :1]
        feats = feats[:, 1:]

        B, N, C = feats.shape

        # ---- KNN ----
        k = min(self.num_neighbors, N - 1)
        idx = pointnet2_utils.ball_query(
            0.1, k, coords.contiguous(), coords.contiguous()
        ).view(B, N, k)

        neighbor_feat = pointnet2_utils.grouping_operation(
            feats.permute(0, 2, 1).contiguous(), idx
        ).permute(0, 2, 3, 1)
        #
        center_feat = feats.unsqueeze(2)

        diff_feat = neighbor_feat - center_feat
        local_mean = diff_feat.mean(dim=2)
        local_var  = diff_feat.abs().mean(dim=2)
        semantic_raw = (local_mean.abs() + local_var).detach()
        semantic = self.semantic_proj(semantic_raw)

        #  offset
        offset_input = torch.cat([local_mean + local_var + 0.1* feats, semantic], dim=-1)
        offset = self.offset_net(offset_input.permute(0, 2, 1))
        # offset = feats
        offset = offset.permute(0, 2, 1)
        # offset_np = offset.detach().cpu().numpy()
        # first3 = offset_np[..., :3]
        # fourth = offset_np[..., 3]
        delta_p = torch.tanh(offset[..., :3]) * self.dp_scale if self.enable_dp else 0
        delta_t = torch.tanh(offset[..., 3]) if self.enable_dt else 0
        # self._save_counter += 1
        # if self._save_counter % self.save_interval == 0:
        #     with open('first3.txt', 'ab') as f:
        #         np.savetxt(f, first3.reshape(-1, 3), fmt='%.6f', delimiter=' ')
        #     with open('fourth.txt', 'ab') as f:
        #         np.savetxt(f, fourth.reshape(-1, 1), fmt='%.6f', delimiter=' ')
        #     with open('offset_p.txt', 'ab') as f:
        #         np.savetxt(f, first3.reshape(-1, 3), fmt='%.6f', delimiter=' ')
        #     with open('offset_t.txt', 'ab') as f:
        #         np.savetxt(f, fourth.reshape(-1, 1), fmt='%.6f', delimiter=' ')

        # DSR
        neighbor_pos = pointnet2_utils.grouping_operation(
            coords.permute(0, 2, 1).contiguous(), idx
        ).permute(0, 2, 3, 1)

        center_pos = coords.unsqueeze(2)
        pos_diff = neighbor_pos - (center_pos + delta_p.unsqueeze(2))
        dist2 = (pos_diff ** 2).sum(dim=-1)

        w = torch.softmax(-dist2 / (self.sigma_s + 1e-12), dim=2)
        resampled_feat = (neighbor_feat * w.unsqueeze(-1)).sum(dim=2)
        gate = torch.sigmoid(feats.norm(dim=-1,keepdim=True))
        resampled_feat =  resampled_feat + feats*gate

        if self.enable_dt:
            ## GDR
            reordered_feat, new_order = sorting_local_gaussian(
                resampled_feat,
                delta_t,
                base_idx=None,
                radius=self.R_t,
                sigma=self.sigma_t,
            )

            ## sinkhorn
            # reordered_feat, new_order = sorting_local_sinkhorn(
            #     resampled_feat, delta_t, base_idx=base_idx,
            #     radius=self.num_neighbors, tau=self.sigma_t,
            #     n_iters=5)
        else:
            reordered_feat = resampled_feat
            new_order = None

        out = torch.cat([cls_token, reordered_feat], dim=1)
        return out, coords + delta_p, new_order

class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            mamba_type="v3",
            enable_dp=False,
            enable_dt=False,
            # num_neighbors = 8
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 16
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)   # 24
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.mamba_type = mamba_type
        self.enable_dp= enable_dp
        self.enable_dt =enable_dt
        # self.num_neighbors = num_neighbors

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        #
        # # ???SSM??
        self.conv1d_d = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            groups=self.d_inner//2,
            padding=1,
            **factory_kwargs,)

        self.A_d = nn.Parameter(
            torch.log(repeat(
                torch.linspace(0.5, 2.0, self.d_state, device=device),
                "n -> d n", d=self.d_inner
            ))
        )
        self.A_d._no_weight_decay = True

        self.x_proj_d = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs
        )
        self.dt_proj_d = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
            **factory_kwargs
        )

        self.point_deform_module = DeformableScan3DForPointCloud(
            in_channels=self.d_model,
            dp_scale=1,
            # resample_k=3,
            # expand=self.expand,
            enable_dp=self.enable_dp,
            enable_dt=self.enable_dt,
        )

        self.AttentiveAdaptiveFusion = AttentiveAdaptiveFusion(dim=768,groups=128)

    def forward(self, hidden_states, inference_params=None, coords=None, base_idx=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """

        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.mamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-2]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )  # [1,768,129]

                out = F.linear(rearrange(out + out_b.flip([-2]), "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)
                return out,coords,base_idx

            elif self.mamba_type == "v3":
                # 1
                out_fwd = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                # 2
                A_b = -torch.exp(self.A_b_log.float())
                out_bwd = mamba_inner_fn_no_out_proj(
                    xz.flip([-2]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )

                new_point_feat,new_coord ,new_order = self.point_deform_module(coords=coords, feats=hidden_states, base_idx=base_idx)
                new_point_feat = new_point_feat.permute(0, 2, 1)

                #  conv1d_d + act
                x_deform_conv = self.act(self.conv1d_d(new_point_feat))  # [B, D_inner, L]

                #  SSM
                # x_dbl_d = self.x_proj_d(rearrange(x_deform_conv, "b d l -> (b l) d"))

                x_list = []
                B,_,_ = x_deform_conv.shape
                for b in range(B):
                    x_b = x_deform_conv[b].permute(1, 0)  # [L, D]
                    x_list.append(x_b)
                x_dbl_d = torch.cat(x_list, dim=0)  # [B*L, D]
                x_dbl_d = self.x_proj_d(x_dbl_d)

                dt_d, B_d, C_d = torch.split(x_dbl_d, [self.dt_rank, self.d_state, self.d_state], -1)

                # dt_proj_d
                dt_d = self.dt_proj_d.weight @ dt_d.t()

                # dt_d = dt_d.view(self.d_inner, dt_d.shape[1])
                B_d = rearrange(B_d, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
                C_d = rearrange(C_d, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
                dt_d = rearrange(dt_d, "d (b l) -> b d l", l=seqlen)

                # dt_d: [B, D_inner, L]
                dt_cls = dt_d[:, :, :1]  # [B, D_inner, 1]
                dt_point = dt_d[:, :, 1:]  # [B, D_inner, N]
                # CASU
                dist = torch.norm(new_coord[:,1:]-new_coord[:,:-1],dim=-1)
                dist = F.pad(dist , (1,0 ), value=0.0)
                geo_scale = 1.0 + torch.tanh(dist)

                dt_point = dt_point * geo_scale.unsqueeze(1)

                dt_d = torch.cat([dt_cls, dt_point],dim=2)

                #  selective_scan_fn? out_def [B, D_def, L]
                out_def = selective_scan_fn(
                    x_deform_conv,
                    dt_d,
                    -torch.exp(self.A_d.float()),
                    B_d,
                    C_d,
                    torch.ones(int(self.d_inner), device=device),
                    delta_softplus=True
                )

                # weight = F.softmax(self.path_weight, dim=0)
                # out = out_fwd  + out_bwd.flip([-2]) + out_def

                out = self.AttentiveAdaptiveFusion(out_fwd, out_bwd.flip([-2]), out_def)
                out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

                return out, new_coord, new_order

            else:
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                return out

        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv:])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")

            out = self.out_proj(y)

            return out

    # same as mamba
    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
