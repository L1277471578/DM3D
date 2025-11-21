

import math
from typing import Optional

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
        # 重塑并转置以混合组
        x = x.view(batch, self.groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch, channels, length)

class FrequencyEnhanceBlock(nn.Module):
    def __init__(self, in_channels: int,groups=32) -> None:
        super().__init__()
        # 2 * in_channels
        self.conv = nn.Conv1d(2 * in_channels, 2 * in_channels, kernel_size=1,groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        fft_result = torch.fft.fft(x, dim=-1)   # 1D FFT
        real, imag = fft_result.real, fft_result.imag  # [B,C,L]


        freq_cat = torch.cat([real, imag], dim=1)  # [B,2C,L]
        freq_out = self.conv(freq_cat)             # [B,2C,L]


        c = freq_out.shape[1] // 2
        real_new = freq_out[:, :c, :]
        imag_new = freq_out[:, c:, :]


        complex_tensor = torch.complex(real_new, imag_new)
        # IFFT
        x_out = torch.fft.ifft(complex_tensor, dim=-1)  # [B,C,L]
        return x_out.real


class AttentiveAdaptiveFusion(nn.Module):
    def __init__(self, dim: int, bias: bool = False,groups=32) -> None:
        super().__init__()

        self.fuse_proj = nn.Conv1d(dim * 3, dim, kernel_size=1, bias=bias,groups=groups)
        self.freq_enhance = FrequencyEnhanceBlock(in_channels=dim,groups=groups)
        self.ChannelShuffle =ChannelShuffle(groups)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # 注意力权重
        w_x = torch.sigmoid(x)
        w_y = torch.sigmoid(y)
        w_z = torch.sigmoid(z)

        x_att = x * (w_y + w_z) / 2
        y_att = y * (w_x + w_z) / 2
        z_att = z * (w_x + w_y) / 2

        fused = torch.cat([x_att, y_att, z_att], dim=1)  # [B, 3D, L]
        fused = self.fuse_proj(fused)  # [B, D, L]
        fused = self.ChannelShuffle(fused)

        out = self.freq_enhance(fused)  # [B, D, L]
        return out


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
            # out_channels,
            num_neighbors=16,
            dp_scale=1,
            use_ca=True,
            resample_k=3,  #
            temperature=0.01,  #
            enable_dp=True,
            enable_dt=True,
            expand=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_neighbors = num_neighbors
        self.dp_scale = dp_scale
        self.resample_k = resample_k
        self.temperature = temperature
        self.expand = expand
        self.enable_dp = enable_dp
        self.enable_dt = enable_dt
        self.use_ca = use_ca
        # self.out_channels = out_channels

        #
        self.dynamic_conv = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, kernel_size=1),
            nn.GELU(),
            # nn.Linear(in_channels, in_channels )
        )

        #  △x,△y,△z,△t
        self.offset_net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels,kernel_size=5,padding=2,groups=in_channels),
            nn.Conv1d(in_channels,in_channels,kernel_size=1),
            ChannelAttention(in_channels) if use_ca else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(in_channels, 4, kernel_size=1)
        )

        self.sigma = nn.Parameter(torch.tensor(0.2))


    def knn_resample(self, new_coord, original_coord, original_feat, batch, k=3, temp=0.04):

        assert new_coord.device == original_coord.device == original_feat.device == batch.device

        batch_center = batch[:new_coord.size(0)]  # [M]


        dists = torch.cdist(new_coord, original_coord)  # [M, N]

        mask = batch_center.unsqueeze(1) != batch.unsqueeze(0)
        dists = dists.masked_fill(mask, float('inf'))


        topk = torch.topk(dists, k=k, dim=1, largest=False)
        weights = F.softmax(-topk.values / temp, dim=1)  # [M, k]


        knn_indices = topk.indices  # [M, k]

        knn_feat = original_feat[knn_indices.view(-1)]  # [M*k, C]
        knn_feat = knn_feat.view(knn_indices.size(0), knn_indices.size(1), -1)  # [M, k, C]

        #
        return (knn_feat * weights.unsqueeze(-1)).sum(dim=1)  # [M, C]

    def forward(self, coords: torch.Tensor, feats: torch.Tensor):
        """
            coords:  [B, N, 3]
            feats:  [B, N, C]
            global_ctx:  [B, C]
        """
        B, _, C = feats.shape
        cls_token = feats[:, 0, :].view(B, 1, C)
        feats = feats[:, 1:, :]  # B G C
        B, N, C = feats.shape
        device = coords.device  #

        #  [B*N, 3]
        flat_coords = coords.view(-1, 3)

        #
        try:
            k = min(self.num_neighbors, N - 1)
            if k > 0:
                idx = pointnet2_utils.ball_query(
                    0.1, k, coords, coords
                ).view(B, N, k)
                idx = idx.to(device)
            else:
                idx = torch.zeros(B, N, k, device=device, dtype=torch.long)
        except Exception as e:
            print(f"ball_Q: {e}")
            idx = torch.zeros(B, N, self.num_neighbors, device=device, dtype=torch.long)

        #
        feats_trans = feats.permute(0, 2, 1).contiguous()
        group_feat = pointnet2_utils.grouping_operation(feats_trans, idx)
        group_feat = group_feat.permute(0, 2, 1, 3)

        #  [B, N, C, k]
        center_feat = feats.unsqueeze(-1).expand(-1, -1, -1, k)

        #  [B, N, 2C, k]
        local_feat = torch.cat([center_feat, group_feat], dim=2)

        #
        local_feat_flat = local_feat.view(B * N, -1, k)
        dynamic_weights = self.dynamic_conv(local_feat_flat)

        #  [B*N, C]
        context_feat = torch.sum(group_feat.contiguous().view(B * N, C, k) * dynamic_weights, dim=2)

        # offset_input = torch.cat([
        #     feats.reshape(B * N, C),
        #     context_feat,
        # ], dim=1)
        offset_input=feats.reshape(B * N, C)+context_feat
        #  [1, 3C, B*N]
        offset_input = offset_input.unsqueeze(0).permute(0, 2, 1)

        #  [1, 4, B*N]
        offset = self.offset_net(offset_input).squeeze(0).permute(1, 0)

        delta_p = offset1[:, :3] if self.enable_dp else torch.zeros_like(flat_coords, device=device)
        delta_p = torch.tanh(delta_p) * self.dp_scale

        # offset = torch.tanh(offset) * self.dp_scale

        # delta_p = offset[:, :3] if self.enable_dp else torch.zeros_like(flat_coords, device=device)
        delta_t = offset[:, 3] if self.enable_dt else torch.zeros(B * N, device=device)

        new_coord = flat_coords + delta_p

        batch_idx = torch.arange(B, device=device).repeat_interleave(N)
        resampled_feat = self.knn_resample(
            new_coord,
            flat_coords,
            feats.reshape(B * N, C),
            batch_idx,
            k=self.resample_k,
            temp=self.temperature
        )


        resampled_feat += feats.reshape(B * N, C)

        if self.enable_dt:
        #     delta_t: [B*N] -> [B,N]
            delta_t = delta_t.view(B, N)
            device = delta_t.device
            base_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # [B, N]
            pos = base_idx + delta_t  # [B, N]

            u = torch.arange(N, device=device).float().unsqueeze(0).expand(B, N)  # [B, N]

            sigma = self.sigma  # 超参，可以调
            diff = u.unsqueeze(-1) - pos.unsqueeze(1)  # [B, N(target), N(source)]

            # 高斯权重
            weights = torch.exp(-0.5 * (diff / sigma) ** 2)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            feats_2d = resampled_feat.view(B, N, C)  # [B, N, C]
            resampled = torch.einsum('btk,bkc->btc', weights, feats_2d)  # [B, N, C]

            resampled_feat = resampled
        # # #

        resampled_feat = resampled_feat.reshape(B, N, C)
        # new_coord = new_coord.reshape(B, N, 3)
        resampled_feat = torch.cat((cls_token, resampled_feat), dim=1)  # B G+1 C : 1 129 384
        return resampled_feat




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
            num_neighbors=8
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
        self.num_neighbors = num_neighbors

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
        self.conv1d_d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            groups=self.d_inner,
            padding=1,
            **factory_kwargs,
        )

        self.A_d = nn.Parameter(
            torch.log(repeat(
                torch.linspace(0.5, 2.0, self.d_state, device=device),
                "n -> d n", d=self.d_inner
            ))
        )
        self.A_d._no_weight_decay = True

        #
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
            # out_channels=self.d_inner,
            num_neighbors=self.num_neighbors,
            dp_scale=0.1,
            use_ca=True,
            resample_k=3,
            temperature=0.01,
            expand=self.expand,
            enable_dp=self.enable_dp,
            enable_dt=self.enable_dt,
        )

        self.linear = nn.Conv1d(self.d_model, self.d_inner,kernel_size=1)

        self.AttentiveAdaptiveFusion = AttentiveAdaptiveFusion(dim=768, groups=128) #768 32

        #
        # self.gate = nn.Linear(
        #     self.d_inner * 3,  # 3个路径
        #     3,  # 3个权重
        #     bias=False,
        #     **factory_kwargs
        # )
        # self.path_weight = nn.Parameter(torch.ones(3))

    def forward(self, hidden_states, inference_params=None, coords=None):
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
        )  # ([8, 1536, 129])

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
                return out

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

                # 2. 反向路径
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


                new_point_feat = self.point_deform_module(coords=coords, feats=hidden_states)
                new_point_feat = new_point_feat.permute(0, 2, 1)
                new_point_feat = self.linear(new_point_feat)
                # feat_out = new_point_feat.reshape(batch, seqlen, -1)  # [N, C']

                # x_deform = rearrange(feat_out, "(b l) d -> b d l", b=batch, l=seqlen)

                x_deform_conv = self.act(self.conv1d_d(new_point_feat))  # [B, D_inner, L]

                x_dbl_d = self.x_proj_d(rearrange(x_deform_conv, "b d l -> (b l) d"))
                dt_d, B_d, C_d = torch.split(x_dbl_d, [self.dt_rank, self.d_state, self.d_state], -1)

                dt_d = self.dt_proj_d.weight @ dt_d.t()

                # dt_d = dt_d.view(self.d_inner, dt_d.shape[1])
                B_d = rearrange(B_d, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
                C_d = rearrange(C_d, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
                dt_d = rearrange(dt_d, "d (b l) -> b d l", l=seqlen)

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
                # out = out_fwd * weight[0] + out_bwd.flip([-2]) * weight[1] + out_def * weight[2]

                out = self.AttentiveAdaptiveFusion(out_fwd, out_bwd.flip([-2]), out_def)

                out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                end_time3 = time.time()


                return out


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