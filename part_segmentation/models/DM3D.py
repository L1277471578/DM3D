from typing import Optional, Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message


from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation


from part_segmentation.models.bimamba_ssm.modules.def_mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz)  # B N K : get K idx for every center
        assert idx.size(1) == num_points  # N center
        assert idx.size(2) == self.group_size  # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()  # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]  # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size,
                                                   feat.shape[-1]).contiguous()  # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, neighborhood_feat


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Local Geometry Aggregation
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # get knn xyz and feature
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)  # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(-2)  # B G 1 C


        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        # knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C
        knn_x = torch.cat([knn_x, lc_x.view(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w


# Max Pooling
class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w, knn_xyz):
        geo_weight = torch.exp(-torch.norm(knn_xyz, dim=-1, keepdim=True))
        geo_weight = geo_weight.permute(0, 3, 1, 2)  # B 1 G K
        geo_weight = geo_weight.expand_as(knn_x_w)  # B 2C G K
        weighted_knn_x_w = knn_x_w * geo_weight

        # Feature Aggregation (Pooling)

        e_x = torch.exp(weighted_knn_x_w)  # B 2C G K
        up = (weighted_knn_x_w * e_x).mean(-1)  # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x


# shared MLP
class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute

    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)


## MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# K_Norm + K_Pool + Shared MLP
class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128,
                 act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm, ):
        super().__init__()

        '''
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x
        '''

        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()

    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        cls_token = feat[:, 0, :].view(B, 1, C)
        feat = feat[:, 1:, :]  # B G C

        knn_xyz, knn_x = self.lga.group_feat(center, feat)
        lc_x_w = self.lga(center, feat)  # B 2C G K

        lc_x_w = self.kpool(lc_x_w,knn_xyz)  # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))  # pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))  # B G C : 1 128 384

        lc_x = self.act(lc_x)

        lc_x = torch.cat((cls_token, lc_x), dim=1)  # B G+1 C : 1 129 384
        return lc_x


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.2,
            k_group_size=32, alpha=100, beta=1000, num_group=128,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
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

        self.norm1 = norm_cls(dim)
        self.lnp = LNPBlock(
            lga_out_dim=dim * 2,
            k_group_size=k_group_size,
            alpha=alpha,
            beta=beta,
            mlp_in_dim=dim * 2,
            mlp_out_dim=dim,
            num_group=num_group,
            act_layer=nn.SiLU,
            drop_path=drop_path,
            norm_layer=norm_cls
        )

        self.mixer = mixer_cls(dim)
        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim)
        self.dp = nn.Dropout(0.5)


        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, center: Optional[Tensor] = None,
                inference_params=None
                ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        # if not self.fused_add_norm:
        #     residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
        #     hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        #     if self.residual_in_fp32:
        #         residual = residual.to(torch.float32)
        # else:
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
        #     hidden_states, residual = fused_add_norm_fn(
        #         self.drop_path(hidden_states),
        #         self.norm.weight,
        #         self.norm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         residual_in_fp32=self.residual_in_fp32,
        #         eps=self.norm.eps,
        #     )

        # x + norm(x)->lfa(x)->dropout
        hidden_states = hidden_states + self.drop_path(self.lnp(center, self.norm1(hidden_states)))  # x: 32 129 384. center: 32 128 3
        # hidden_states=self.dp(hidden_states)
        # x + norm(x)->mamba(x)->dropout
        # x = x + self.drop_path(self.mamba_shuffle(x))
        hidden_states = self.mixer(self.norm2(hidden_states),coords=center)
        hidden_states = hidden_states + self.drop_path(hidden_states )
        hidden_states=self.dp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
        k_group_size=8,
        alpha=100,
        beta=1000,
        num_group=128,
        **kwargs
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
        k_group_size=k_group_size,
        alpha=alpha,
        beta=beta,
        num_group=num_group,
    )
    block.layer_idx = layer_idx
    return block


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MixerModelForSegmentation(MixerModel):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.fetch_idx = fetch_idx

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input_ids, pos, center, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None

        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, center=center, inference_params=inference_params
            )

            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list

from models.pointnet2_utils import PointNetFeaturePropagation
class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q

class get_model(nn.Module):
    def __init__(self, cls_dim, config):
        super().__init__()

        self.trans_dim = config.trans_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.depth = config.depth
        self.cls_dim = cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)
        self.drop_path_rate = config.drop_path_rate
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.norm = nn.LayerNorm(self.trans_dim).float()

        # self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(64),
                                        # nn.GroupNorm(num_groups=1, num_channels=64),
                                        # nn.LeakyReLU(0.2))

        # self.propagation_0 = PointNetFeaturePropagation(in_channel=384 + 3, mlp=[self.trans_dim * 4, 1024])
        #
        # self.convs1 = nn.Conv1d(1856, 256, 1)
        #
        # self.bns1 = nn.BatchNorm1d(256)
        # self.dp1 = nn.Dropout(0.5)
        # self.dp2 = nn.Dropout(0.5)
        # # self.convs2 = nn.Conv1d(512, 256, 1)
        # self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        # # self.convs3 = nn.Conv1d(512, self.cls_dim, 1)
        # # self.bns1 = nn.BatchNorm1d(256)
        # # self.bns2 = nn.BatchNorm1d(256)
        #
        # self.relu = nn.ReLU()

        self.propagation_2 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_1 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3 + 16,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()


    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu')

            if 'model_state_dict' in ckpt:
                base_ckpt = ckpt['model_state_dict']
            else:
                base_ckpt = ckpt

            # 移除可能存在的'module.'前缀
            base_ckpt = {k.replace("module.", ""): v for k, v in base_ckpt.items()}

            # 获取当前模型的状态字典
            model_dict = self.state_dict()

            # 筛选兼容的权重
            compatible_weights = {}
            for k, v in base_ckpt.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        compatible_weights[k] = v
                        print(f"Loading compatible parameter: {k}")
                    else:
                        print(f"Skipping incompatible parameter: {k} (expected {model_dict[k].shape}, got {v.shape})")
                else:
                    print(f"Skipping unexpected parameter: {k}")

            # 更新模型状态字典
            model_dict.update(compatible_weights)

            # 加载更新后的状态字典
            self.load_state_dict(model_dict)
            print(f'Partially loaded checkpoint from {bert_ckpt_path}')

            # 统计加载情况
            total_params = len(base_ckpt)
            loaded_params = len(compatible_weights)
            print(f"Loaded {loaded_params}/{total_params} parameters ({loaded_params / total_params * 100:.2f}%)")
        else:
            print(f'No ckpt is loaded, training from scratch!')



    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # pos = self.pos_embed(center)

        cls_tokens = self.cls_token.expand(B,-1,-1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat([cls_tokens ,group_input_tokens],dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)

        feature_list = self.blocks(x, pos, center=center)
        feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]

        # x = torch.cat((feature_list), dim=1)  # 1152
        # x_max = torch.max(x, 2)[0]
        # x_avg = torch.mean(x, 2)
        # x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(B, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)
        #
        # f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
        #
        # x = torch.cat((f_level_0, x_global_feature), 1)
        #
        # # x = self.dp1(x)
        # x = self.relu(self.bns1(self.convs1(x)))
        # x = self.dp1(x)
        # # x = self.relu(self.bns2(self.convs2(x)))
        # # print(x.shape)
        # x = self.convs3(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # return x

        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        center_level_0 = pts.transpose(-1, -2).contiguous()
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

        center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()

        # init the feature by 3nn propagation
        f_level_3 = feature_list[2]
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])

        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # FC layers
        feat =  F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss