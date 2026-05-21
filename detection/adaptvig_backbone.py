import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

import numpy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'adaptvig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}
    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()
        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1)
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4)
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pw2(x)
        x = self.norm3(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x


class AdaptConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super().__init__()
        self.K = K
        
        # Learnable temperature parameter for the exponential decay
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x # Initialize with zeros

        # 1. Guaranteed Local Connections (Always On)
        x_c_local = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c_local - x)
        x_r_local = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r_local - x)

        # 2. Gated Long-Range Connections using Exponential Decay
        Hbit, Wbit = H.bit_length(), W.bit_length()
        for i in range(1, Hbit):
            dist_exp = 2**i
            neighbor = torch.cat([x[:, :, -dist_exp:, :], x[:, :, :-dist_exp, :]], dim=2)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            
            # Exponential Decay Gate
            # Use abs() + eps on temperature to ensure it's a valid, non-zero divisor
            gate = torch.exp(-dist / (torch.abs(self.temperature) + 1e-6))
            
            # Apply soft gate directly
            x_j = torch.max(x_j, (neighbor - x) * gate)

        for i in range(1, Wbit):
            dist_exp = 2**i
            neighbor = torch.cat([x[:, :, :, -dist_exp:], x[:, :, :, :-dist_exp]], dim=3)
            dist = torch.norm(x - neighbor, p=1, dim=1, keepdim=True)
            gate = torch.exp(-dist / (torch.abs(self.temperature) + 1e-6))
            x_j = torch.max(x_j, (neighbor - x) * gate)
            
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class GlobalAttentionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super().__init__()
        self.K = K
        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Add Layer Normalization for stability
        self.q_norm = nn.LayerNorm(in_channels // 4)
        self.k_norm = nn.LayerNorm(in_channels // 4)

        self.softmax = nn.Softmax(dim=-1)
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x 
        
        # Project to Q, K, V
        q = self.q_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.k_conv(x).view(B, -1, H*W).permute(0, 2, 1) # Permute to (B, N, C) for norm
        v = self.v_conv(x).view(B, -1, H*W).permute(0, 2, 1)

        # Normalize Q and K before the dot product
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled Dot-Product Attention
        attn = self.softmax(torch.bmm(q, k.transpose(-2, -1)) / (C//4)**0.5)
        
        x_attn = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        
        x_j = torch.max(x_j, x_attn - x)
        
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class ConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            bias=True, groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x


class AdaptiveGrapher(nn.Module):
    def __init__(self, in_channels, K, graph_conv_type='gating'):
        super(AdaptiveGrapher, self).__init__()
        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        if graph_conv_type == 'gating':
            self.graph_conv = AdaptConv(in_channels, in_channels, K=K)
        else: # 'global'
            self.graph_conv = GlobalAttentionGraphConv(in_channels, in_channels, K=K)

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class AdaptiveGraphConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, graph_conv_type='gating', use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.mixer = AdaptiveGrapher(in_dim, K, graph_conv_type)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self, x):
        return self.conv(x)


class AdaptViG(torch.nn.Module):
    def __init__(self, blocks, channels, kernels, stride,
                 act_func, dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000,
                 pretrained=None, out_indices=None):
        super(AdaptViG, self).__init__()
        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained

        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        dpr_idx = 0
        self.stem = Stem(input_dim=3, output_dim=channels[0])
        self.backbone = nn.ModuleList()
        num_stages = len(blocks)
        for i in range(num_stages):
            stage = []
            local_stages, global_stages = blocks[i]
            graph_conv_type = 'global' if i == num_stages - 1 else 'gating'
            if i > 0:
                stage.append(Downsample(channels[i-1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=kernels, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(AdaptiveGraphConvBlock(channels[i], drop_path=dpr[dpr_idx], K=K[i], graph_conv_type=graph_conv_type))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def init_weights(self):
        logger = get_root_logger()
        print("Pretrained weights being loaded")
        logger.warn('Pretrained weights being loaded')
        ckpt_path = self.pretrained
        ckpt = _load_checkpoint(
            ckpt_path, logger=logger, map_location='cpu')
        print("ckpt keys: ", ckpt.keys())
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict_ema']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt

        state_dict = _state_dict
        missing_keys, unexpected_keys = \
            self.load_state_dict(state_dict, False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        B, C, H, W = x.shape

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs

if has_mmdet:
    @det_BACKBONES.register_module()
    def adaptvig_m_feat(pretrained=True, **kwargs):
        model = AdaptViG(blocks=[[4,4], [4,4], [12,4], [4,4]],
                        channels=[48, 96, 192, 320],
                        kernels=3,
                        stride=1,
                        act_func='gelu',
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=[8, 4, 2, 1],
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/AdaptViG_M/model_best.pth')
        model.default_cfg = default_cfgs['adaptvig']
        return model

    @det_BACKBONES.register_module()
    def adaptvig_b_feat(pretrained=True, **kwargs):
        model = AdaptViG(blocks=[[5,5], [5,5], [15,5], [5,5]],
                        channels=[48, 96, 192, 384],
                        kernels=3,
                        stride=1,
                        act_func='gelu',
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=[8, 4, 2, 1],
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/AdaptViG_B/model_best.pth')
        model.default_cfg = default_cfgs['adaptvig']
        return model
