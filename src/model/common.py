import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# =========================================================
# 1. 可见光特征提取器（内部下采样 8 倍，384 -> 48）
# =========================================================
class VisibleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super(VisibleFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),  # 384 -> 192
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1), # 192 -> 96
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1), # 96 -> 48
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# =========================================================
# 2. 空间交叉域注意力模块（CDM）
# =========================================================
class CrossDomainAttention(nn.Module):
    def __init__(self, channels=64):
        super(CrossDomainAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    # ✅ 确保这一行 forward 的缩进与上面的 __init__ 完全对齐
    def forward(self, thermal_feat, vis_feat):
        B, C, H, W = thermal_feat.size()
        
        # 投影并展平空间维度 [B, C, H*W]
        proj_query = self.query_conv(thermal_feat).view(B, C, -1)             
        proj_key   = self.key_conv(vis_feat).view(B, C, -1).permute(0, 2, 1)  
        
        # 计算空间关联矩阵 (Query x Key)
        energy = torch.bmm(proj_key, proj_query)                              
        attention = self.softmax(energy)                                      
        
        # 抽取高频先验 Value
        proj_value = self.value_conv(vis_feat).view(B, C, -1)                 
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))               
        out = out.view(B, C, H, W)                                            
        
        return self.gamma * out + thermal_feat