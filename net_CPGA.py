import os
import cv2
import glob
import math
import einops
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as tf
from torch.autograd import Function
import torch.nn.init as init
from thop import profile
from  einops.layers.torch import Rearrange, Reduce
from ops.dcn.deform_conv import ModulatedDeformConv
warnings.filterwarnings("ignore") 


class AU(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(AU, self).__init__()
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias))
        self.conv_fuse = nn.Sequential(nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),)

    def modeling(self, x, res):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # sprint('res',res.shape)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(res)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, res, agg):
        # [N, C, 1, 1]
        context = self.modeling(x, res)
        # context = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        if agg != None:
            x = self.conv_fuse(torch.cat([x,agg],dim=1)) + channel_add_term
        else:
            x = x + channel_add_term

        return x


### --------- Aggregation Unit Block ----------
class AggUnit(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(AggUnit, self).__init__()
        self.act = nn.LeakyReLU(0.2)
        self.gcnet = AU(n_feat, bias=bias)
        self.tail = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)

    def forward(self, x, res, agg=None):
        res = self.act(self.gcnet(x, res, agg))
        res = self.tail(res) + x
        return res


class UAggNet(nn.Module):
    def __init__(self, nf):
        super(UAggNet, self).__init__()
        self.down1 = nn.Conv2d(nf,nf, kernel_size=3, stride=2, padding=3//2, bias =True)
        self.down2 = nn.Conv2d(nf,nf, kernel_size=3, stride=2, padding=3//2, bias =True)
        self.up1 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1)
        self.AggUnit = AggUnit(nf)

    def forward(self, x, res):

        x1 = self.down1(x)
        res1 = self.down1(res)
        x2 = self.down2(x1)
        res2 = self.down2(res1)
        agg2 = self.AggUnit(x2,res2, None)  ## level 2
        agg2up = self.up2(agg2)
        agg1up = self.AggUnit(x1,res1, agg2up)  ## level 1
        agg0up = self.up2(agg1up)
        agg0 = self.AggUnit(x,res, agg0up)   ## level 0

        return agg0



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output



class SIFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ## For Spatial mask
        self.res_conv = nn.Sequential(nn.Conv2d(1, channels, 3, 1, 1),
                                    )
        self.UAggNet = UAggNet(nf=channels)

    def forward(self, feat, res):
        R_M = self.res_conv(res)
        # Attention mask generator
        out = self.UAggNet(feat, R_M)
        return out


class TIFM(nn.Module):
    def __init__(self, in_nc=7, out_nc=64, nf=64, radius=3, base_ks=3, deform_ks=3):
        super(TIFM, self).__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.radius = radius
        self.backbone = nn.Sequential(nn.Conv2d(1, nf, 3, 1, 1),
                            nn.Conv2d(nf, nf, 3, 1, 1),
                            )
        self.aggblock =  DualTemporalFE(nf,nf) 
        self.offset_mask = nn.Conv2d( nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2 )
        self.deform_conv = ModulatedDeformConv(in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc)

    def forward(self, lqs, preds, mv):
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks
        T = 2*self.radius + 1
        ### feature extraction
        n, cT, h, w =  lqs.shape
        lqs_ = lqs.view(-1, 1, h, w)
        feat = self.backbone(lqs_)
        BT, C, H, W = feat.shape
        pred_feat = self.backbone(preds)
        feat01 = feat.new_zeros(BT, C, H, W).view(-1, T, C, H, W)
        B, T, C, H, W = feat01.shape
        # aligned feat from MV  aligned 1;  7 frames

        alg_MV_fea = feat.new_zeros(BT, C, H, W).view(-1, T, C, H, W)
        B, T, C, H, W = alg_MV_fea.shape
        feat_ = feat.view(-1, T, C, H, W)
        for i in range(T):
            if i != 0:
                alg_MV_fea[:,i,...] = flow_warp(feat_[:,i-1,...], mv[:,i-1,...].cuda())
            else:
                alg_MV_fea[:,i,...] = feat_[:,i,...].clone()
        alg_MV_fea = alg_MV_fea.view(BT, C, H, W)

        #### Temporal features: [alg_MV_fea], [pred_feat], [feat]
        feat = feat.contiguous().view(B, T, C, H, W)   
        # alg_MV_fea = feat.contiguous().view(B, T, C, H, W) 
        alg_MV_fea = alg_MV_fea.contiguous().view(B, T, C, H, W)   
        pred_feat = pred_feat.contiguous().view(B, T, C, H, W) 

        ### temporal aggregation 
        feat_f = self.aggblock(feat, alg_MV_fea, pred_feat)

        # compute offset and mask
        off_msk = self.offset_mask(feat_f)
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(off_msk[:, in_nc*2*n_off_msk:, ...])
        # perform deformable convolutional fusion
        fused_feat = F.relu(self.deform_conv(lqs, off, msk),  inplace=True)

        return fused_feat



class ChannelAtt_shift_bi(nn.Module):
    def __init__(self, channel, channel_, kernel_size, reduction, move_pixel=0, move_channel=0, direction='H'):
        super(ChannelAtt_shift_bi, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())
        self.conv = nn.Conv2d(channel, channel_, kernel_size=3, padding=(3-1)//2, bias=False)
        self.m_p = move_pixel
        self.m_c = move_channel

    def shift_bi_features(self, input, move_pixel, move_channel=0, direction='H'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == 'H':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
            zero_right[:, :, move_pixel:, :] = input[:, mid_channel:mid_channel + move_channel, :H - move_pixel,:]  # down

        elif direction == 'W':
            zero_left[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel:mid_channel, :, move_pixel:]  # left
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :, :W - move_pixel]  # right

        else:
            raise NotImplementedError("Direction should be 'H' or 'W'.")
        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def forward(self, x):
        x1 = self.shift_bi_features(x, self.m_p, self.m_c, 'H')
        y = self.avg_pool(x1)
        y = self.conv_du(y)
        out = x * y
        out = self.conv(out)
        return out


class ContextBlock(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


### --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        self.act = nn.LeakyReLU(0.2)
        self.channel1 = n_feat // 2
        self.channel2 = n_feat-self.channel1       
        self.gcnet = ContextBlock(self.channel1, bias=bias)
        self.tail = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)

    def forward(self, x):
        # res = self.cal(x)
        x1, x2 = torch.split(x,[self.channel1,self.channel2],dim=1)
        res1 = self.act(self.gcnet(x1))
        com1 = res1 + x2
        res2 = self.act(self.gcnet(com1))
        com2 = res2 + com1
        res = self.tail(torch.cat((com1,com2),dim=1))
        return res


class DualTemporalFE(nn.Module):
    def __init__(self,  in_nc, out_nc):
        super(DualTemporalFE, self).__init__()
        self.center_frame_idx = 7 //2
        for i in range(1, 3):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(in_nc, in_nc, 3, stride=2, padding=3//2),
                    nn.ReLU(inplace=True),
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*in_nc, in_nc, 3, padding=3//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_nc, in_nc, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(in_nc, in_nc, 3, stride=2, padding=3//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_nc, in_nc, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        
        self.feat_fusion =  nn.Sequential(nn.Conv2d(7 * in_nc, in_nc, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),)
    def forward(self, feats, aligned_feats, pred_feats):
        b, t, c, h, w = aligned_feats.size()
        # temporal attention
        embedding_ref = feats[:, self.center_frame_idx, :, :, :].clone()
        embedding_ref = embedding_ref.repeat(7,1,1,1)
        embedding1 = aligned_feats.view(-1, c, h, w)
        embedding2 = pred_feats.view(-1, c, h, w)

        corr1 = embedding1 * embedding_ref    # (b, h, w)
        corr2 = embedding2 * embedding_ref   # (b, h, w)
        corr_prob1 = torch.sigmoid(corr1)  # (b, t, h, w).unsqueeze(1)
        corr_prob1 = corr_prob1.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)

        corr_prob2 = torch.sigmoid(corr2)  # (b, t, h, w)  .unsqueeze(1)  .unsqueeze(1)
        corr_prob2 = corr_prob2.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feats = aligned_feats.view(b, -1, h, w) * corr_prob1 
        aligned_feats2 = pred_feats.view(b, -1, h, w) * corr_prob2
        feats = feats.view(b, -1, h, w)
        # feature extraction (with downsampling)
        out_lst = [0.1 * self.feat_fusion(aligned_feats + aligned_feats2) + self.feat_fusion(feats)] 
        for i in range(1, 3):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(3 - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(torch.cat([out, out_lst[i]], 1) )

        TA_feat = out 

        return TA_feat


# Residual channel attention block (RCB) 
class RCB_shift(nn.Module):
    def __init__(self, nChannels, nDenselayer=2, growthRate=32):
        super(RCB_shift, self).__init__()
        # nChannels_ = nChannels
        modules = []
        direction_list = ['H','W']
        for i in range(nDenselayer):    
            modules.append(make_shift(nChannels, growthRate, direction_list[i]))
            # nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_3x3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        # out = self.attention(out)
        out = 0.2 * self.conv_3x3(out)
        out = out + x
        return out



class make_shift(nn.Module):
    def __init__(self, nChannels, growthRate, direction, kernel_size=3):
        super(make_shift, self).__init__()
        self.conv = ChannelAtt_shift_bi(nChannels, nChannels, kernel_size=kernel_size, reduction=8, move_pixel=2, move_channel=4, direction=direction)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv(x))
        out  = out + x
        # out = torch.cat([x, out], 1)
        return out



class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=64, nb=3, out_nc=1, base_ks=3):
        super(PlainCNN, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        hid_conv_lst = []
        for _ in range(nb):
            hid_conv_lst += [ 
                RCB_shift(nf)
                ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out) 
        out = self.out_conv(out) 
        return out



class CPGA(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # align & aggregate
        self.TempInterFrame = TIFM()
        self.SpaIntraFrame = SIFM(channels)
        self.qenet = PlainCNN(nb=2)

    def forward(self, lqs, mvs, preds, ress):
        n, cT, h, w =  lqs.shape
        center_frm = lqs[:, cT//2, ...].unsqueeze(1)
        preds = preds.view(-1, 1, h, w)
        # ress = ress[:,cT//2, ...].unsqueeze(1)
        # n, T, c, h, w = lqs.shape
        mvs = mvs.contiguous().view(n, -1, h, w, 2)
        temp_feat  = self.TempInterFrame(lqs, preds, mvs)
        spa_feat = self.SpaIntraFrame(temp_feat, ress)
        out = self.qenet(spa_feat)
        out = out + center_frm

        return out



if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = CPGA().cuda()
    from thop import profile
    with torch.no_grad():

        input = torch.randn(1, 7*6, 416, 240).cuda()
        flops, params = profile(net, inputs=(input, ))
        total = sum([param.nelement() for param in net.parameters()])
        print('   Number of params: %.2fM' % (total / 1e6))
        print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))