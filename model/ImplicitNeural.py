import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
# import models
# from models import register
# from utils import make_coord
# from torchvision.ops import DeformConv2d
# from torchvision.ops import deform_conv
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dFunction


class INT(torch.nn.Module):
    def __init__(self, in_chs=54, out_chs=18, num_filter=32, kernel_size=3, stride=1, padding=1, bias=True):
        super(INT, self).__init__()
        self.num_filter = num_filter
        # self.out_chs = out_chs
        self.conv1 = torch.nn.Conv2d(in_chs, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv3 = torch.nn.Conv2d(num_filter, out_chs, kernel_size, stride, padding, bias=False)

        self.act = torch.nn.GELU()

    def forward(self, x):
        residual = x

        feat = self.conv1(x)
        feat = self.act(feat)
        feat = self.conv2(feat)

        feat = torch.add(feat, residual[:, 0:self.num_filter, ...])
        out = self.conv3(feat)
        return out, feat


class SVI(nn.Module):
    '''
    SVI: Spatial-Varying Integration
    LIIF-based implicit neural network for image integration
    '''
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()

        # mlp
        self.num = 8
        self.ch_g = 32
        self.idx_ref = 4
        self.ch_f = 1
        self.k_f = 3
        self.int = INT(in_chs=51)  # depth-1  amp-1  feat-32  offset-9
        # self.dcnv = DeformConv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_offset_aff = nn.Conv2d(in_channels=self.ch_g, out_channels=3 * self.num, kernel_size=3, stride=1,padding=1, bias=True)  # 输入ch=8  输出ch=24
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()  # 卷积参数初始化

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def deform_conv(self, feat, offset, aff):
        feat = ModulatedDeformConv2dFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups)

        return feat

    def get_subpixel_values(self, image, offset):
        batch_size, channels, height, width = image.size()

        # Reshape offset to separate the 9 subpixel offsets (2 for each: x and y)
        offset = offset.view(batch_size, 9, 2, height, width)
        offset_x = offset[:, :, 0, :, :]  # shape [batch, 9, height, width]
        offset_y = offset[:, :, 1, :, :]  # shape [batch, 9, height, width]

        # Create mesh grid for original pixel coordinates
        # base_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        base_grid = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'),
                                   indexing='ij')
        base_grid = torch.stack(base_grid, dim=0).float()  # shape [2, height, width]
        base_grid = base_grid.unsqueeze(0).unsqueeze(0)  # shape [1, 1, 2, height, width]
        base_grid = base_grid.repeat(batch_size, 9, 1, 1, 1)  # shape [batch, 9, 2, height, width]

        # Calculate the subpixel coordinates by adding offsets
        subpixel_coords_x = base_grid[:, :, 0, :, :] + offset_x  # shape [batch, 9, height, width]
        subpixel_coords_y = base_grid[:, :, 1, :, :] + offset_y  # shape [batch, 9, height, width]

        # Normalize coordinates to be in the range [-1, 1] for grid_sample
        subpixel_coords_x = 2.0 * subpixel_coords_x / (width - 1) - 1.0
        subpixel_coords_y = 2.0 * subpixel_coords_y / (height - 1) - 1.0

        # Stack coordinates to create a grid for grid_sample
        subpixel_coords = torch.stack((subpixel_coords_x, subpixel_coords_y),
                                      dim=-1)  # shape [batch, 9, height, width, 2]
        subpixel_coords = subpixel_coords.view(batch_size * 9, height, width, 2)  # shape [batch*9, height, width, 2]

        # Reshape image for grid_sample
        image = image.unsqueeze(1).repeat(1, 9, 1, 1, 1)  # shape [batch, 9, channels, height, width]
        image = image.view(batch_size * 9, channels, height, width)  # shape [batch*9, channels, height, width]

        # Use grid_sample to get the subpixel values
        subpixel_values = F.grid_sample(image, subpixel_coords,
                                        align_corners=True)  # shape [batch*9, channels, height, width]

        # Reshape back to the original batch size
        subpixel_values = subpixel_values.view(batch_size, 9, height, width)  # shape [batch, 9, height, width]

        return subpixel_values

    def forward(self, inp, feat, offset, aff, amp):
        # feat = inp
        # B, C, H, W = inp.shape
        # print(f'inp.shape: {inp.shape}\nrgb.shape: {rgb.shape}\nfeat.shape: {feat.shape}\noffset.shape: {offset.shape}')
        guide_feat = torch.cat([feat, amp, offset], dim=1)
        coef, guidance = self.int(guide_feat)
        a1, a2 = torch.chunk(coef, 2, dim=1)
        # offset, aff = self._get_offset_affinity(feat)
        # ret = self.deform_conv(ret_tmp, offset, aff)
        inp_mul = self.get_subpixel_values(inp, offset)
        # print(f'a1.shape: {a1.shape}\na2.shape: {a2.shape}\naff.shape: {aff.shape}\ninp_mul.shape: {inp_mul.shape}')
        ret = torch.sum(aff * (a1 * inp_mul + a2), dim=1, keepdim=True)

        return ret, guidance
