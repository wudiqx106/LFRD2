import DCUNet.complex_nn as complex_nn
from .common import *
from .ImplicitNeural import *


def mse_loss_func(pred, gt, mask):
    return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])


########################################################################################################################
class cplx_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="zeros", isBN=True):
        super(cplx_conv, self).__init__()
        self.conv = complex_nn.ComplexConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                             padding=padding, padding_mode=padding_mode)
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.bn = complex_nn.ComplexBatchNorm2d(out_channel)
        self.isBN = isBN

    def forward(self, x):
        x = self.conv(x)
        if self.isBN:
            x = self.bn(x)
        x = self.activation(x)

        return x


#   bilateral learning
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, use_bias=True, activation=
                nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), kernel_size, padding=padding, stride=stride,
                              bias=use_bias)
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=False) if activation else None
        self.bn = nn.BatchNorm2d(out_channel) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.LeakyReLU(negative_slope=0.05, inplace=False), batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


########################################################################################################################
class MSDEB(nn.Module):
    """
    Following channel split, there's convolution and activation
    """
    def __init__(self, in_ch=32, G0=32, scales=4):
        super(MSDEB, self).__init__()
        kSize = 3
        self.conv_b1 = ConvBlock(in_channel=in_ch, out_channel=G0, kernel_size=kSize, batch_norm=True)
        self.conv_b2 = ConvBlock(in_channel=in_ch, out_channel=G0, kernel_size=kSize, batch_norm=True)
        self.conv_b3 = ConvBlock(in_channel=in_ch, out_channel=G0, kernel_size=kSize, batch_norm=True)
        self.G0 = G0
        self.scales = scales

        self.conv1 = nn.Conv2d(in_ch, G0, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(G0, G0, 1, stride=1, padding=0)
        self.conv_3_1 = nn.Conv2d(G0 // scales, G0 // scales, 3, stride=1, padding=(kSize - 1) // 2)
        self.conv_3_2 = nn.Conv2d(G0 // scales, G0 // scales, 5, stride=1, padding=(5 - 1) // 2)
        self.conv_3_3 = nn.Conv2d(G0 // scales, G0 // scales, 7, stride=1, padding=(7 - 1) // 2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.05, inplace=False)

    def forward(self, x):
        Low = self.conv1(x)
        X = torch.split(Low, int(self.G0/self.scales), dim=1)
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]

        y1 = x1
        y2 = self.lrelu(self.conv_3_1(x2))
        y3 = self.lrelu(self.conv_3_2(x3))
        y4 = self.lrelu(self.conv_3_3(x4))

        out = torch.cat([y1, y2, y3, y4], 1)
        out = self.conv2(out)

        return out


####################################################################################
def conv3x3_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros"):
    layer = nn.Sequential(
        complex_nn.ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 padding_mode=padding_mode),
        complex_nn.ComplexBatchNorm2d(out_channels)
    )
    return layer


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.branch1 = conv3x3_bn(in_channel, out_channel, stride=stride)
        self.branch2 = conv3x3_bn(out_channel, out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.branch1(x)
        out = F.relu(out, True)
        out = self.branch2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)


class Basic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Basic, self).__init__()
        # self.channel_att = channel_att
        # self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
            complex_nn.ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                     padding_mode='zeros'),
            complex_nn.ComplexBatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # complex_nn.ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            #                          padding_mode='zeros'),
            # # complex_nn.ComplexBatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


    def forward(self, x):
        x = self.conv1(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(UNet, self).__init__()
        conv = complex_nn.ComplexConv2d
        tconv = complex_nn.ComplexConvTranspose2d
        # bn = complex_nn.ComplexBatchNorm2d

        self.conv1 = conv(in_channel=in_channel, out_channel=8, kernel_size=3, stride=1, padding=1)
        self.conv1_ = conv(in_channel=8, out_channel=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = Basic(16, 32)
        self.conv3 = conv(in_channel=32, out_channel=32, kernel_size=5, stride=2, padding=2)
        self.conv4 = Basic(32, 64)
        self.conv5 = conv(in_channel=64, out_channel=64, kernel_size=5, stride=2, padding=2)
        self.conv5_ = conv(in_channel=64, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = residual_block(128, 128)
        self.conv7 = residual_block(128, 128)
        # 卷积上采样
        self.conv8 = tconv(in_channel=128, out_channel=64, kernel_size=4, stride=2, padding=1)
        self.conv9 = Basic(64+64, 64)
        self.conv10 = tconv(in_channel=64, out_channel=32, kernel_size=4, stride=2, padding=1)
        self.conv11 = Basic(32+32, 32)
        self.conv12 = tconv(in_channel=32, out_channel=16, kernel_size=4, stride=2, padding=1)
        self.conv13 = conv(in_channel=16+in_channel, out_channel=4, kernel_size=3, stride=1, padding=1)
        self.conv14 = conv(in_channel=4, out_channel=out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_ = self.conv1_(conv1)
        conv2 = self.conv2(conv1_)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv5_ = self.conv5_(conv5)
        conv6 = self.conv6(conv5_)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(torch.cat([conv8, conv4], dim=1))
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(torch.cat([conv10, conv2], dim=1))
        conv12 = self.conv12(conv11)
        conv13 = self.conv13(torch.cat([conv12, x], dim=1))
        conv14 = self.conv14(conv13)

        return conv14


# Depth refinement
class Format(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Format, self).__init__()
        self.MSDEB = MSDEB
        # self.MSDEB = MSDBB

        self.conv1 = self.MSDEB(in_ch=in_channel, G0=32)
        self.conv2 = self.MSDEB(in_ch=32, G0=32)
        self.conv3 = self.MSDEB(in_ch=32, G0=32)
        # 卷积上采样
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv5 = self.MSDEB(in_ch=32 + 32, G0=32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv7 = self.MSDEB(in_ch=32 + 32, G0=32)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(torch.cat([conv2, conv4], dim=1))
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(torch.cat([conv6, conv1], dim=1))
        conv8 = self.conv8(conv7)

        return conv8


class ImpInt(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(ImpInt, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time    # number of propagation  18
        self.affinity = self.args.affinity      # affinity type (dynamic pos-neg, dynamic pos, 'static pos-neg, static pos, none'

        self.ch_g = ch_g    # 8
        self.ch_f = ch_f    # 1
        self.k_g = k_g      # 3
        self.k_f = k_f      # 3
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1  # 8
        self.idx_ref = self.num // 2        # 4
        self.dt = 1

        self.INP = []
        for i in range(args.prop_time):
            self.INP.append(SVI())
        self.INP = nn.ModuleList(self.INP)

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        # TODO : Need more efficient way
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                if self.args.legacy:
                    offset_tmp[:, 0, :, :] = \
                        offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1, :, :] = \
                        offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

                conf_tmp = ModulatedDeformConv2dFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups)
                # conf_tmp = self.conf_tmp()
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        # aff_ref = 1.0 - aff_sum
        aff_ref = - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConv2dFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups)

        return feat

    def CC(self, n, alpha):
        return (n + 1) ** (1 - alpha) - n ** (1 - alpha)

    def Coff(self, alpha):
        coff = torch.exp(torch.lgamma(2 - alpha))

        return coff

    def forward(self, feat_init, guidance, confidence=None, order=None, samfeat=None, source=None, amp=None):
        '''
        feat_fix: upsampled LR depth  --  input
        source: LR depth
        '''
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == feat_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence)
        else:
            offset, aff = self._get_offset_affinity(guidance, None)

        feat_result = feat_init

        list_feat = [feat_result]
        coff = self.Coff(order)

        for k in range(1, self.prop_time + 1):
            Depth_grad, samfeat = self.INP[k - 1](list_feat[-1], samfeat, offset, aff, amp)

            if k == 1:
                diff = source - list_feat[-1]
                feat_result = coff * (Depth_grad + diff * 0.01) + list_feat[-1]
            else:
                diff = source - list_feat[-1]
                feat_result = coff * (Depth_grad + diff * 0.01) + list_feat[-1]
                for i in range(1, k):
                    if torch.isnan(self.CC(k - i, order)).any():
                        print('\nThere is Nan in {}th iteration\n'.format(i))
                    if torch.isnan(list_feat[i]).any():
                        print('\nThere is Nan in list_feat[{}]\n'.format(i))
                    feat_result -= self.CC(i, order) * (list_feat[k-i] - list_feat[k-i-1])

            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class FracDiff(nn.Module):
    def __init__(self, args, n_feat=32):
        super(FracDiff, self).__init__()

        self.args = args

        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # self.depth_refine = DepthRefine()

        # Guidance Branch +int(n_feat/2)
        # 1/1
        self.gd_dec2 = conv_bn_relu(1, 32, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec1 = conv_bn_relu(32, 32, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(32, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            self.cf_dec2 = conv_bn_relu(1, 32, kernel=3, stride=1, bn=False,
                                        padding=1)
            self.cf_dec1 = conv_bn_relu(32, 32, kernel=3, stride=1, bn=False,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.Sigmoid()
            )

        self.order_dec2 = conv_bn_relu(1, 32, kernel=3, stride=1, bn=False,
                                    padding=1)
        self.order_dec1 = conv_bn_relu(32, 32, kernel=3, stride=1, bn=False,
                                    padding=1)
        self.order_dec0 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        )

        self.iter_layer = ImpInt(args, self.num_neighbors, 1, 3,
                                self.args.prop_kernel)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, depth, amp, confidence_aug):
        # Init Depth Decoding
        # depth, samfeats = self.depth_refine(pred)

        # Guidance Decoding
        gd_fd1 = self.gd_dec2(depth)
        gd_fd1 = self.gd_dec1(gd_fd1)
        guide = self.gd_dec0(gd_fd1)

        cf_fd1 = self.cf_dec2(confidence_aug)
        cf_fd2 = self.cf_dec1(cf_fd1)
        confidence = self.cf_dec0(cf_fd2)

        order_fd1 = self.order_dec2(depth)
        order_fd2 = self.order_dec1(order_fd1)
        order = self.order_dec0(order_fd2) * 0.8 + 0.1

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.iter_layer(depth, guide, confidence, order, gd_fd1, depth, amp)

        # Remove negative depth
        # y = torch.clamp(y, min=0)

        results = [y] + [depth]

        return {'y_pred': results, 'list_feat': y_inter}

    def get_loss(self, output, sample):
        y_pred = output['y_pred']

        y, mask_hr, grad_gt = (sample[k] for k in ('hr', 'mask_hr', 'grad'))
        max = sample['max'].unsqueeze(-1).unsqueeze(-1)
        min = sample['min'].unsqueeze(-1).unsqueeze(-1)
        prediction = y_pred[0] * (max - min) + min
        gt = y * (max - min) + min

        l1_loss = l1_loss_func(prediction, gt, mask_hr)
        mse_loss = mse_loss_func(prediction, gt, mask_hr)

        ####   compute loss
        loss = 0
        for j in range(len(y_pred)):
            if j == 0:
                loss = l1_loss_func(y_pred[j], y, mask_hr)
            else:
                loss += l1_loss_func(y_pred[j], y, mask_hr) * 0.1

        return loss, {
            'l1_loss': l1_loss.detach().item(),
            'mse_loss': mse_loss.detach().item(),
            'optimization_loss': l1_loss.detach().item(),
        }

