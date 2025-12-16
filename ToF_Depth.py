import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import prms_init

height = 180
Height = 176
width = 240
para20 = [-0.7373, 0.2128, -0.015, 0.6600, 0.1972, 0.1486, -1.4819e-04, -0.0026]
args, _ = prms_init.prms_config()
is_synthetic = args.is_synthetic
if args.is_synthetic:       # synthetic
    fx = 210.0
    fy = 210.0
    cx = 120.0
    cy = 90.0
    para100 = [-0.7427, -0.016, 0, 0, 0, 0, 0.000154, -0.00018]
else:       # real
    fx = 190.6186
    fy = 200.7526
    cx = 116.8406
    cy = 101.4767
    para100 = [-0.7427, -0.016, -0.006, 0.0012, -0.0007, 0.0002, 0.000154, -0.00018]


def CreateR():
    r_array = torch.zeros([height, width], dtype=torch.float).cuda()
    for j in range(height):
        for i in range(width):
            r_array[j, i] = torch.sqrt((torch.tensor(i, dtype=torch.float) - cx) ** 2
                                       + (torch.tensor(j, dtype=torch.float) - cy) ** 2)
    return r_array


def CreateR_train():
    r_array = torch.zeros([Height, width], dtype=torch.float).cuda()
    for j in range(Height):
        for i in range(width):
            r_array[j, i] = torch.sqrt((torch.tensor(i, dtype=torch.float) - cx) ** 2
                                       + (torch.tensor(j, dtype=torch.float) - cy) ** 2)
    return r_array

R_array = CreateR()
R_array_train = CreateR_train()

class DepthAcquisitoin(nn.Module):
    def __init__(self, gradient_L1=True):
        super(DepthAcquisitoin, self).__init__()
        self.gradient = DepthGradient(gradient_L1)
        # self.DepthOut = DepthOut()
        self.T = torch.tensor(51)
        self.k20 = 2 * math.pi / 15
        self.k100 = 2 * math.pi / 3
        self.MA = 1
        self.MB = 5
        self.K0 = 1
        self.pi = math.pi
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # para20 = para20
        self.r_array = R_array
        self.R_array = R_array_train
        self.para20 = torch.tensor([[float(i)] for i in para20], dtype=torch.float32).cuda()
        # para100 = tof_config['para100']
        self.para100 = torch.tensor([float(i) for i in para100], dtype=torch.float32).cuda()

    def ComputeTemperature(self, raw):
        tint1 = raw[181, 160]
        tint2 = raw[181, 163]
        tand1 = tint1 & 3
        tand2 = tint2 & 255

        return (tand1 * pow(16, 2) + tand2 - 296) * 0.1852 + 25

    def ComputeDepth20(self, raw):
        I1 = (raw[:, 3, ...] - raw[:, 1, ...]).squeeze()
        Q1 = (raw[:, 0, ...] - raw[:, 2, ...]).squeeze()

        dis = 7.5 * torch.atan2(I1, Q1) / (2 * math.pi)
        dis[dis < 0] += 7.5
        depth = -0.0625 * dis ** 2 + 0.9937 * dis - 0.01428
        dis[dis < 0] = 0

        return depth

    def ComputeDepth100(self, raw):

        I1 = (raw[:, 7, ...] - raw[:, 5, ...]).squeeze()
        Q1 = (raw[:, 4, ...] - raw[:, 6, ...]).squeeze()
        dis = 1.5 * torch.atan2(I1, Q1) / (2 * math.pi)
        dis[dis < 0] += 1.5
        T_Error = self.para100[7] * self.T
        depth = dis + self.para100[0] + self.para100[1] * dis + self.para100[2] * torch.cos(4 * self.k100 * dis)\
            + self.para100[3] * torch.sin(4 * self.k100 * dis) + self.para100[4] * torch.cos(8 * self.k100 * dis)\
            + self.para100[5] * torch.sin(8 * self.k100 * dis) + T_Error + self.para100[6] * self.r_array

        return depth

    def Unwrapping(self, raw):
        depth20 = self.ComputeDepth20(raw)
        depth100 = self.ComputeDepth100(raw)
        # depth = torch.zeros([self.height, self.width], dtype=torch.float).cuda()
        pA = depth20 / 7.5
        pB = depth100 / 1.5
        e = pA * self.MB - pB * self.MA
        nB = (self.K0 * torch.round(e)) % self.MB
        dep_mean = depth100 + 1.5 * nB * (self.para100[1] + 1)
        dep_mean *= 1000

        return dep_mean

    def LabelUnwrapping(self, phi, depth_label, isFlatten):
        depth20 = depth_label / 1000
        dis = 1.5 * phi / (2 * math.pi)
        dis[dis < 0] += 1.5
        T_Error = self.para100[7] * self.T
        depth100 = dis + self.para100[0] + self.para100[1] * dis + self.para100[2] * torch.cos(4 * self.k100 * dis) \
                   + self.para100[3] * torch.sin(4 * self.k100 * dis) + self.para100[4] * torch.cos(8 * self.k100 * dis) \
                   + self.para100[5] * torch.sin(8 * self.k100 * dis) + T_Error + self.para100[6] * self.R_array

        pA = depth20 / 7.5
        pB = depth100 / 1.5
        e = pA * self.MB - pB * self.MA
        nB = (self.K0 * torch.round(e)) % self.MB
        dep_mean = depth100 + 1.5 * nB * (self.para100[1] + 1)

        if isFlatten:
            H, W = torch.meshgrid(torch.arange(Height).cuda(), torch.arange(width).cuda())
            depth = (2 * dep_mean - 0.0007).true_divide(torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                    + torch.sqrt((0.007 / dep_mean - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))
            depth[depth_label == 0] = 0
            depth *= 1000
        else:
            dep_mean[depth_label == 0] = 0
            depth = dep_mean * 1000

        return depth

    def PhaseUnwrapping(self, phi, img1):
        depth20 = self.ComputeDepth20(img1)
        dis = 1.5 * phi / (2 * math.pi)
        dis[dis < 0] = dis[dis < 0] + 1.5
        T_Error = self.para100[7] * self.T
        depth100 = dis + self.para100[0] + self.para100[1] * dis + self.para100[2] * torch.cos(4 * self.k100 * dis)\
            + self.para100[3] * torch.sin(4 * self.k100 * dis) + self.para100[4] * torch.cos(8 * self.k100 * dis)\
            + self.para100[5] * torch.sin(8 * self.k100 * dis) + T_Error + self.para100[6] * self.r_array
        pA = depth20 / 7.5
        pB = depth100 / 1.5
        e = pA * self.MB - pB * self.MA
        nB = (self.K0 * torch.round(e)) % self.MB
        dep_mean = depth100 + 1.5 * nB * (self.para100[1] + 1)

        H, W = torch.meshgrid(torch.arange(height).cuda(), torch.arange(width).cuda())
        depth = (2 * dep_mean - 0.0007).true_divide(torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                                           + torch.sqrt(
                    (0.007 / dep_mean - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))

        return depth * 1000

    def DepthLabelUnwrapping_test(self, phi, depth_label, isFlatten=False):
        depth20 = depth_label
        if len(phi.shape) == 3:
            phi = torch.atan2(phi[..., 1], phi[..., 0]).squeeze()
        dis = 1.5 * phi / (2 * math.pi)
        dis[dis < 0] = dis[dis < 0] + 1.5
        T_Error = self.para100[7] * self.T
        depth100 = dis + self.para100[0] + self.para100[1] * dis + self.para100[2] * torch.cos(4 * self.k100 * dis) \
                   + self.para100[3] * torch.sin(4 * self.k100 * dis) + self.para100[4] * torch.cos(8 * self.k100 * dis) \
                   + self.para100[5] * torch.sin(8 * self.k100 * dis) + self.para100[6] * self.r_array + T_Error
        pA = depth20 / 7.5
        pB = depth100 / 1.5
        e = pA * self.MB - pB * self.MA
        nB = (self.K0 * torch.round(e)) % self.MB
        dep_mean = depth100 + 1.5 * nB * (self.para100[1] + 1)

        if isFlatten:
            # 曲面拉平
            H, W = torch.meshgrid(torch.arange(height).cuda(), torch.arange(width).cuda())
            depth = (2 * dep_mean - 0.0007).true_divide(torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                                                        + torch.sqrt(
                (0.007 / (dep_mean + 1e-6) - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))
            depth[depth_label == 0] = 0
            depth *= 1000
        else:
            dep_mean[depth_label == 0] = 0
            depth = dep_mean * 1000

        return depth

    def flatten(self, dep_mean):
        H, W = torch.meshgrid(torch.arange(Height).cuda(), torch.arange(width).cuda())
        depth = ((2 * dep_mean - 0.0007).true_divide(torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                + torch.sqrt((0.007 / (dep_mean + 1e-6) - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)))
        depth[dep_mean == 0] = 0
        depth *= 1000

        return depth


    def OutUnwrapping(self, out):
        phi = torch.atan2(out[..., 1], out[..., 0]).squeeze()
        confidence = (out[..., 1] + out[..., 0]).squeeze()
        dis = 1.5 * phi / (2 * math.pi)
        dis[dis < 0] = dis[dis < 0] + 1.5
        T_Error = self.para100[7] * self.T
        depth100 = dis + self.para100[0] + self.para100[1] * dis + self.para100[2] * torch.cos(4 * self.k100 * dis) \
                   + self.para100[3] * torch.sin(4 * self.k100 * dis) + self.para100[4] * torch.cos(8 * self.k100 * dis) \
                   + self.para100[5] * torch.sin(8 * self.k100 * dis) + self.para100[6] * self.r_array + T_Error
        dep_mean = depth100

        H, W = torch.meshgrid(torch.arange(height).cuda(), torch.arange(width).cuda())
        depth = ((2 * dep_mean - 0.0007) / (torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                                            + torch.sqrt(
                    (0.007 / (dep_mean + 1e-6) - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))) * 1000

        return depth

    def gradientX(self, img, confidence):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])

        lc = F.pad(confidence, [1, 0, 0, 0])
        rc = F.pad(confidence, [0, 1, 0, 0])
        x_loss = torch.abs((l - r)[..., 0:w, 0:h]) * torch.exp(-torch.abs((lc - rc)[..., 0:w, 0:h]))

        return x_loss

    def gradientY(self, img, confidence):
        w, h = img.size(-2), img.size(-1)
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])

        uc = F.pad(confidence, [0, 0, 1, 0])
        dc = F.pad(confidence, [0, 0, 0, 1])
        y_loss = torch.abs((u - d)[..., 0:w, 0:h]) * torch.exp(-torch.abs((uc - dc)[..., 0:w, 0:h]))

        return y_loss


class DepthGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(DepthGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])

        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2))

def compute_mask(img1, img2, depth_label, depth_origin_tmp, is_synthetic, tof_config):
    maxc = 800
    fx = tof_config['fx']
    fy = tof_config['fy']
    cx = tof_config['cx']
    cy = tof_config['cy']
    batch, _, height, width = img1.shape
    W, H = torch.tensor(np.meshgrid(range(width), range(height))).cuda()
    mask_raw3 = torch.ones([batch, height, width], dtype=torch.float).cuda()
    mask = torch.ones([batch, height, width], dtype=torch.float).cuda()

    confidence_train = torch.abs(img1[:, 3, ...] - img1[:, 1, ...]) + torch.abs(img1[:, 0, ...] - img1[:, 2, ...])
    confidence_label = torch.abs(img2[:, 3, ...] - img2[:, 1, ...]) + torch.abs(img2[:, 0, ...] - img2[:, 2, ...])
    confidence_ratio = confidence_train/(confidence_label + 1e-6)
    mask_raw3[confidence_ratio > 1] = 0

    #   confidence mask
    intrix = ((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1
    confidence_aug = (confidence_label * intrix) ** 2 * depth_label / 1000
    confidence_orig = ((confidence_train * intrix) ** 2 * depth_origin_tmp / 1000).unsqueeze(1)

    weights = torch.ones([batch, height, width], dtype=torch.float).cuda()
    label_mask = depth_label.detach().clone()

    if is_synthetic:
        mask_raw3[label_mask == 0] = 0
        mask_valid = mask_raw3 * mask
        mask3 = mask_raw3 * mask
    else:
        mask[confidence_aug < 80] = 0      # 80
        label_mask[(label_mask > 1788)] = 0     # 1788
        label_mask[(label_mask < 180)] = 0     # 180
        mask_raw3[label_mask == 0] = 0
        weights[label_mask > maxc] = label_mask[label_mask > maxc] * (
                1 - (label_mask[label_mask > maxc] / maxc) ** 2 / 5) ** 2 / 512
        mask_valid = mask_raw3 * mask
        mask3 = mask_raw3 * weights * mask

    mask4 = mask3.unsqueeze(dim=1)

    return mask_valid, mask3, mask4, weights, confidence_orig


#################### complex-valued raw loss and amplitude loss
def CosineLoss(out, img2, mask3, mask_valid, depth_label, tof_config):
    height = tof_config['height']
    width = tof_config['width']
    Batch = out.shape[0]
    label_imag = (img2[:, 3, ...] - img2[:, 1, ...])  # rho*sin(phi)
    label_real = (img2[:, 0, ...] - img2[:, 2, ...])  # rho*cos(phi)
    label_mask = depth_label * mask3

    out_real = out[..., 0].squeeze()
    out_imag = out[..., 1].squeeze()

    rho_out = torch.sqrt(out_real ** 2 + out_imag ** 2) / 2
    rho_label = torch.sqrt(label_real ** 2 + label_imag ** 2) / 2

    PSum = torch.zeros([Batch, height, width], dtype=torch.float).cuda()
    PSum[label_mask > 0] = 1
    M = torch.sum(PSum) + 1

    complex_loss = torch.sum((torch.abs(out_real - label_real) + torch.abs(out_imag - label_imag)) * mask3) / M  # 复数差的模
    rho_loss = torch.sum(torch.abs(rho_out - rho_label) * mask3) / M

    return rho_loss, complex_loss, rho_out, M