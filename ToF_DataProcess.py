import math
import numpy as np
import scipy.signal as signal
from math import pi

T = 48.5

def CreateR(tof_config):
    height = tof_config['height']
    width = tof_config['width']
    cx = tof_config['cx']
    cy = tof_config['cy']
    r_array = np.zeros([height, width], dtype=float)
    for i in range(height):
        for j in range(width):
            r_array[i, j] = np.sqrt((j - cx)**2 + (i - cy)**2)

    return r_array


def ComputeDepth20(raw):
    I1 = (raw[:, 3, ...] - raw[:, 1, ...]).squeeze()
    Q1 = (raw[:, 0, ...] - raw[:, 2, ...]).squeeze()
    # I1 = signal.medfilt2d(I1, kernel_size=3)
    # Q1 = signal.medfilt2d(Q1, kernel_size=3)

    dis = 7.5 * np.arctan2(I1, Q1) / (2 * math.pi)
    dis[dis < 0] += 7.5
    depth = -0.0625 * dis ** 2 + 0.9937 * dis - 0.01428
    depth[depth < 0] = 0

    return depth


def ComputeDepth100(raw, tof_config):
    para100 = tof_config['para100']
    para100 = [float(i) for i in para100]

    r_array = CreateR(tof_config)
    k100 = 2 * math.pi / 3

    I1 = (raw[:, 7, ...] - raw[:, 5, ...]).squeeze()
    Q1 = (raw[:, 4, ...] - raw[:, 6, ...]).squeeze()

    dis = 1.5 * np.arctan2(I1, Q1) / (2 * math.pi)
    dis[dis < 0] += 1.5
    T = 38
    depth = dis + para100[0] + para100[1] * dis + para100[2] * np.cos(4 * k100 * dis) \
            + para100[3] * np.sin(4 * k100 * dis) + para100[4] * np.cos(8 * k100 * dis) \
            + para100[5] * np.sin(8 * k100 * dis) + para100[7] * T + para100[6] * r_array

    return depth


def Unwrapping(raw, tof_config, kernel_size=3, isFilter=False):
    para100 = tof_config['para100']
    para_100 = float(para100[1])
    depth20 = ComputeDepth20(raw)
    depth100 = ComputeDepth100(raw, tof_config)
    MA = 1;MB = 5;pA = depth20 / 7.5;pB = depth100 / 1.5;K0 = 1
    e = pA * MB - pB * MA
    nB = np.mod(K0 * np.around(e), MB)

    dep_mean = depth100 + 1.5 * nB * (para_100 + 1)

    if isFilter:
        dep_mean = signal.medfilt2d(dep_mean, kernel_size=kernel_size)

    return dep_mean


def SaveDepth(path, filename, depth, tof_config):
    height = tof_config['height']
    width = tof_config['width']
    fx = tof_config['fx']
    fy = tof_config['fy']
    cx = tof_config['cx']
    cy = tof_config['cy']

    X = np.zeros([height, width], dtype=np.float32)
    Y = np.zeros([height, width], dtype=np.float32)
    fid = open(path + filename + '.txt', 'w')
    for j in range(height):
        for i in range(width):
            X[j, i] = depth[j, i] * (i - cx) / fx
            Y[j, i] = depth[j, i] * (j - cy) / fy
            fid.write('{};{};{}\n'.format(X[j, i], Y[j, i], depth[j, i]))
    fid.close()


def SaveTXT(i, depth_eval, depth_label, depth_origin, path_UDC, path_normal, path_origin, tof_config):
    SaveDepth(path_UDC, '/depth_eval_{}'.format(i), depth_eval, tof_config)
    SaveDepth(path_normal, '/depth_label_{}'.format(i), depth_label, tof_config)
    SaveDepth(path_origin, '/depth_origin_{}'.format(i), depth_origin, tof_config)


def RadiationCorr(depth_tmp, tof_config):
    height = tof_config['height']
    width = tof_config['width']
    fx = tof_config['fx']
    fy = tof_config['fy']
    cx = tof_config['cx']
    cy = tof_config['cy']
    W, H = np.meshgrid(range(width), range(height))
    depth = (2 * depth_tmp - 0.0007) / (np.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1) +
            np.sqrt((0.007 / (depth_tmp + 1e-6) - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))

    depth[depth_tmp == 0] = 0
    depth = depth * 1000

    return depth

def get_mask(img2, depth_dn, depth_normal, tof_config, isSynthetic=False):
    height = tof_config['height']
    width = tof_config['width']
    fx = tof_config['fx']
    fy = tof_config['fy']
    cx = tof_config['cx']
    cy = tof_config['cy']

    img2 = img2.squeeze()
    I100 = img2[3, ...] - img2[1, ...]
    Q100 = img2[0, ...] - img2[2, ...]
    mask = np.ones([height, width], dtype=np.float)
    confidence_label = np.abs(I100) + np.abs(Q100)
    if not isSynthetic:
        W, H = np.meshgrid(range(width), range(height))
        intrix = ((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1
        confidence_aug = (confidence_label * intrix) ** 2 * depth_normal / 1000
        mask[confidence_aug < 80] = 0
    mask[depth_normal > 1788] = 0
    mask[depth_dn > 1788] = 0
    mask[depth_normal < 180] = 0
    mask[depth_dn < 180] = 0

    return mask, confidence_label
