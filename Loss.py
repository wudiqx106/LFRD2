import torch
from torch import nn
import torch.nn.functional as F


class DepthGrad(nn.Module):
    """
    depth gradient loss function.
    """
    def __init__(self, gradient_L1=True):
        super(DepthGrad, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)
        # self.depth_loss = depth_loss()
        # self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, ground_truth):
        # return self.l1_loss(pred, ground_truth) + \
        #         self.l1_loss(self.gradient(pred), self.gradient(ground_truth))
        return torch.abs(self.gradient(pred) - self.gradient(ground_truth))

class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        # confidence = torch.sqrt(torch.abs(img[3] - img[1]) + torch.abs(img[0] - img[2]))
        l = F.pad(img, [1, 0, 0, 0])    # left
        r = F.pad(img, [0, 1, 0, 0])    # right
        u = F.pad(img, [0, 0, 1, 0])    # up
        d = F.pad(img, [0, 0, 0, 1])    # down
        if self.L1:
            # return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h]) + confidence
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )
