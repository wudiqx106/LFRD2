from configobj import ConfigObj
from validate import Validator
import numpy as np
from torch.utils.data import Dataset
import ToF_Depth
import pickle
import torch
import os
# from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


####################dataset preprocess
DepthAcquire = ToF_Depth.DepthAcquisitoin()

#  data in one .pkl file, for small size dataset
class TOF_data(Dataset):    # real data
    def __init__(self, TrainData_path, LabelData_path, training=True):
        self.training = training
        f_t = open(TrainData_path, 'rb')    # 训练1
        self.data_t = pickle.load(f_t)
        # print(f'data_t:{len(self.data_t[5])}')
        f_t.close()
        self.train = torch.from_numpy(np.stack(self.data_t, axis=0)).float()

        f_l = open(LabelData_path, 'rb')    # 标签1(新数据) 不需要减去暗通道数据
        self.data_l = pickle.load(f_l)
        # print(f'data_l:{len(self.data_l[0])}')
        f_l.close()
        self.label = torch.from_numpy(np.stack(self.data_l, axis=0)).float()
        # self.label = self.label[0:8,...]

        self.length = len(self.data_t[0])
        print("dataset has been loaded into mem already....")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.training:
            return self.train[:, idx, 2:-2, :], self.label[:, idx, 2:-2, :]
        else:
            return self.train[:, idx,...], self.label[:, idx,...]      # eval

class ToF_data2(Dataset):   # syntheticdata
    def __init__(self, Data_path, training=True):
        # fill = (0, 0, 0, 0, 0)

        self.DataTransform = A.Compose([
            # transforms.ToPILImage(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-3, 3)),
            ToTensorV2(transpose_mask=False)
        ])

        self.training = training
        self.Datapath = Data_path
        self.raw_data_list = sorted(os.listdir(self.Datapath))

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, idx):
        raw_idx = self.raw_data_list[idx]    # acquire raw data according to index
        raw_path = os.path.join(self.Datapath, raw_idx)
        self.raw = np.fromfile(raw_path, dtype=np.float32).reshape([10, 180, 240])
        self.raw_training = self.raw.transpose(1, 2, 0)
        # print('raw:', self.raw.shape)

        if self.training:
            self.train_data = self.DataTransform(image=self.raw_training)['image']
            self.label = self.train_data[0:5, ...]
            self.train = self.train_data[5:10, ...]

            return self.train[:, 2:-2, :], self.label[:, 2:-2, :]
        else:
            self.train_data = torch.from_numpy(self.raw)
            self.label = self.train_data[0:5, ...]
            self.train = self.train_data[5:10, ...]     # torch.Size([5, 180, 240])

            return self.train, self.label
        

def transform(img1):
    '''
    test synthetic
    '''
    ImagPart = (img1[:, 3, ...] - img1[:, 1, ...]).unsqueeze(dim=1).unsqueeze(dim=-1)    # rho*sin(phi)
    RealPart = (img1[:, 0, ...] - img1[:, 2, ...]).unsqueeze(dim=1).unsqueeze(dim=-1)    # rho*cos(phi)

    return torch.cat([RealPart, ImagPart], dim=-1)


def TestTransform(raw):
    '''
    test real输入格式转换
    '''
    ImagPart = (raw[:, 7, ...] - raw[:, 5, ...]).unsqueeze(dim=0).unsqueeze(dim=-1)
    RealPart = (raw[:, 4, ...] - raw[:, 6, ...]).unsqueeze(dim=0).unsqueeze(dim=-1)  # rho*cos(phi)

    return torch.cat([RealPart, ImagPart], dim=-1)


def Label2Depth(out, depth_label, isFlatten=False):
    phi = torch.atan2(out[..., 1], out[..., 0]).squeeze()
    depth_out = DepthAcquire.LabelUnwrapping(phi, depth_label, isFlatten).unsqueeze(dim=1)

    return depth_out

