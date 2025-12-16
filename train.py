from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import Loss
import pickle
import ToF_Depth
import ToF_Dataset
import torch.backends.cudnn as cudnn

# from model import *
from model.cplx import *
import itertools
from logger import *


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(args, config):
    losses = AverageMeter()
    global loss, loss_g, loss_d
    is_synthetic = args.is_synthetic
    is_normalized = args.is_normalized


    if is_synthetic:
        print("Synthetic data training")

        # parameters
        train_config = config['training']
        train_epoch = train_config['synthetic_epoch']
        batch_size = train_config['batch_size']
        num_workers = train_config['num_workers']
        lr_rate1 = train_config['lr_rate1']
        # lr_rate2 = train_config['lr_rate2']
        lr_decaystep = train_config['lr_decaystep']
        lr_gamma = train_config['lr_gamma']
        drop_last = train_config['drop_last']
        tof_config = config['ToF_configs_train_synthetic']

        dataset_config = config['DataTrain_configs']
        Data_path = dataset_config['Data_path_synthetic']
        precheckpoint = dataset_config['precheckpoint_synthetic']
        checkpoint = dataset_config['checkpoint_synthetic']
        presample = dataset_config['presample']
        sample = dataset_config['sample']
        mGPU = dataset_config['mgpu']

        log_name = 'log_{}.txt'.format(time.strftime("%Y-%m-%d", time.localtime()))
        file = open(log_name, 'a')
        loss_log_name = 'loss_log_{}.txt'.format(time.strftime("%Y-%m-%d", time.localtime()))
        file_loss = open(loss_log_name, 'w')

        if not os.path.exists(precheckpoint):
            os.makedirs(precheckpoint)
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        if not os.path.exists(presample):
            os.makedirs(presample)
        if not os.path.exists(sample):
            os.makedirs(sample)

        backbone = UNet()
        net = FracDiff(args)
        if args.cuda:
            net = net.cuda()
            backbone = backbone.cuda()
        if args.load_ckt:
            checkpoint_train_backbone = dataset_config['checkpoint_train_backbone']
            backbone.load_state_dict(torch.load(checkpoint_train_backbone))

            # checkpoint_train_net = dataset_config['checkpoint_train_net']
            # net.load_state_dict(torch.load(checkpoint_train_net))
        if mGPU:
            net = nn.DataParallel(net)
            backbone = nn.DataParallel(backbone)

        train_set = ToF_Dataset.ToF_data2(Data_path)
        train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                num_workers=num_workers)
        print('Data Loading Complete')
        optim_all = torch.optim.Adam(itertools.chain(backbone.parameters(), net.parameters()), lr=lr_rate1)
        lr_step_all = lr_scheduler.StepLR(optim_all, step_size=lr_decaystep, gamma=lr_gamma)

        cudnn.benchmark = True
        depth_grad = Loss.DepthGrad()

        ######################train complex U-net using cascaded network
        start = 250
        net = net.train()
        backbone = backbone.train()
        for e in range(start, start+train_epoch):
            t1 = time.time()
            for img1, img2 in train_data:
                img1 = img1.cuda()  # img1: torch.Size([16, 6, 176, 240])
                img2 = img2.cuda()  # img1: torch.Size([16, 7, 176, 240])
                with torch.autograd.set_detect_anomaly(True):
                    if is_normalized:
                        depth_label = img2[:, 4, ...] * 3000
                    else:
                        depth_label = img2[:, 4, ...] * 1000    # torch.Size([16, 176, 240])

                    depth_origin_tmp = img1[:, 4, ...]

                    mask_valid, mask3, mask4, weights, confidence_aug = ToF_Depth.compute_mask(img1, img2, depth_label,
                                                                                               depth_origin_tmp, is_synthetic, tof_config)
                    input = ToF_Dataset.transform(img1)

                    out_dn_res = backbone(input)  # output complex values
                    out_dn = out_dn_res + input
                    if is_normalized:
                        out_dn = out_dn * 2 - 1
                    rho_loss, complex_loss, amplitude, M = ToF_Depth.CosineLoss(out_dn, img2, mask3, mask_valid,
                                                                                depth_label, tof_config)

                    depth = ToF_Dataset.Label2Depth(out_dn, depth_label)  # output depth

                    output = net(depth, amplitude.unsqueeze(1), confidence_aug)  # output complex values
                    pred = output['y_pred']
                    depth_dn = pred[0].squeeze()

                    label_mask = depth_label.detach()
                    for j in range(len(pred)):
                        if j == 0:
                            DepthLoss = torch.sum(torch.abs(pred[j].squeeze() - label_mask) * mask3) / M
                        else:
                            DepthLoss += torch.sum(torch.abs(pred[j].squeeze() - label_mask) * mask3) / M * 0.1

                    DepthGrad_loss = torch.sum(depth_grad(depth_dn, label_mask) * mask3) / M
                    if is_normalized:
                        total_loss = (DepthLoss + DepthGrad_loss) * 1e-2
                    else:
                        total_loss = (1e-1 * DepthLoss + 1e0 * DepthGrad_loss) * 1e0 + complex_loss * 1e-1

                optim_all.zero_grad()
                total_loss.backward()
                optim_all.step()
                losses.update(total_loss.item())

            lr_step_all.step()
            t2 = time.time()
            print('{} [*epoch]: {},lr_net: {}, loss: {:.3f}, time: {:.3f}'
                  .format(time.strftime("%H:%M:%S", time.localtime()), e + 1, lr_step_all.get_last_lr(), losses.avg,
                          t2 - t1))
            file.write('{} [*epoch]: {},lr_net: {}, loss: {}, time: {:.3f}'
                       .format(time.strftime("%H:%M:%S", time.localtime()), e + 1, lr_step_all.get_last_lr(), losses.avg,
                               t2 - t1))
            file.write('\n')
            file.write('{}'.format(losses.avg))
            file_loss.write('\n')

            if (e + 1) % 10 == 0:
                print('save checkpoint_{}'.format(e + 1))
                torch.save(net.state_dict(), os.path.join(checkpoint, 'net_parameter_{}.pth'.format(e + 1)))
                torch.save(backbone.state_dict(), os.path.join(checkpoint, 'backbone_parameter_{}.pth'.format(e + 1)))

    else:
        print("Real data training")

        # parameters
        train_config = config['training']
        # pretrain_epoch = train_config['pretrain_epoch']
        train_epoch = train_config['real_epoch']
        batch_size = train_config['batch_size']
        num_workers = train_config['num_workers']
        lr_rate2 = train_config['lr_rate2']
        lr_decaystep = train_config['lr2_decaystep']
        lr_gamma = train_config['lr_gamma']
        drop_last = train_config['drop_last']
        tof_config = config['ToF_configs_train_real']

        dataset_config = config['DataTrain_configs']
        TrainData_path = dataset_config['TrainData_path_316']
        LabelData_path = dataset_config['LabelData_path_316']
        precheckpoint = dataset_config['precheckpoint_real']
        checkpoint = dataset_config['checkpoint_real']
        presample = dataset_config['presample']
        sample = dataset_config['sample']
        mGPU = dataset_config['mgpu']

        # time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_name = 'loss_log_real_{}.txt'.format(time.strftime("%Y-%m-%d", time.localtime()))
        file = open(log_name, 'a')

        if not os.path.exists(precheckpoint):
            os.makedirs(precheckpoint)
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        if not os.path.exists(presample):
            os.makedirs(presample)
        if not os.path.exists(sample):
            os.makedirs(sample)

        backbone = UNet()
        net = FracDiff(args)

        if args.cuda:
            net = net.cuda()
            backbone = backbone.cuda()
        if args.load_ckt:
            checkpoint_train_backbone = dataset_config['checkpoint_train_backbone']
            backbone.load_state_dict(torch.load(checkpoint_train_backbone))
            checkpoint_train_net = dataset_config['checkpoint_train_net']
            net.load_state_dict(torch.load(checkpoint_train_net))
            print('Checkpoints have been loaded!')
        if mGPU:
            net = nn.DataParallel(net)
            backbone = nn.DataParallel(backbone)

        train_set = ToF_Dataset.TOF_data(TrainData_path, LabelData_path)
        train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=drop_last)
        print('Data Loading Complete')
        optim_all = torch.optim.Adam(itertools.chain(backbone.parameters(), net.parameters()), lr=lr_rate2)
        lr_step_all = lr_scheduler.StepLR(optim_all, step_size=lr_decaystep, gamma=lr_gamma)

        cudnn.benchmark = True
        # DepthAcquire = ToF_Depth.DepthAcquisitoin()
        depth_grad = Loss.DepthGrad()

        ######################train complex U-net using cascaded network
        start = 0
        loss_list = 1000
        net = net.train()
        backbone = backbone.train()
        for e in range(start, train_epoch):
            t1 = time.time()
            for img1, img2 in train_data:
                # img1: (raw4, raw5, raw6, raw7, RealValue, ImagValue)
                # img2: (raw4, raw5, raw6, raw7, RealValue, ImagValue, Depth)
                img1 = img1.cuda()  # img1: torch.Size([16, 6, 176, 240])
                img2 = img2.cuda()  # img2: torch.Size([16, 7, 176, 240])
                with torch.autograd.set_detect_anomaly(True):
                    depth_label_tmp = img2[:, 6, ...]  # / 1000     # torch.Size([16, 176, 240])
                    label_raw = ToF_Dataset.transform(img2)
                    origin_raw = ToF_Dataset.transform(img1)
                    depth_label = ToF_Dataset.Label2Depth(label_raw, depth_label_tmp,
                                                          isFlatten=False).squeeze()  # depth: ([16, 176, 240])

                    depth_origin_tmp = ToF_Dataset.Label2Depth(origin_raw, depth_label_tmp, isFlatten=False).squeeze()
                    mask_valid, mask3, mask4, weights, confidence_aug = ToF_Depth.compute_mask(img1, img2, depth_label, depth_origin_tmp,
                                                                               is_synthetic, tof_config)
                    input = ToF_Dataset.transform(img1)
                    out_dn_res = backbone(input)  # output complex values
                    out_dn = out_dn_res + input

                    depth = ToF_Dataset.Label2Depth(out_dn, depth_label, isFlatten=False)  # depth: ([16, 1, 176, 240])

                    # depth_label = DepthAcquire.flatten(depth_label / 1000)
                    rho_loss, complex_loss, amplitude, M = ToF_Depth.CosineLoss(out_dn, img2, mask3, mask_valid,
                                                                                depth_label, tof_config)
                    output = net(depth, amplitude.unsqueeze(1), confidence_aug)  # depth refinement
                    pred = output['y_pred']
                    depth_dn = pred[0].squeeze()

                    label_mask = depth_label.detach()
                    for j in range(len(pred)):
                        if j == 0:
                            DepthLoss = torch.sum(torch.abs(pred[j].squeeze() - label_mask) * mask3) / M
                        else:
                            DepthLoss += torch.sum(torch.abs(pred[j].squeeze() - label_mask) * mask3) / M * 0.2

                    DepthGrad_loss = torch.sum(depth_grad(depth_dn, label_mask) * mask3) / M
                    total_loss = 1e0 * DepthLoss + DepthGrad_loss * 1e0 + complex_loss * 1e-1

                    optim_all.zero_grad()
                    total_loss.backward()
                    optim_all.step()
                    losses.update(total_loss.item())

            lr_step_all.step()
            t2 = time.time()

            print('{} [*epoch]: {},lr_net: {}, loss: {:.3f}, time: {:.3f}'
                  .format(time.strftime("%H:%M:%S", time.localtime()), e + 1, lr_step_all.get_last_lr(), losses.avg, t2 - t1))
            file.write('{} [*epoch]: {},lr_net: {}, loss: {}, time: {:.3f}'
                       .format(time.strftime("%H:%M:%S", time.localtime()), e + 1, lr_step_all.get_last_lr(), losses.avg, t2 - t1))
            file.write('\n')

            if (e + 1) % 50 == 0:
                print('save checkpoint_{}'.format(e + 1))
                torch.save(net.state_dict(), os.path.join(checkpoint, 'net_parameter_{}.pth'.format(e + 1)))
                torch.save(backbone.state_dict(),
                           os.path.join(checkpoint, 'backbone_parameter_{}.pth'.format(e + 1)))
