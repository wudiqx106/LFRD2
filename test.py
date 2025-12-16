from torch.utils.data import DataLoader
import os
import ToF_DataProcess
import time
import pickle
import ToF_Depth
import torch.backends.cudnn as cudnn
from ToF_DataProcess import SaveDepth
from model.cplx import *
import ToF_Dataset
import numpy as np


def test(args, config):
    global test_path_UDC, test_path_normal
    is_synthetic = args.is_synthetic

    if is_synthetic:
        print('Synthetic data test')
        test_config = config['DataTest_configs']
        test_path_generate = test_config['test_path_generate']

        test_path = test_config['test_synthetic']
        tof_config = config['ToF_configs_synthetic']
        test_path_UDC = tof_config['test_synthetic_out1']
        test_path_normal = tof_config['test_synthetic_out2']
        test_path_origin = tof_config['test_synthetic_out3']

        # if not os.path.exists(test_path_evaluate):
        #     os.makedirs(test_path_evaluate)
        if not os.path.exists(test_path_generate):
            os.makedirs(test_path_generate)
        # if not os.path.exists(test_path_confidence):
        #     os.makedirs(test_path_confidence)

        net = FracDiff(args)
        backbone = UNet()
        checkpoint_test_backbone = test_config['checkpoint_test_backbone']
        checkpoint_test_net = test_config['checkpoint_test_net']
        backbone.load_state_dict(torch.load(checkpoint_test_backbone))
        net.load_state_dict(torch.load(checkpoint_test_net))
        net.eval()
        backbone.eval()

        if args.cuda:
            net = net.cuda()
            backbone = backbone.cuda()

        if not os.path.exists(test_path_UDC):
            os.makedirs(test_path_UDC)
        if not os.path.exists(test_path_normal):
            os.makedirs(test_path_normal)
        if not os.path.exists(test_path_origin):
            os.makedirs(test_path_origin)

        test_set = ToF_Dataset.ToF_data2(test_path, training=False)
        test_data = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
        print('Data Loading Complete')
        DepthAcquire = ToF_Depth.DepthAcquisitoin()
        cudnn.benchmark = True
        sum = 0
        i = 0
        for img1, img2 in test_data:
            with torch.no_grad():
                img1 = img1.cuda()
                img2_cuda = img2.cuda()
                img2 = img2.numpy()  # torch.Size([1, 5, 176, 240])
                depth_origin_tmp0 = img1[0:1, 4, ...]
                depth_origin_tmp = depth_origin_tmp0.cpu().numpy().squeeze()
                depth_normal_tmp0 = img2[0:1, 4, ...].squeeze()
                depth_normal_tmp = depth_normal_tmp0.squeeze()

                t1 = time.perf_counter()  # beginning of the algorithm
                ReplicationPad = nn.ReflectionPad2d(padding=(4, 4, 2, 2))
                input = ReplicationPad(img1)

                input = ToF_Dataset.transform(input)
                out_dn_res = backbone(input)  # output complex values
                out = out_dn_res + input


                out = out[:, :, 2:-2, 4:-4, :].squeeze()
                depth_out = DepthAcquire.DepthLabelUnwrapping_test \
                    (out, torch.from_numpy(depth_normal_tmp).cuda(), isFlatten=True).unsqueeze(dim=0).unsqueeze(
                    dim=0)

                amplitude = torch.sqrt(input[..., 0] ** 2 + input[..., 1] ** 2) / 2
                mask_valid, mask3, mask4, weights, confidence_aug = ToF_Depth.compute_mask(img1, img2_cuda,
                                                                                           img2_cuda[0:1, 4, ...],
                                                                                           depth_origin_tmp0,
                                                                                           is_synthetic, tof_config)

                depth_out = net(depth_out, amplitude[:, :, 2:-2, 4:-4], confidence_aug)

                pred = depth_out['y_pred']
                depth_dn = pred[0].squeeze()
                depth_dn = depth_dn.detach().cpu().squeeze().numpy()
                t2 = time.perf_counter()  # end of the algorithm

                depth_origin = ToF_DataProcess.RadiationCorr(depth_origin_tmp, tof_config)
                depth_normal = ToF_DataProcess.RadiationCorr(depth_normal_tmp, tof_config)
                mask, _ = ToF_DataProcess.get_mask(img2, depth_dn, depth_normal, tof_config, isSynthetic=True)
                depth_normal *= mask
                depth_dn *= mask

                sum += np.sum(np.abs(depth_normal - depth_dn)) / np.sum(mask)
                # ToF_DataProcess.SaveTXT(i, depth_dn, depth_normal, depth_origin, test_path_UDC, test_path_normal,
                #                         test_path_origin, tof_config)

                print('Image NO: {} || time consuming: {:.2f} || MAE：{}'.format(i, (t2 - t1) * 1000,
                                                                                np.sum(np.abs(
                                                                                    depth_normal - depth_dn)) / np.sum(
                                                                                    mask)))

                i = i + 1
        print(sum / 1203)
    else:
        print('Real data test')
        test_config = config['DataTest_configs']
        test_path_mask = test_config['test_path_mask']
        test_path_generate = test_config['test_path_generate']

        test_path_train = test_config['test_path1']
        test_path_label = test_config['test_path2']

        tof_config = config['ToF_configs_real']
        test_path_UDC = tof_config['test_real_out1']
        test_path_normal = tof_config['test_real_out2']
        test_path_origin = tof_config['test_real_out3']

        if not os.path.exists(test_path_mask):
            os.makedirs(test_path_mask)
        if not os.path.exists(test_path_generate):
            os.makedirs(test_path_generate)

        backbone = UNet()
        checkpoint_test_backbone = test_config['checkpoint_test_backbone']
        backbone.load_state_dict(torch.load(checkpoint_test_backbone))
        backbone.eval()

        net = FracDiff(args)
        checkpoint_test_net = test_config['checkpoint_test_net']
        net.load_state_dict(torch.load(checkpoint_test_net))
        net.eval()

        if args.cuda:
            net = net.cuda()
            backbone = backbone.cuda()

        if not os.path.exists(test_path_UDC):
            os.makedirs(test_path_UDC)
        if not os.path.exists(test_path_normal):
            os.makedirs(test_path_normal)
        if not os.path.exists(test_path_origin):
            os.makedirs(test_path_origin)

        test_set = ToF_Dataset.TOF_data(test_path_train, test_path_label, training=False)
        test_data = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
        DepthAcquire = ToF_Depth.DepthAcquisitoin()
        cudnn.benchmark = True
        i = 0  # number of images
        sum = 0
        for img1, img2 in test_data:
            with torch.no_grad():
                depth_origin_tmp = ToF_DataProcess.Unwrapping(img1.numpy(), tof_config, isFilter=False).squeeze()
                img1 = img1.cuda()  # torch.Size([1, 8, 180, 240])
                img2_cuda = img2.cuda()
                img2 = img2.numpy()  # torch.Size([1, 8, 176, 240])
                depth_normal_tmp = ToF_DataProcess.Unwrapping(img2, tof_config, isFilter=True).squeeze()  #
                fx = tof_config['fx']
                fy = tof_config['fy']
                cx = tof_config['cx']
                cy = tof_config['cy']
                height = 180
                width = 240
                W, H = torch.tensor(np.meshgrid(range(width), range(height))).cuda()
                intrix = ((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1
                confidence = torch.abs(img2_cuda[:, 3, ...] - img2_cuda[:, 1, ...]) + torch.abs(
                    img2_cuda[:, 0, ...] - img2_cuda[:, 2, ...])
                confidence_aug = (confidence * intrix) ** 2 * torch.tensor(depth_origin_tmp).cuda() / 1000
                confidence_aug = confidence_aug.type(torch.FloatTensor).cuda()

                t1 = time.perf_counter()  # beginning of the algorithm
                ReplicationPad = nn.ReflectionPad2d(padding=(4, 4, 2, 2))

                input = ReplicationPad(img1)
                input = ToF_Dataset.TestTransform(input)
                out_dn_res = backbone(input)  # output complex values
                out = out_dn_res + input

                out = out[:, :, 2:-2, 4:-4, :].squeeze()

                depth_out = DepthAcquire.DepthLabelUnwrapping_test \
                    (out, torch.from_numpy(depth_normal_tmp).cuda(), isFlatten=True).unsqueeze(dim=0).unsqueeze(dim=0)

                # depth refinement
                amplitude = torch.sqrt(input[..., 0] ** 2 + input[..., 1] ** 2) / 2

                depth_out = depth_out.type(torch.FloatTensor).cuda()
                depth_out = net(depth_out, amplitude[:, :, 2:-2, 4:-4], confidence_aug.unsqueeze(0))  # output depth

                pred = depth_out['y_pred']
                depth_dn = pred[0].squeeze()
                depth_dn = depth_dn.detach().cpu().squeeze().numpy()
                t2 = time.perf_counter()  # end of the algorithm

                depth_normal = ToF_DataProcess.RadiationCorr(depth_normal_tmp, tof_config)
                depth_origin = ToF_DataProcess.RadiationCorr(depth_origin_tmp, tof_config)

                temp = out.detach().cpu().squeeze().numpy()  # compute the amplitude
                confidence_UDC = np.abs(temp[..., 0]) + np.abs(temp[..., 1])

                mask, confidence_label = ToF_DataProcess.get_mask(img2, depth_dn, depth_normal, tof_config)

                depth_dn *= mask
                depth_normal *= mask
                depth_origin *= mask

                print('Image NO: {}\t || time consuming: {:.2f}\t || MAE：{}'.format(i, (t2 - t1) * 1000,
                                                                                np.sum(np.abs(
                                                                                    depth_normal - depth_dn)) / np.sum(
                                                                                    mask)))

                sum += np.sum(np.abs(depth_normal - depth_dn)) / np.sum(mask)
                # save data
                # cv2.imwrite(test_path_normal + '/ir_normal_{}.png'.format(i), confidence_label)
                # cv2.imwrite(test_path_UDC + '/ir_eval_{}.png'.format(i), confidence_UDC)
                # cv2.imwrite(test_path_mask + '/mask_{}.png'.format(i), mask*255)
                ToF_DataProcess.SaveTXT(i, depth_dn, depth_normal, depth_origin, test_path_UDC, test_path_normal,
                                        test_path_origin, tof_config)

                i = i + 1
        print(sum / 105)
