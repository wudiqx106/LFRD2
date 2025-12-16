'''
Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond
'''

import prms_init
# from torch.cuda.amp import autocast,GradScaler
import test
import train


if __name__ == '__main__':
    FPN_Path = '/media/qiao/data/Data_set/ToF316/dark/Train_Dark1.pkl'
    args, config = prms_init.prms_config()

    if args.train:
        train.train(args, config)
    else:
        test.test(args, config)
