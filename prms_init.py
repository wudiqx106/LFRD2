from configobj import ConfigObj
from validate import Validator
import argparse


def read_config(config_file, config_spec):
    configspec = ConfigObj(config_spec, raise_errors=True)
    config = ConfigObj(config_file,
                       configspec=configspec,
                       raise_errors=True,
                       file_error=True)
    config.validate(Validator())

    return config


def prms_config():
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--config_file', dest='config_file', default='model_specs/model_config.conf', help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec', default='model_specs/model_config_spec.conf',
                        help='path to config spec file')
    parser.add_argument('--cuda', '-c', default=True, action='store_true', help='whether to work on the GPU')
    parser.add_argument('--mGPU', '-m', default=False, action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', type=str, default='best',
                        help='the checkpoint to eval')

    # fractional diffusion
    parser.add_argument('--prop_kernel', type=int, default=3, help='propagation kernel size')
    parser.add_argument('--preserve_input', action='store_true', help='preserve input points by replacement')
    parser.add_argument('--from_scratch', action='store_true', default=True, help='train from scratch')
    parser.add_argument('--prop_time', type=int, default=6, help='number of propagation')
    parser.add_argument('--affinity', type=str, default='TGASS', choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, '
                             'static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma', type=float, default=0.5,
                        help='affinity gamma initial multiplier '
                             '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--conf_prop', default=True, action='store_true', help='confidence for propagation')
    parser.add_argument('--no_conf', action='store_false', dest='conf_prop', help='no confidence for propagation')
    parser.add_argument('--legacy', default=False, help='legacy code support for pre-trained models')

    parser.add_argument('--train', action='store_true', help='whether to work on the training mode')
    parser.add_argument('--is_synthetic', action='store_true', help='whether to train on synthetic data')
    parser.add_argument('--is_normalized', action='store_true', help='whether to train on normalized data')
    parser.add_argument('--load_ckt', action='store_true', help='whether to load checkpoint')
    args = parser.parse_args()
    config = read_config(args.config_file, args.config_spec)

    return args, config
