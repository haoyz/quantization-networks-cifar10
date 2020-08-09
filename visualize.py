import torch
import models
import argparse
import datasets
import numpy as np
import os.path as osp
from models.qw import WQuantization
import matplotlib.pyplot as plt
from utils.training import init_model, resume_checkpoint


def get_qw_values(wk):
    if wk == '3-pm-2':
        qw_values = [-2, -1, 0, 1, 2]
    elif wk == '3-pm-4':
        qw_values = [-4, -2, -1, 0, 1, 2, 4]
    else:
        wk = int(wk)
        if wk == 1:
            qw_values = [-1, 1]
        else:
            qw_values = list(range(1 - pow(2, wk - 1), pow(2, wk - 1)))
    return qw_values


def count_bins(param, bin_size):
    param = param.reshape(-1)
    param = np.sort(param)
    total = param.shape[0]
    min_value, max_value = param[0], param[-1]
    l = min_value - bin_size
    proportions = []
    lower_bounds = []
    while l < max_value + bin_size * 2:
        mask = np.logical_and(param >= l, param < l + bin_size)
        count = np.sum(mask)
        proportions.append(count / total)
        lower_bounds.append(l)
        l += bin_size
    return proportions, lower_bounds


def main(args):
    train_loader, val_loader, num_classes = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset),
                                                            args.img_size, args.scale_size, args.batch_size,
                                                            args.workers)
    model, count, alpha, beta = init_model(args, num_classes, QA_flag=args.qa)
    w_quantizer = WQuantization(model, alpha, beta, QW_values=get_qw_values(args.wk), initialize_biases=False)
    args.resume = args.checkpoint
    args.resume_epoch = 0
    alpha, beta = resume_checkpoint(args, model, None, None, None, w_quantizer)
    for i in range(len(alpha)):
        alpha[i].cpu()
        beta[i].cpu()

    for i, module in enumerate(w_quantizer.target_modules):
        param = module.data.numpy()
        counts, lower_bounds = count_bins(param, 0.01)
        plt.plot(lower_bounds, counts)
        plt.title('Module {} parameter distribution'.format(i))
        plt.show()

        q_param = w_quantizer.forward(module.data * beta[i].data.cpu().detach(), 1, w_quantizer.QW_biases[i], train=False)
        q_param = q_param.detach().cpu().numpy()
        counts, lower_bounds = count_bins(q_param, 1)
        plt.plot(lower_bounds, counts, 'r')
        plt.title('Quantized module {} parameter distribution'.format(i))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('checkpoint', type=str, help='path to quantized model')
    # Data configs
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--scale_size', type=int, default=256, help="val resize image size, default: 256 for ImageNet")
    parser.add_argument('--img_size', type=int, default=224, help="input image size, default: 224 for ImageNet")

    # Model configs
    parser.add_argument('-a', '--arch', type=str, default='alexnet', choices=models.names())

    # Quantization training configs
    parser.add_argument('--qw', action='store_true', help='quantize weights')
    parser.add_argument('--train_qw', action='store_true', help='train weight quantization')
    parser.add_argument('--qa', action='store_true', help='quantize activations')
    parser.add_argument('--train_qa', action='store_true', help='train activation quantization')
    parser.add_argument('--ak', type=int, default=1, help='activation quantization bit width')
    parser.add_argument('--wk', type=str, default='1',
                        help='weight quantization bit width, integer or \'3-pm-2\'/\'3-pm-4\'')
    parser.add_argument('-T', '--temperature', type=int, default=10)
    parser.add_argument('--qa_gamma', type=float, default=0.0027,
                        help='gamma value for activation quantization outliers')
    parser.add_argument('--qa_sample_batch_size', type=int, default=1000,
                        help='batch size of sample for initialization of activation quantization')

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
