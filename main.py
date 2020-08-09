#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py is used to train the weight quantized model.

from __future__ import print_function, absolute_import
import argparse
from torch.backends import cudnn

from utils.training import *
from utils.evaluation import *

import logging


def main(args):
    args.qw = args.qw or args.train_qw
    args.qa = args.qa or args.train_qa

    logging.basicConfig(level=logging.DEBUG)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    data_dir = osp.join(args.data_dir, args.dataset)
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        sys.stdout = Logger(osp.join(args.logs_dir, 'evaluate-log.txt'))
    print('\n################## setting ###################')
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print('################## setting ###################\n')

    train_loader, val_loader, num_classes = datasets.create(args.dataset, data_dir, args.img_size, args.scale_size,
                                                            args.batch_size, args.workers)

    # Create model
    model, count, alpha, beta = init_model(args, num_classes)

    # Create optimizers
    optimizer, optimizer_alpha, optimizer_beta = init_optimizers(args, model.parameters(), alpha, beta)

    # Create LR schedulers
    lr_scheduler, lr_scheduler_alpha, lr_scheduler_beta = init_schedulers(args, optimizer, optimizer_alpha,
                                                                          optimizer_beta)

    # Load in-quantized model from checkpoint
    if args.pretrained:
        load_pretrained_checkpoint(args, model)

    # Quantization of all weights is performed with a single QuaOp object
    w_quantizer = init_weight_quantization(args, model, alpha, beta)

    # Resume quantization training or starting quantizing other parts
    if args.resume:
        alpha, beta = resume_checkpoint(args, model, optimizer, optimizer_alpha, optimizer_beta, w_quantizer)

    # Initialize parameters for activation quantization
    if args.qa and not model.QA_inited:
        # assert args.qw, 'Weights have to be quantized if activations are to be quantized'
        logging.info('Quantization parameters for activation are not initialized. Initializing...')
        init_activation_quantization(args, model, w_quantizer, alpha, beta)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_alpha': optimizer_alpha.state_dict() if args.qw else None,
            'optimizer_beta': optimizer_beta.state_dict() if args.qw else None,
            'alpha': alpha,
            'beta': beta,
            'bias': w_quantizer.QW_biases if w_quantizer is not None else None
        }
        save_checkpoint(checkpoint, False, fpath=osp.join(args.logs_dir, 'initialized.pth.tar'))

    # Freeze weight quantization parameters if not training weight quantization
    if not args.train_qw:
        for alpha_var in alpha:
            alpha_var.requires_grad = False
        for beta_var in beta:
            beta_var.requires_grad = False

    # Load onto GPU
    model = nn.DataParallel(model).cuda()

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    evaluator = Evaluator(model, criterion, alpha, beta, w_quantizer, args.logs_dir, quan_weight=args.qw)
    if args.evaluate:
        print('Test model: \n')
        evaluator.evaluate(val_loader, W_T=1)
        return

    # Trainer
    trainer = Trainer(model, criterion, alpha, beta, w_quantizer, args.logs_dir,
                      quan_weight=args.qw, quan_activation=args.qa,
                      train_quan_weight=args.train_qw, train_quan_activation=args.train_qa, freeze_bn=args.freeze_bn)

    # Start training
    start_epoch = best_top1 = 0
    trainer.show_info(with_arch=True, with_grad=False, with_weight_quan=True)
    for epoch in range(start_epoch, args.epochs):
        # adjust_lr(args, epoch, optimizer)
        t = (epoch + 1) * args.temperature  # linear
        print('W_T = ', t)

        trainer.train(epoch, train_loader, optimizer, optimizer_alpha, optimizer_beta, W_T=t, ac_T=t,
                      print_info=args.print_info)

        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, W_T=t, epoch=epoch)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_alpha': optimizer_alpha.state_dict() if args.qw else None,
            'optimizer_beta': optimizer_beta.state_dict() if args.qw else None,
            'alpha': alpha,
            'beta': beta,
            'bias': w_quantizer.QW_biases if w_quantizer is not None else None
        }

        lr_scheduler.step(epoch)
        if args.qw:
            lr_scheduler_alpha.step(epoch)
            lr_scheduler_beta.step(epoch)

        save_checkpoint(checkpoint, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.2%}  model_best: {:5.2%} \n'.
              format(epoch, top1, best_top1))

        if (epoch + 1) % 5 == 0:
            save_checkpoint(checkpoint, False, fpath=osp.join(args.logs_dir, 'epoch_' + str(epoch) + '.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # Data configs
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--scale_size', type=int, default=256, help="val resize image size, default: 256 for ImageNet")
    parser.add_argument('--img_size', type=int, default=224, help="input image size, default: 224 for ImageNet")

    # Model configs
    parser.add_argument('-a', '--arch', type=str, default='alexnet', choices=models.names())

    # Optimizer configs
    parser.add_argument('--adam', action='store_true', help="use Adam optimizer (SGD by default)")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--decay_steps', type=int, nargs='*', default=[100, 150])

    # Common training configs
    parser.add_argument('--pretrained', type=str, default='', metavar='PATH',
                        help='path to full precision pre-trained model to load')
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to W/A/WA quantized model to load')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--print-info', type=int, default=10)

    # Quantization training configs
    parser.add_argument('--qw', action='store_true', help='quantize weights')
    parser.add_argument('--train_qw', action='store_true', help='train weight quantization')
    parser.add_argument('--qa', action='store_true', help='quantize activations')
    parser.add_argument('--train_qa', action='store_true', help='train activation quantization')
    parser.add_argument('--wk', type=str, default='1',
                        help='weight quantization bit width, integer or \'3-pm-2\'/\'3-pm-4\'')
    parser.add_argument('--ak', type=int, default=1, help='activation quantization bit width')
    parser.add_argument('-T', '--temperature', type=int, default=10)
    parser.add_argument('--qa_gamma', type=float, default=0.0027,
                        help='gamma value for activation quantization outliers')
    parser.add_argument('--qa_sample_batch_size', type=int, default=1000,
                        help='batch size of sample for initialization of activation quantization')
    parser.add_argument('--freeze_bn', action='store_true', help='set to True to freeze BN layer while training')

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
