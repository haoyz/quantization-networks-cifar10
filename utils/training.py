import time
import torch
import torch.nn as nn
import os.path as osp
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import models
import datasets
from models.qw import WQuantization
from utils.evaluation import accuracy
from . import load_checkpoint, AverageMeter


def freeze_bn(model):
    """
    Freeze batch normalization parameters. Do it before quantization training.

    Args:
        model: A torch.nn.Module object.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # Freeze BN
            # m.eval()
            # Freeze BN affine
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def init_model(args, num_classes):
    """
    Initialize quantization network model, generating alpha and beta variables for each weighted layer.

    Args:
        args: The args object obtained with argparse.
        num_classes: The number of classes of image classification network.

    Returns:
        A tuple of 4 elements.
            model: The network model object (nn.Module).
            count: The number of weighted layers (nn.Conv2d and nn.Linear).
            alphas: All alpha variables in the quantization network.
            betas: All beta variables in the quantization network.
    """
    if args.qa:
        max_quan_value = pow(2, args.ak)
        ac_quan_values = [i for i in range(max_quan_value)]
        print('ac_quan_values: ', ac_quan_values)
        model = models.create(args.arch, pretrained=False, num_classes=num_classes, QA_flag=True,
                              QA_values=ac_quan_values, QA_outlier_gamma=args.qa_gamma)
    else:
        model = models.create(args.arch, pretrained=False, num_classes=num_classes)

    # Create alphas and betas if quantize weights
    count = 0
    alphas = []
    betas = []
    if args.qw:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count = count + 1
        for i in range(count - 2):
            alphas.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))
            betas.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))

    return model, count, alphas, betas


def init_optimizers(args, params, alpha, beta):
    """
    Initialize optimizers for the quantization network model.

    Args:
        args: The args object obtained with argparse.
            The following requirements should be met:
            args.lr is a valid floating point number (0~1).
            args.weight_decay is a valid floating point number (0~1).
            args.momentum is a valid floating point number (0~1).
        params: All the parameters to optimize.
        alpha: All the alpha values to optimize.
        beta: All the beta values to optimize.

    Returns:
        A tuple of 3 optimizer objects, one for network weights, one for alphas and one for betas.
    """
    optimizer_alpha = None
    optimizer_beta = None
    if args.adam:
        print('The optimizer is Adam !!!')
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        if args.qw:
            optimizer_alpha = torch.optim.Adam(alpha, lr=args.lr)
            optimizer_beta = torch.optim.Adam(beta, lr=args.lr)
    else:
        print('The optimizer is SGD !!!')
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.qw:
            optimizer_alpha = torch.optim.SGD(alpha, lr=args.lr, momentum=args.momentum)
            optimizer_beta = torch.optim.SGD(beta, lr=args.lr, momentum=args.momentum)
    return optimizer, optimizer_alpha, optimizer_beta


def init_schedulers(args, optimizer, optimizer_alpha, optimizer_beta):
    """
    Initialize LR schedulers.

    Args:
        args: The args object obtained with argparse.
            The following requirements should be met:
            args.qw is a boolean value indicating whether the model weights should be quantized.
        optimizer: The optimizer for the model.
        optimizer_alpha: The optimizer for the alpha variables in weight quantization. Can be None.
        optimizer_beta: The optimizer for the beta variables in weight quantization. Can be None.

    Returns:
        A tuple of 3 elements:
            lr_scheduler: The LR scheduler for the model's optimizer.
            lr_scheduler_alpha: The LR scheduler for the alpha optimizer.
            lr_scheduler_beta: The LR scheduler for the beta optimizer.
    """
    lr_scheduler_alpha = None
    lr_scheduler_beta = None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps, last_epoch=-1)
    if args.qw:
        lr_scheduler_alpha = torch.optim.lr_scheduler.MultiStepLR(optimizer_alpha, milestones=args.decay_steps,
                                                                  last_epoch=-1)
        lr_scheduler_beta = torch.optim.lr_scheduler.MultiStepLR(optimizer_beta, milestones=args.decay_steps,
                                                                 last_epoch=-1)

    return lr_scheduler, lr_scheduler_alpha, lr_scheduler_beta


def init_weight_quantization(args, model, alpha, beta):
    """
    Initialize weight quantizer.

    Args:
        args: The args object obtained with argparse.
            The following requirements should be met:
            args.qw is a bool value, True or False.
            args.wk is a string that contains one integer or is equal to '3-pm-2' or '3-pm-4'
        model: The model to be quantized.
        alpha: All the alpha values to optimize.
        beta: All the beta values to optimize.

    Returns:
        A WQuantization object if args.qw is True, otherwise None.
    """
    w_quantizer = None
    if args.qw:
        if args.wk == '3-pm-2':
            qw_values = [-2, -1, 0, 1, 2]
        elif args.wk == '3-pm-4':
            qw_values = [-4, -2, -1, 0, 1, 2, 4]
        else:
            wk = int(args.wk)
            if wk == 1:
                qw_values = [-1, 1]
            else:
                qw_values = list(range(1 - pow(2, wk - 1), pow(2, wk - 1)))
        print('qw_values: ', qw_values)
        w_quantizer = WQuantization(model, alpha, beta, QW_values=qw_values, initialize_biases=(not args.resume))
    return w_quantizer


def init_activation_quantization(args, model, w_quantizer, alpha, beta):
    """
    Initialize biases, alphas and betas for activation quantization

    Args:
        args: The args object obtained with argparse.
        model: The model that contains activation quantization modules.
    """
    freeze_bn(model)
    model = torch.nn.DataParallel(model).cuda()
    init_data_loader, _, _ = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset), args.img_size,
                                             args.scale_size, args.qa_sample_batch_size, args.workers)
    for i, (inputs, targets) in enumerate(init_data_loader):
        # Run the model once to initialize activation quantization parameters
        if args.qw:
            w_quantizer.save_params()
            w_quantizer.quantize_params(T=1, alpha=alpha, beta=beta, train=False)
            model(inputs.cuda(), input_ac_T=1)
            w_quantizer.restore_params()
        else:
            model(inputs.cuda(), input_ac_T=1)
        model.QA_inited = True
        break


def load_pretrained_checkpoint(args, model):
    """
    Load pre-trained weights for starting quantization training.
    Args:
        args: The args object obtained with argparse.
            The following requirements should be met:
            args.pretrained is a valid path to a pre-trained model weight file.
            args.arch is a valid network architecture name and contains 'alexnet' or 'resnet'.

        model: The quantization network model (nn.Module).
    """
    print('=> Start load params from pre-trained model...')
    checkpoint = load_checkpoint(args.pretrained)
    if 'alexnet' in args.arch or 'resnet' in args.arch or 'simplenet' in args.arch:
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise RuntimeError('The arch is ERROR!!!')


def resume_checkpoint(args, model, optimizer, optimizer_alpha, optimizer_beta, w_quantizer):
    """
    Load pre-trained checkpoint (weights and optimizers) for continuing quantization training.
    Args:
        args: The args object obtained with argparse.
            It's required that the args.resume option is a valid checkpoint file path.
        model: The quantization network model (nn.Module).
        optimizer: The optimizer for the network.
        optimizer_alpha: The optimizer for the alphas.
        optimizer_beta: The optimizer for the betas.
        w_quantizer: The weight quantizer object.

    Returns:
        A tuple of 2 elements. The alpha variables and beta variables saved in the checkpoint.
    """
    checkpoint = load_checkpoint(args.resume)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError as ex:
        error_msgs = ex.args[0].split('\n\t')[1:]
        for msg in error_msgs:
            if 'Activation quantization parameters not found' in msg:
                model.QA_inited = False
            else:
                print(msg)
    # try:
    # Params for activation quantization may not be included in the optimizers
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # optimizer_alpha.load_state_dict(checkpoint['optimizer_alpha'])
    # optimizer_beta.load_state_dict(checkpoint['optimizer_beta'])
    # except ValueError as ex:
    #     print(ex)
    alpha = checkpoint['alpha']
    beta = checkpoint['beta']
    if args.qw:
        w_quantizer.QW_biases = checkpoint['bias']
        w_quantizer.inited = True
    start_epoch = args.resume_epoch
    print("=> Finetune Start epoch {} ".format(start_epoch))
    return alpha, beta


class Trainer(object):
    def __init__(self, model, criterion, alpha, beta, qua_op, log_dir, quan_weight=False, quan_activation=False,
                 train_quan_weight=False, train_quan_activation=False, freeze_bn=False):
        """
        Constructor of the trainer.
        Args:
            model: The quantization network model (nn.Module).
            criterion: The loss criterion object (nn.Module).
            alpha: The alpha variables.
            beta: The beta variables.
            qua_op: A QuaOp object.
            log_dir: A string. The log directory that stores checkpoints and log files.
        """
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.init = False
        self.w_quantizer = qua_op
        self.quan_weight = quan_weight
        self.quan_activation = quan_activation
        self.train_quan_weight = train_quan_weight
        self.train_quan_activation = train_quan_activation
        self.freeze_bn = freeze_bn
        self.summary_writer = SummaryWriter(log_dir + '_train')

    def train(self, epoch, data_loader, optimizer, optimizer_alpha, optimizer_beta, W_T=1, ac_T=1, print_freq=1,
              print_info=10, max_norm=5.0):
        """
        Run training for one epoch.

        Args:
            epoch: The epoch number, starting with 0.
            data_loader: The training data loader.
            optimizer: The optimizer for the network.
            optimizer_alpha: The optimizer for the alphas.
            optimizer_beta: The optimizer for the betas.
            W_T: Temperature in weight quantization.
            ac_T: Temperature in activation quantization
            print_freq: An integer, defining how many iterations to print training log once.
            print_info: An integer, defining how many epoch to print trainer info once.
            max_norm: A floating number. The maximum L2 norm that gradients will be clipped with.
                In the paper, it's 5 for AlexNet classification,
                (unmentioned for ResNet classification),
                and 20 for SSD object detection.
        """
        self.model.train()
        if self.freeze_bn:
            freeze_bn(self.model)
            print('Batch normalization layer frozen')

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            if epoch == 0 and i == 0 and self.quan_weight and not self.w_quantizer.inited:
                self.init = True
            else:
                self.init = False
            data_time.update(time.time() - end)

            inputs_var, targets_var = self._parse_data(inputs)

            if self.quan_weight:
                # Save float parameters of the all modules
                self.w_quantizer.save_params()
                # Quantize convolution parameters (convolution only?)
                self.w_quantizer.quantize_params(W_T, self.alpha, self.beta, train=self.train_quan_weight)

            # Forward propagation, calculate loss value and precisions (and averages)
            loss, prec1, prec5 = self._forward(inputs_var, targets_var, ac_T)
            losses.update(loss.data, targets_var.size(0))
            top1.update(prec1, targets_var.size(0))
            top5.update(prec5, targets_var.size(0))

            optimizer.zero_grad()
            if self.quan_weight:
                optimizer_alpha.zero_grad()
                optimizer_beta.zero_grad()

            # Back propagation
            loss.backward()

            if self.quan_weight:
                # Clip gradients with a maximum L2 norm
                torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), math.sqrt(max_norm))

                # Restore params
                self.w_quantizer.restore_params()
                # Calculate gradients of alphas and betas
                alpha_grad, beta_grad = self.w_quantizer.update_quantization_gradients(W_T, self.alpha, self.beta)

                for index in range(len(self.alpha)):
                    self.alpha[index].grad = Variable(torch.FloatTensor([alpha_grad[index]]).cuda())
                    self.beta[index].grad = Variable(torch.FloatTensor([beta_grad[index]]).cuda())

            optimizer.step()
            if self.quan_weight:
                optimizer_alpha.step()
                optimizer_beta.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec@1 {:.2%} ({:.2%})\t'
                      'Prec@5 {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))

        self.summary_writer.add_scalar('Eval/Loss', losses.avg, global_step=epoch)
        self.summary_writer.add_scalar('Eval/Prec@1', top1.avg * 100, global_step=epoch)
        self.summary_writer.add_scalar('Eval/Prec@5', top5.avg * 100, global_step=epoch)
        self.summary_writer.add_scalar('LR/W', optimizer.param_groups[0]['lr'], global_step=epoch)
        if self.quan_weight:
            self.summary_writer.add_scalar('LR/Alpha', optimizer_alpha.param_groups[0]['lr'], global_step=epoch)
            self.summary_writer.add_scalar('LR/Beta', optimizer_beta.param_groups[0]['lr'], global_step=epoch)

        if (epoch + 1) % print_info == 0:
            self.show_info()

    def show_info(self, with_arch=False, with_grad=True, with_weight_quan=True):
        if with_arch:
            print('\n\n################# model modules ###################')
            for name, m in self.model.named_modules():
                print('{}: {}'.format(name, m))
            print('################# model modules ###################\n\n')

        if with_grad:
            print('################# model params diff ###################')
            for name, param in self.model.named_parameters():
                mean_value = torch.abs(param.data).mean()
                mean_grad = 1e-8 if param.grad is None else torch.abs(param.grad).mean().data + 1e-8
                print('{}: size{}, data_abd_avg: {}, dgrad_abd_avg: {}, data/grad: {}'.format(name,
                                                                                              param.size(), mean_value,
                                                                                              mean_grad,
                                                                                              mean_value / mean_grad))
            print('################# model params diff ###################\n\n')

        else:
            print('################# model params ###################')
            for name, param in self.model.named_parameters():
                print('{}: size{}, abs_avg: {}'.format(name,
                                                       param.size(),
                                                       torch.abs(param.data.cpu()).mean()))
            print('################# model params ###################\n\n')

        if with_weight_quan:
            print('################# weight quantization params ###################')
            for index, alpha_var in enumerate(self.alpha):
                print('QW#{} alpha = {}, beta = {}'.format(index, alpha_var.data, self.beta[index].data))
            print('################# weight quantization params ###################\n\n')

    def _parse_data(self, inputs):
        imgs, labels = inputs
        inputs_var = Variable(imgs)
        targets_var = Variable(labels.cuda())
        return inputs_var, targets_var

    def _forward(self, inputs, targets, ac_T):
        outputs = self.model(inputs, ac_T)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            prec1 = prec1[0]
            prec5 = prec5[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec1, prec5
