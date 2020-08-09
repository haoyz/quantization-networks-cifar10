#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import
from torch.autograd import Variable

import time
from utils import *

from torch.utils.tensorboard import SummaryWriter


def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret


class Evaluator(object):
    def __init__(self, model, criterion, alpha, beta, qua_op, log_dir, quan_weight=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.w_quantizer = qua_op
        self.quan_weight = quan_weight
        self.summary_writer = SummaryWriter(log_dir + '_val')

    def evaluate(self, data_loader, W_T=1, print_freq=1, epoch=0):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()

        end = time.time()
        if self.quan_weight:
            self.w_quantizer.quantize_params(W_T, self.alpha, self.beta, train=False)

        for i, inputs in enumerate(data_loader):
            inputs_var, targets_var = self._parse_data(inputs)

            loss, prec1, prec5 = self._forward(inputs_var, targets_var)

            losses.update(loss.data, targets_var.size(0))
            top1.update(prec1, targets_var.size(0))
            top5.update(prec5, targets_var.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.4f} ({:.4f})\t'
                      'Prec@1 {:.2%} ({:.2%})\t'
                      'Prec@5 {:.2%} ({:.2%})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              top1.val, top1.avg,
                              top5.val, top5.avg))
        if self.quan_weight:
            self.w_quantizer.restore_params()

        print(' * Prec@1 {:.2%} Prec@5 {:.2%}'.format(top1.avg, top5.avg))
        self.summary_writer.add_scalar('Eval/Loss', losses.avg, global_step=epoch)
        self.summary_writer.add_scalar('Eval/Prec@1', top1.avg * 100, global_step=epoch)
        self.summary_writer.add_scalar('Eval/Prec@5', top5.avg * 100, global_step=epoch)

        return top1.avg

    def _parse_data(self, inputs):
        imgs, labels = inputs
        inputs_var = Variable(imgs, volatile=True)
        targets_var = Variable(labels.cuda(), volatile=True)
        return inputs_var, targets_var

    def _forward(self, inputs, targets):
        outputs = self.model(inputs, input_ac_T=1)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            prec1 = prec1[0]
            prec5 = prec5[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec1, prec5
