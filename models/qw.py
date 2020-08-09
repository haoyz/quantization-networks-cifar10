#!/usr/bin/env python
# -*- coding: utf-8 -*-
# qw.py is used to quantize the weight of model.

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import numpy
from utils.cluster import params_cluster
import logging


def sigmoid_t(x, b=0, t=1):
    """
    The sigmoid function with T for soft quantization function.
    Args:
        x: input
        b: the bias
        t: the temperature
    Returns:
        y = sigmoid(t(x-b))
    """
    temp = -1 * t * (x - b)
    temp = torch.clamp(temp, min=-10.0, max=10.0)
    return 1.0 / (1.0 + torch.exp(temp))


def step(x, bias):
    """ 
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x)
    mask = torch.gt(x - bias, 0.0)
    y[mask] = 1.0
    return y


class WQuantization(object):
    """
    Weight quantizer.

    Args:
        model: the model to be quantified.
        QW_biases (list): the bias of quantization function.
                          QW_biases is a list with m*n shape, m is the number of layers,
                          n is the number of sigmoid_t.
        QW_values (list): the list of quantization values, 
                          such as [-1, 0, 1], [-2, -1, 0, 1, 2].

    Returns:
        Quantized model.
    """

    def __init__(self, model, alpha, beta, QW_values=None, initialize_biases=True):
        # Count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        # Omit the first conv layer and the last linear layer
        start_range = 1
        end_range = count_targets - 2
        self.bin_range = numpy.linspace(start_range,
                                        end_range, end_range - start_range + 1) \
            .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.inited = False
        # Decide modules to quantize
        self.QW_biases = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

        print('target_modules number: ', len(self.target_modules))
        # self.QW_biases = QW_biases
        self.QW_values = QW_values
        # The number of sigmoid_t, or say, the number of steps
        self.n = len(self.QW_values) - 1
        # 5p / 4
        self.threshold = self.QW_values[-1] * 5 / 4.0
        # The gap between each two adjacent quantization values
        self.scales = []
        offset = 0.
        for i in range(self.n):
            gap = self.QW_values[i + 1] - self.QW_values[i]
            self.scales.append(gap)
            offset += gap
        self.offset = offset / 2.

        for index in range(self.num_of_params):
            # if init:
            # According to the paper, beta is initialized as (5p / 4) * (1 / q),
            # where p is the max absolute value of elements in Y,
            # and q is the max absolute value of elements in X.
            # Alpha is initialized with (1 / beta) to keep the magnitude of the inputs unchanged.
            beta[index].data = torch.Tensor([self.threshold / self.target_modules[index].data.abs().max()]).cuda()
            alpha[index].data = torch.reciprocal(beta[index].data)

        if initialize_biases:
            print('Initializing weight quantization biases')
            for param in self.saved_params:
                # Do clustering
                biases, _ = params_cluster(param.detach().cpu().numpy(), QW_values)
                self.QW_biases.append(biases)
            self.inited = True

    def forward(self, x, T, quan_bias, train=True):
        """
        Forward propagation using quantized x.
        While training, a sigmoid function with temperature T is used for quantization.
        While testing, a unit step function is used instead.

        Args:
            x: The (beta * x) in the quantization formula.
            T: The temperature.
            quan_bias: The b in the quantization formula.
            train: A bool value. Is training or not.

        Returns:
            y: The quantized weight or activation without being multiplied by alpha.
                Mathematically, ( \sum_i=1^n{ s_i * step(beta * x - b_i) } - o ), or ( y_d / \alpha ).
        """
        if train:
            # \sum_i=1^n{ s_i * sigmoid(T * (beta * x_d - b_i)) } - o
            y = sigmoid_t(x, b=quan_bias[0], t=T) * self.scales[0]
            for j in range(1, self.n):
                y += sigmoid_t(x, b=quan_bias[j], t=T) * self.scales[j]
        else:
            # \sum_i=1^n{ s_i * step(beta * x - b_i) } - o
            y = step(x, bias=quan_bias[0]) * self.scales[0]
            for j in range(1, self.n):
                y += step(x, bias=quan_bias[j]) * self.scales[j]
        y = y - self.offset

        return y

    def backward(self, x, T, quan_bias):
        """
        Calculate gradients of y.

        Args:
        	x: The (beta * x) value.
        	T: The temperature value.
        	quan_bias: The bias values.

        Returns:
            y_grad: A torch.Tensor.
            Denote u_d as T(\beta * x_d - b_d^i), y_grad is \partial{y_d} / \partial{u_d} / \alpha_d.
        """
        # This is actually g_d in the formula, because it's not multiplied by alpha.
        # y_i = s_i * sigmoid(T * (beta * x_d - b_i)) = s_i * g_d_i
        y_1 = sigmoid_t(x, b=quan_bias[0], t=T) * self.scales[0]
        # y_grad = \sum_i=1^n{ y_i * (s_i - y_i) / s_i }
        y_grad = (y_1.mul(self.scales[0] - y_1)).div(self.scales[0])
        for j in range(1, self.n):
            y_temp = sigmoid_t(x, b=quan_bias[j], t=T) * self.scales[j]
            y_grad += (y_temp.mul(self.scales[j] - y_temp)).div(self.scales[j])

        return y_grad.mul(T)

    def quantize_params(self, T, alpha, beta, train=True):
        """
        Perform network weight quantization.

        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list.
            train: a flag represents the quantization
                  operation in the training stage.
        """
        # Maximum temperature is 2000
        T = (T > 2000) * 2000 + (T <= 2000) * T
        for index in range(self.num_of_params):
            # beta * x
            x = self.target_modules[index].data.mul(beta[index].data)

            # \sum_i=1^n{s_i * sigmoid(T * (beta * x_d - b_i))} - o
            y = self.forward(x, T, self.QW_biases[index], train=train)

            # alpha * \sum_i=1^n{s_i * sigmoid(T * (beta * x_d - b_i))} - o
            self.target_modules[index].data = y.mul(alpha[index].data)

    def save_params(self):
        """
        Save the full-precision parameters for backward propagation.
        """
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def restore_params(self):
        """
        Restore the full-precision parameters for backward propagation.
        """
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def update_quantization_gradients(self, T, alpha, beta):
        """
        Calculate the gradients of all the parameters (alpha and beta).
        The gradients of model parameters are saved in the [Variable].grad.data.
        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list.
        Returns:
            alpha_grad: the gradient of alpha.
            beta_grad: the gradient of beta.
        """
        beta_grad = [0.0] * len(beta)
        alpha_grad = [0.0] * len(alpha)
        # Maximum temperature is 2000
        T = (T > 2000) * 2000 + (T <= 2000) * T
        for index in range(self.num_of_params):
            # beta * x
            x = self.target_modules[index].data.mul(beta[index].data)

            # Denote u_d as T(\beta * x_d - b_d^i), y_grad is \partial{y_d} / \partial{u_d} / \alpha_d.
            # y_grad = \sum_i=1^n{ s_i * g_d_i * (1 - g_d_i) }
            # set T = 1 when train binary model
            # y_grad = self.backward(x, 1, self.QW_biases[index]).mul(T)
            # set T = T when train the other quantization model
            y_grad = self.backward(x, T, self.QW_biases[index])

            # \partial{l} / \partial{beta_d}  = ( \partial{l} / \partial{y_d} ) * ( \partial{y_d} / \partial{u_d} ) * ( \partial{u_d} / \partial{beta_d} )
            #                                 = (\partial{y_d} / \partial{u_d} / \alpha_d) * alpha_d * x_d * ( \partial{l} / \partial{y_d} )
            beta_grad[index] = y_grad.mul(self.target_modules[index].data).mul(alpha[index].data). \
                mul(self.target_modules[index].grad.data).sum()

            # \partial{l} / \partial{alpha_d} = ( \partial{l} / \partial{alpha_d} ) * ( \partial{l} / \partial{y_d} )
            #                                 = ( y_d / alpha_d ) * ( \partial{l} / \partial{y_d} )
            alpha_grad[index] = self.forward(x, T, self.QW_biases[index]). \
                mul(self.target_modules[index].grad.data).sum()

            # \partial{l} / \partial{x_d}  = ( \partial{l} / \partial{y_d} ) * ( \partial{y_d} / \partial{u_d} ) * ( \partial{u_d} / \partial{x_d} )
            #                              = (\partial{y_d} / \partial{u_d} / \alpha_d) * alpha_d * beta_d * ( \partial{l} / \partial{y_d} )
            self.target_modules[index].grad.data = y_grad.mul(beta[index].data).mul(alpha[index].data). \
                mul(self.target_modules[index].grad.data)

        return alpha_grad, beta_grad
