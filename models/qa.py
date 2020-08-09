#!/usr/bin/env python
# -*- coding: utf-8 -*-
# qa.py is used to quantize the activation of model.
from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np

from utils.cluster import params_cluster


class SigmoidT(torch.autograd.Function):
    """ sigmoid with temperature T for training
        we need the gradients for input and bias
        for customization of function, refer to https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input, scales, n, b, T):
        """
        Sigmoid T forward propagation.
        Formula:
            \sum_i^N{ \frac{ 1 }{1 + e^{T * {x - b}}} }

        Args:
            ctx: A SigmoidTBackward context object.
            input: The input tensor, which is a parameter in a network.
            scales: A list of floating numbers with length = n. The scales of the unit step functions.
            n: An integer. The number of possible quantization values - 1.
            b: A list of integers with length = n. The biases of the unit step functions.
            T: An integer. The temperature.

        Returns:
            A tensor with same shape as the input.
        """
        ctx.save_for_backward(input)
        ctx.T = T
        ctx.b = b
        ctx.scales = scales
        ctx.n = n

        # \sum_i^n{ sigmoid(T(beta * x_i - b_i)) }
        buf = ctx.T * (input - ctx.b[0])
        buf = torch.clamp(buf, min=-10.0, max=10.0)
        output = ctx.scales[0] / (1.0 + torch.exp(-buf))
        for k in range(1, ctx.n):
            buf = ctx.T * (input - ctx.b[k])
            buf = torch.clamp(buf, min=-10.0, max=10.0)
            output += ctx.scales[k] / (1.0 + torch.exp(-buf))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of the activation quantization.
        Args:
            ctx: A SigmoidTBackward context object.
            grad_output: The gradients propagated backwards to this layer.

        Returns:
            A tuple of 5 elements. Gradients for input, scales, n, b and T.
            However, none of scales, n, b and T require gradients so only the first element is not None.
        """
        # set T = 1 when train binary model in the backward.
        ctx.T = 1
        input, = ctx.saved_tensors
        b_buf = ctx.T * (input - ctx.b[0])
        b_buf = torch.clamp(b_buf, min=-10.0, max=10.0)
        b_output = ctx.scales[0] / (1.0 + torch.exp(-b_buf))
        temp = b_output * (1 - b_output) * ctx.T
        for j in range(1, ctx.n):
            b_buf = ctx.T * (input - ctx.b[j])
            b_buf = torch.clamp(b_buf, min=-10.0, max=10.0)
            b_output = ctx.scales[j] / (1.0 + torch.exp(-b_buf))
            temp += b_output * (1 - b_output) * ctx.T
        grad_input = Variable(temp) * grad_output
        # corresponding to grad_input
        return grad_input, None, None, None, None


sigmoidT = SigmoidT.apply


def step(x, b):
    """ 
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x)
    mask = torch.gt(x - b, 0.0)
    y[mask] = 1.0
    return y


class Quantization(nn.Module):
    """
    Quantization Activation. Only used when activations are quantized too.
    Args:
       quant_values: the target quantized values, like [-4, -2, -1, 0, 1 , 2, 4]
       quan_bias and init_beta: the data for initialization of quantization parameters (biases, beta)
                  - for activations, format as `N x 1` for biases and `1x1` for (beta)
                    we need to obtain the intialization values for biases and beta offline

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Usage:
        - for activations, just pending this module to the activations when build the graph
    """

    def __init__(self, quant_values, outlier_gamma=0.001):
        super(Quantization, self).__init__()
        self.values = quant_values
        self.outlier_gamma = outlier_gamma
        # number of sigmoids
        self.n = len(self.values) - 1
        self.alpha = Parameter(torch.Tensor([1]))
        self.beta = Parameter(torch.Tensor([1]))
        self.register_buffer('biases', torch.zeros(self.n))
        self.register_buffer('scales', torch.zeros(self.n))

        # boundary = np.array(quan_bias)
        self.init_scale_and_offset()
        self.inited = False
        # self.init_biases(boundary)
        # self.init_alpha_and_beta(init_beta)

    def init_scale_and_offset(self):
        """
        Initialize the scale and offset of quantization function.
        """
        for i in range(self.n):
            gap = self.values[i + 1] - self.values[i]
            self.scales[i] = gap

    def init_biases(self, biases):
        """
        Initialize the bias of quantization function.
        init_data in numpy format.
        """
        # activations initialization (obtained offline)
        assert biases.size == self.n
        self.biases.copy_(torch.from_numpy(biases))
        # print('baises inited!!!')

    def init_alpha_and_beta(self, beta):
        """
        Initialize the alpha and beta of quantization function.
        init_data in numpy format.
        """
        # activations initialization (obtained offline)
        self.beta.data = torch.Tensor([beta]).cuda()
        self.alpha.data = torch.reciprocal(self.beta.data)

    def forward(self, input, T=1):
        if not self.inited:
            print('Initializing activation quantization layer')
            params = input.data.detach().cpu().numpy()
            biases, (min_value, max_value) = params_cluster(params, self.values, gamma=self.outlier_gamma)
            print('biases = {}'.format(biases))
            self.init_biases(np.array(biases))
            # Method in Quantization Networks
            # self.init_alpha_and_beta((self.values[-1] * 5) / (4 * input.data.abs().max()))

            # Method in Fully Quantized Networks
            self.init_alpha_and_beta(self.values[-1] / max_value)
            self.inited = True
            return input

        input = input.mul(self.beta)
        if self.training:
            output = sigmoidT(input, self.scales, self.n, self.biases, T)
        else:
            output = step(input, b=self.biases[0]) * self.scales[0]
            for i in range(1, self.n):
                output += step(input, b=self.biases[i]) * self.scales[i]

        output = output.mul(self.alpha)
        return output

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        alpha_key = prefix + 'alpha'
        beta_key = prefix + 'beta'
        if alpha_key in state_dict and beta_key in state_dict:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                          missing_keys, unexpected_keys, error_msgs)
            self.inited = True
        else:
            error_msgs.append('Activation quantization parameters not found for {} '.format(prefix[:-1]))

    def __repr__(self):
        return 'Quantization(alpha={}, beta={}, values={}, n={})'.format(self.alpha.data, self.beta.data, self.values,
                                                                         self.n)
