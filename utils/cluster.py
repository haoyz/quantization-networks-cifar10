#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cluster.py is used to get the bias(b_i) of quantization function.

from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import os

import sys
import pdb

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def params_cluster(params, Q_values, gamma=0):
    """
    Cluster parameter values into (2^k - 1) peaks.
    Args:
        params: A numpy.ndarray object. The parameters to be clustered.
        Q_values: A list of integers. The quantization values.
        gamma: A floating number in range [0, 0.5). Parameter values in percentiles [0, gamma) and (1 - gamma, 1] are outliers to be abandoned.

    Returns:
        A list of floating numbers. The (2^k - 1) peaks.
    """
    print("The max and min values of params: ", params.max(), params.min())
    print("The shape of params: ", params.shape)

    pre_params = np.sort(params.reshape(-1, 1), axis=0)
    n_params = pre_params.shape[0]

    if gamma != 0:
        assert gamma > 0 and gamma < 0.5, 'Gamma value for outliers should be in range (0, 0.5)'
        n_outliers = int(n_params * gamma)
        print('n_outliers: ', n_outliers)
        print('max_abs_value before ignoring outliers:', max(abs(pre_params[0, 0]), pre_params[-1, 0]))
        # Ignore outliers
        pre_params = pre_params[n_outliers:n_params - n_outliers]

    # max_value is the `q` in the article
    max_value = abs(pre_params).max().tolist()
    print("max_abs_value: ", max_value)

    if gamma == 0:
        # Take the method that Quantization Networks used
        pre_params = pre_params * 5 / 4.0 * (Q_values[-1] / max_value)
    else:
        # Take the method that FQN used
        pre_params = pre_params * (Q_values[-1] / max_value)

    # Cluster
    n_clusters = len(Q_values)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(pre_params)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_

    # print("cluster_centers: ", centroids)
    # print("label_pred: ", label_pred)

    temp = label_pred[0]
    saved_index = [0] * (n_clusters - 1)
    num_labels = 0
    for index, label in enumerate(label_pred):
        if label != temp:
            saved_index[num_labels] = index
            num_labels += 1
            temp = label

    # print("boundary_index: ", saved_index)

    # print(pre_params[saved_index[0]-1], pre_params[saved_index[0]])
    # print(pre_params[saved_index[1]-1], pre_params[saved_index[1]])

    boundary = [0] * (n_clusters - 1)
    for i in range(n_clusters - 1):
        temp = (pre_params[saved_index[i] - 1] + pre_params[saved_index[i]]) / 2
        boundary[i] = temp.tolist()[0]
    # print("boundary: ", boundary)
    return boundary, (pre_params[0], pre_params[-1])


def main(args):
    Q_values = [-4, -2, -1, 0, 1, 2, 4]
    # Q_values = [-2, -1, 0, 1, 2]
    # Q_values = [-1, 0, 1]

    all_file = sorted(os.listdir(args.root))
    for filename in all_file:
        if '.npy' in filename:
            params_road = osp.join(args.root, filename)
            params = np.load(params_road)
            boundary, (min_value, max_value) = params_cluster(params, Q_values)
            print(filename, boundary)
            print('min_value: ', min_value)
            print('max_value: ', max_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter cluster")
    # file road
    parser.add_argument('-r', '--root', type=str, default=".")

    main(parser.parse_args())
