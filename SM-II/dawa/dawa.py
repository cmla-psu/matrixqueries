# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:45:09 2021.

@author: 10259

DAWA data dependent method.
"""

import numpy as np


def WRange(param_m, param_n):
    """Range Workload."""
    # work = np.zeros([param_m, param_n])
    query = []
    var = np.ones(param_m)
    for i in range(param_m):
        num_1 = np.random.randint(param_n)
        num_2 = np.random.randint(param_n)
        num = np.random.randint(1, 11)
        if num_1 < num_2:
            low = num_1
            high = num_2
        else:
            low = num_2
            high = num_1
        # work[i, low:high+1] = 1
        query.append([low, high])
        var[i] = num
    return query, var


def addNoise(data, epsilon):
    """Add laplace noise to ensure differential privacy."""
    b = np.ones_like(data)/epsilon
    noise = np.random.laplace(loc=0.0, scale=b)
    ndata = data + noise
    return ndata


def true_count(query, dataset):
    """Return the true count for a range query."""
    low, high = query
    return np.sum(dataset[low: high+1])


def grouping(index, data):
    """Group the data."""
    gdata = np.zeros_like(data)
    for group in index:
        low, high = group
        avg = np.mean(data[low:high+1])
        gdata[low:high+1] = avg
    return gdata


def addGNoise(index, gdata, epsilon):
    """Add noise to the grouped data."""
    ngdata = np.copy(gdata).astype(np.float16)
    for idx in index:
        low, high = idx
        length = high-low+1
        noise = np.random.normal(loc=0.0, scale=1./epsilon)/length
        for j in range(low, high+1):
            ngdata[j] = ngdata[j] + noise
    return ngdata


if __name__ == '__main__':
    # np.random.seed(0)
    param_n = 100
    param_m = 2*param_n
    # queries, bound = WRange(param_m, param_n)
    queries = np.load("query.npy")
    bound = np.load("bound.npy")

    cencus = np.load("dataset.npy")
    e1 = 0.1
    e2 = 3.75196 - 0.1
    # e2 = 0.11

    ndata = addNoise(cencus, e1)
    # load grouping information
    index = np.load("index.npy")
    gdata = grouping(index, cencus)
    ngdata = addGNoise(index, gdata, e2)

    total_var = np.zeros(param_m)
    for j in range(10):
        total = 0
        total_g = 0
        var = []
        for i in range(param_m):
            query = queries[i]
            t_count = true_count(query, cencus)
            n_count = true_count(query, ndata)
            g_count = true_count(query, ngdata)
            err = np.square(t_count-n_count)
            err_g = np.square(t_count-g_count)
            total = total + err
            var.append(err_g)
        var = np.array(var)
        total_var = total_var + var
    mean_var = total_var/10
    std = np.sqrt(mean_var)
    print(np.max(std/bound))
