import numpy as np
import scipy as sp
from operator import mul
from collections import defaultdict
import random
import RollingWindow as rw
import pickle
from scipy import stats
import os
import shutil
import copy
from landlab import RasterModelGrid
from landlab import FIXED_VALUE_BOUNDARY
from landlab.components import FlowAccumulator, LinearDiffuser, StreamPowerEroder


def flow_acc_rec(sin):
    for don in mg['donor'][sin, mg['donor'][sin, :] >= 0]:
        mg['processed_node'][don] = 1
        flow_acc_rec(don)
        mg['drainage_area'][sin] += mg['drainage_area'][don]


def flow_accmulation():
    sinks = mg['node_id'][mg['node_type'] == 0]
    for sin in sinks:
        flow_acc_rec(sin, )


def elevation_cal(m):
    sinks = list(mg['node_id'][mg['node_type'] == 0])
    for sin in sinks:
        mg['topgraphic__elevation'][sin] = 0

    while len(sinks) > 0:
        sin = sinks.pop(0)
        for don in mg['donor'][sin]:
            if don >= 0:
                slope = mg['drainage_area'][don] ** (-m)
                mg['topgraphic__elevation'][don] = slope + mg['topgraphic__elevation'][sin]
                sinks.append(don)


def doner_find():
    receiver_of_nei = mg['receiver'][mg['neighbour']]
    receiver_of_nei = np.where(mg['neighbour'] == -1, -1, receiver_of_nei)
    for nei in range(3 ** num_dim):
        mg['donor'][:, nei] = np.where((receiver_of_nei[:, nei] == mg['node_id']), mg['neighbour'][:, nei], -1)


def fill(z, num_dim, window_size=3):
    border = np.pad(np.zeros(tuple(np.asanyarray(z.shape) - 2)), pad_width=1, mode='constant', constant_values=1)
    w = np.where(border == 0, np.max(z) + 0.01, z)
    eps = 0.001
    smt_done = 1
    s = []
    for dim in range(num_dim):
        s.append(slice(0, z.shape[dim]))

    while smt_done == 1:
        smt_done = 0
        proc_ext = np.where((w > z) & (border == 0), 1, 0).astype(np.int8)

        w_pad = np.pad(w, pad_width=1, mode='constant', constant_values=z.min() - 1.)
        z_m = np.copy(rw.rolling_window(w_pad, window=[3] * num_dim))
        z_m = z_m.reshape(z_m.shape[:num_dim] + (-1,))
        for nb in range(0, z_m.shape[-1]):
            case_1 = np.where((proc_ext == 1)
                              & (z >= z_m[tuple(s + [nb])] + eps), 1, 0).astype(np.int8)
            case_2 = np.where((proc_ext == 1) & (case_1 == 0) &
                              (w > z_m[tuple(s + [nb])] + eps), 1, 0).astype(np.int8)
            w_new = np.where(case_1 == 1, z, w)
            w_new = np.where(case_2 == 1, z_m[tuple(s + [nb])] + eps, w_new)

            if np.sum(np.abs(w - w_new)) > 0:
                smt_done = 1
                # print np.sum(np.abs(w - w_new))
            w = np.copy(w_new)
            w_pad = np.pad(w, pad_width=1, mode='constant', constant_values=z.min() - 1.)
            z_m = np.copy(rw.rolling_window(w_pad, window=[3] * num_dim))
            z_m = z_m.reshape(z_m.shape[:num_dim] + (-1,))
    return w


def flow_direction():
    middle_index = 3 ** num_dim // 2
    for nei in range(3 ** num_dim):
        mg['slope'][:, nei] = (mg['topgraphic__elevation'][mg['neighbour'][:, nei]] - \
                               mg['topgraphic__elevation'][mg['neighbour'][:, middle_index]]) \
                              / mg['distance'][0, nei]
        mg['slope'][:, nei] = np.where(mg['neighbour'][:, nei] == -1, np.nan, mg['slope'][:, nei])

    mg['receiver'] = mg['neighbour'][np.arange(len(mg['neighbour'])), np.nanargmin(mg['slope'], axis=1)]
    mg['receiver'] = np.where(np.nanmin(mg['slope'], axis=1) >= 0, -1, mg['receiver'])
    mg['receiver'] = np.where(mg['node_type'] == 0, -1, mg['receiver'])

    receiver_of_nei = mg['receiver'][mg['neighbour']]
    receiver_of_nei = np.where(mg['neighbour'] == -1, -1, receiver_of_nei)
    for nei in range(3 ** num_dim):
        mg['donor'][:, nei] = np.where((receiver_of_nei[:, nei] == mg['node_id']), mg['neighbour'][:, nei], -1)


def initial_z(nrows, ncols, domain_type, max_z):
    if domain_type == '4sided':
        z = np.zeros((2, ncols - nrows + 2))
        npad = tuple([1, 1])
        h = 1.
        while z.shape[-1] < ncols:
            z = np.pad(z, pad_width=npad, mode='constant', constant_values=h)
            h += 1.
        z = (1. - z / z.max()) * max_z
    elif domain_type == '2sided':
        z = np.zeros((nrows, ncols))
        for i in range(nrows):
            z[i, :] = 1. - 2. * np.abs(i - nrows / 2) / nrows
    return z



###########################################
###########################################

D_list = [0] + [10**(x * 0.2) for x in range(-25, 15)]


domain_size = (100, 100)
K, m, U, dx = 1., 0.5, 1., 1.
num_dim = 2

if m < 1:
    min_flage = True
else:
    min_flage = False


### Initiation #################
################################

mg = defaultdict()
mg['node_id'] = np.arange(np.prod(domain_size))

z_ini = initial_z(domain_size[0], domain_size[1], '4sided', 0.1) + np.random.normal(0, 0.0001, domain_size)
z_ini = np.where(z_ini < 0, 0, z_ini)
z_ini = fill(z_ini, num_dim, 3)
mg['topgraphic__elevation'] = z_ini.reshape(np.prod(domain_size))

mg['drainage_area'] = np.ones(np.prod(domain_size)).astype(np.float32)
mg['processed_node'] = np.zeros(np.prod(domain_size)).astype(np.int)
mg['receiver'] = np.zeros(np.prod(domain_size)).astype(np.int64)
mg['slope'] = np.zeros((np.prod(domain_size), 3 ** num_dim)).astype(np.float32)
mg['donor'] = np.zeros((np.prod(domain_size), 3 ** num_dim)).astype(np.int)

temp = np.ones(domain_size)
temp[0, :], temp[-1, :] = 0, 0
temp[:, 0], temp[:, -1] = 0, 0

mg['node_type'] = temp.reshape(np.prod(domain_size))
mg['active_node'] = list(mg['node_id'][mg['node_type'] == 1])

mg['neighbour'] = np.zeros((np.prod(domain_size), 3 ** num_dim)).astype(np.int)
mg['distance'] = np.zeros((1, 3 ** num_dim))

nodes_index = np.unravel_index(mg['node_id'], domain_size)
nei_index = np.zeros((np.prod(domain_size), 3 ** num_dim, num_dim)).astype(np.int)
for nei in range(3 ** num_dim):
    temp_index = np.subtract(np.unravel_index(nei, (3,) * num_dim), (1,) * num_dim)
    mg['distance'][0, nei] = np.sqrt(np.sum(temp_index ** 2))
    for dim in range(num_dim):
        temp = nodes_index[dim] + temp_index[dim]
        nei_index[:, nei, dim] = np.where((temp >= 0) & (temp < domain_size[dim]), temp, -1)
    temp = np.where(nei_index[:, nei, :] == -1, 0, nei_index[:, nei, :])
    mg['neighbour'][:, nei] = np.ravel_multi_index(np.transpose(temp), domain_size)
    mg['neighbour'][:, nei] = np.where(nei_index[:, nei, :].min(axis=1) == -1, -1,
                                       mg['neighbour'][:, nei])

flow_direction()
flow_accmulation()

### OCN #################
########################

if min_flage:
    H_opt = 10 ** 20
else:
    H_opt = 0.
t = 0
T = 100.
H_list = []
H_opt_list = []
log_slope = []
t_no_imp = 0
p = 1.

while t_no_imp <= 1000:

    mg_org = copy.deepcopy(mg)

    rand_node = random.sample(mg['active_node'], 1)[0]
    rec_list = list(mg['neighbour'][rand_node])
    for x in list(mg['donor'][rand_node]) + list([mg['receiver'][rand_node]]) + list([rand_node]):
        if x in rec_list:
            rec_list.remove(x)
    if len(rec_list) > 0:
        rand_rec = random.sample(rec_list, 1)[0]
    else:
        rand_rec = mg['receiver'][rand_node]
    pre_rec = mg['receiver'][rand_node]

    loop, bound = False, True
    p1 = np.unravel_index(rand_node, domain_size)
    p2 = np.unravel_index(rand_rec, domain_size)
    if p1[0] != p2[0] and p1[1] != p2[1]:
        nei_1 = (p1[0], p2[1])
        nei_2 = (p2[0], p1[1])
        nei_1 = np.ravel_multi_index(nei_1, domain_size)
        nei_2 = np.ravel_multi_index(nei_2, domain_size)
        a_temp = mg['drainage_area'].reshape(domain_size)
        if mg['receiver'][nei_1] == nei_2 or mg['receiver'][nei_2] == nei_1:
            loop = True

    H = np.nan
    if loop:
        mg = copy.deepcopy(mg_org)
    else:
        t += 1
        mg['receiver'][rand_node] = rand_rec

        doner_find()
        mg['drainage_area'] = np.ones(np.prod(domain_size)).astype(np.float32)
        flow_accmulation()

        if mg['drainage_area'][mg['node_type'] == 0].sum() == np.prod(domain_size) and \
                mg['processed_node'].sum() == len(mg['active_node']):

            H =  U / K * np.sum(mg['drainage_area'] ** (1 - m))


    if (min_flage is True and H < H_opt) or (min_flage is False and H > H_opt):
        H_opt = H
        t_no_imp = 0
    else:
        t_no_imp += 1
        mg = copy.deepcopy(mg_org)

    H_opt_list.append(H_opt)
    H_list.append(H)

