import copy
import random
from collections import defaultdict

import numpy as np

import Utils

import matplotlib.pylab as plt


class OCN:
    def __init__(self, domain_size, domain_type,
                 initial_max_elevation, initial_std, m_exponent):
        self.mg = defaultdict()
        self.domain_size = domain_size
        self.domain_type = domain_type
        self.initial_max_elevation = initial_max_elevation
        self.initial_std = initial_std
        self.num_dim = len(self.domain_size)
        self.m = m_exponent
        self.H_list = []
        self.H_opt_list = []
        self.Porp_head = []

    def initiate_elevation(self):
        """

        Returns an initial elevation field
        -------

        """
        nrows, ncols = self.domain_size[0], self.domain_size[1]
        if self.domain_type == '4sided':
            z = np.zeros((2, ncols - nrows + 2))
            npad = tuple([1, 1])
            h = 1.
            while z.shape[-1] < ncols:
                z = np.pad(z, pad_width=npad, mode='constant', constant_values=h)
                h += 1.
            z = (1. - z / z.max()) * self.initial_max_elevation
        elif self.domain_type == '2sided':
            z = np.zeros((nrows, ncols))
            for i in range(nrows):
                z[i, :] = 1. - 2. * np.abs(i - nrows / 2) / nrows
        return z

    def initiate(self):
        """
        Creates The initial random domain
        -------

        """
        self.mg['node_id'] = np.arange(np.prod(self.domain_size))
        z_ini = self.initiate_elevation() + np.random.normal(0, self.initial_std, self.domain_size)
        z_ini = np.where(z_ini < 0, 0, z_ini)
        z_ini = self.fill(z_ini, window_size=3)
        self.mg['topographic__elevation'] = z_ini.reshape(np.prod(self.domain_size))

        self.mg['drainage_area'] = np.ones(np.prod(self.domain_size)).astype(np.float32)
        self.mg['processed_node'] = np.zeros(np.prod(self.domain_size)).astype(np.int)
        self.mg['receiver'] = np.zeros(np.prod(self.domain_size)).astype(np.int64)
        self.mg['slope'] = np.zeros((np.prod(self.domain_size), 3 ** self.num_dim)).astype(np.float32)
        self.mg['donor'] = np.zeros((np.prod(self.domain_size), 3 ** self.num_dim)).astype(np.int)

        temp = np.ones(self.domain_size)
        temp[0, :], temp[-1, :] = 0, 0
        temp[:, 0], temp[:, -1] = 0, 0

        self.mg['node_type'] = temp.reshape(np.prod(self.domain_size))
        self.mg['active_node'] = list(self.mg['node_id'][self.mg['node_type'] == 1])

        self.mg['neighbour'] = np.zeros((np.prod(self.domain_size), 3 ** self.num_dim)).astype(np.int)
        self.mg['distance'] = np.zeros((1, 3 ** self.num_dim))

        nodes_index = np.unravel_index(self.mg['node_id'], self.domain_size)
        nei_index = np.zeros((np.prod(self.domain_size), 3 ** self.num_dim, self.num_dim)).astype(np.int)
        for nei in range(3 ** self.num_dim):
            temp_index = np.subtract(np.unravel_index(nei, (3,) * self.num_dim), (1,) * self.num_dim)
            self.mg['distance'][0, nei] = np.sqrt(np.sum(temp_index ** 2))
            for dim in range(self.num_dim):
                temp = nodes_index[dim] + temp_index[dim]
                nei_index[:, nei, dim] = np.where((temp >= 0) & (temp < self.domain_size[dim]), temp, -1)
            temp = np.where(nei_index[:, nei, :] == -1, 0, nei_index[:, nei, :])
            self.mg['neighbour'][:, nei] = np.ravel_multi_index(np.transpose(temp), self.domain_size)
            self.mg['neighbour'][:, nei] = np.where(nei_index[:, nei, :].min(axis=1) == -1, -1,
                                                    self.mg['neighbour'][:, nei])

        self.flow_direction()
        self.flow_accumulation()

    def flow_acc_rec(self, sin):
        for don in self.mg['donor'][sin, self.mg['donor'][sin, :] >= 0]:
            self.mg['processed_node'][don] = 1
            self.flow_acc_rec(don)
            self.mg['drainage_area'][sin] += self.mg['drainage_area'][don]

    def flow_accumulation(self):
        """

        FLow accumulation from D8 flow direction
        -------

        """
        sinks = self.mg['node_id'][self.mg['node_type'] == 0]
        for sin in sinks:
            self.flow_acc_rec(sin)

    def flow_direction(self):
        """

        Builds the D8 flow direction from elevation
        -------

        """

        middle_index = 3 ** self.num_dim // 2
        for nei in range(3 ** self.num_dim):
            self.mg['slope'][:, nei] = (self.mg['topographic__elevation'][self.mg['neighbour'][:, nei]] - \
                                        self.mg['topographic__elevation'][self.mg['neighbour'][:, middle_index]]) \
                                       / self.mg['distance'][0, nei]
            self.mg['slope'][:, nei] = np.where(self.mg['neighbour'][:, nei] == -1, np.nan, self.mg['slope'][:, nei])

        self.mg['receiver'] = self.mg['neighbour'][np.arange(len(self.mg['neighbour'])),
                                                   np.nanargmin(self.mg['slope'], axis=1)]
        self.mg['receiver'] = np.where(np.nanmin(self.mg['slope'], axis=1) >= 0, -1, self.mg['receiver'])
        self.mg['receiver'] = np.where(self.mg['node_type'] == 0, -1, self.mg['receiver'])

        receiver_of_nei = self.mg['receiver'][self.mg['neighbour']]
        receiver_of_nei = np.where(self.mg['neighbour'] == -1, -1, receiver_of_nei)
        for nei in range(3 ** self.num_dim):
            self.mg['donor'][:, nei] = np.where((receiver_of_nei[:, nei] == self.mg['node_id']),
                                                self.mg['neighbour'][:, nei], -1)

    def fill(self, z, window_size=3):
        """
        Parameters
        ----------
        window_size: The size of moving window size

        Returns : Filled elevation filed
        -------

        """
        border = np.pad(np.zeros(tuple(np.asanyarray(z.shape) - 2)), pad_width=1, mode='constant', constant_values=1)
        w = np.where(border == 0, np.max(z) + 0.01, z)
        eps = 0.001
        smt_done = 1
        s = []
        for dim in range(self.num_dim):
            s.append(slice(0, z.shape[dim]))

        while smt_done == 1:
            smt_done = 0
            proc_ext = np.where((w > z) & (border == 0), 1, 0).astype(np.int8)

            w_pad = np.pad(w, pad_width=1, mode='constant', constant_values=z.min() - 1.)
            z_m = np.copy(Utils.rolling_window(w_pad, window=[3] * self.num_dim))
            z_m = z_m.reshape(z_m.shape[:self.num_dim] + (-1,))
            for nb in range(0, z_m.shape[-1]):
                case_1 = np.where((proc_ext == 1)
                                  & (z >= z_m[tuple(s + [nb])] + eps), 1, 0).astype(np.int8)
                case_2 = np.where((proc_ext == 1) & (case_1 == 0) &
                                  (w > z_m[tuple(s + [nb])] + eps), 1, 0).astype(np.int8)
                w_new = np.where(case_1 == 1, z, w)
                w_new = np.where(case_2 == 1, z_m[tuple(s + [nb])] + eps, w_new)

                if np.sum(np.abs(w - w_new)) > 0:
                    smt_done = 1
                w = np.copy(w_new)
                w_pad = np.pad(w, pad_width=1, mode='constant', constant_values=z.min() - 1.)
                z_m = np.copy(Utils.rolling_window(w_pad, window=[3] * self.num_dim))
                z_m = z_m.reshape(z_m.shape[:self.num_dim] + (-1,))
        return w

    def elevation_cal(self):
        """

        Builds the elevation field
        -------

        """

        sinks = list(self.mg['node_id'][self.mg['node_type'] == 0])
        for sin in sinks:
            self.mg['topographic__elevation'][sin] = 0

        while len(sinks) > 0:
            sin = sinks.pop(0)
            for don in self.mg['donor'][sin]:
                if don >= 0:
                    slope = self.mg['drainage_area'][don] ** (-self.m)
                    self.mg['topographic__elevation'][don] = slope + self.mg['topographic__elevation'][sin]
                    sinks.append(don)

    def donor_find(self):
        """
        Finds the list of donors to each cell
        -------

        """
        receiver_of_nei = self.mg['receiver'][self.mg['neighbour']]
        receiver_of_nei = np.where(self.mg['neighbour'] == -1, -1, receiver_of_nei)
        for nei in range(3 ** self.num_dim):
            self.mg['donor'][:, nei] = np.where((receiver_of_nei[:, nei] == self.mg['node_id']),
                                                self.mg['neighbour'][:, nei], -1)

    def optimize(self, max_itr_no_imp):
        min_flag = False
        if self.m < 1: min_flag = True

        H_opt = 0.
        if min_flag: H_opt = 10 ** 20

        t = 0
        self.H_list = []
        self.H_opt_list = []
        t_no_imp = 0

        while t_no_imp <= max_itr_no_imp:
            mg_org = copy.deepcopy(self.mg)
            rand_node = random.sample(self.mg['active_node'], 1)[0]
            rec_list = list(self.mg['neighbour'][rand_node])
            for x in list(self.mg['donor'][rand_node]) + list([self.mg['receiver'][rand_node]]) + list([rand_node]):
                if x in rec_list:
                    rec_list.remove(x)
            if len(rec_list) > 0:
                rand_rec = random.sample(rec_list, 1)[0]
            else:
                rand_rec = self.mg['receiver'][rand_node]
            pre_rec = self.mg['receiver'][rand_node]

            loop, bound = False, True
            p1 = np.unravel_index(rand_node, self.domain_size)
            p2 = np.unravel_index(rand_rec, self.domain_size)
            if p1[0] != p2[0] and p1[1] != p2[1]:
                nei_1 = (p1[0], p2[1])
                nei_2 = (p2[0], p1[1])
                nei_1 = np.ravel_multi_index(nei_1, self.domain_size)
                nei_2 = np.ravel_multi_index(nei_2, self.domain_size)
                a_temp = self.mg['drainage_area'].reshape(self.domain_size)
                if self.mg['receiver'][nei_1] == nei_2 or self.mg['receiver'][nei_2] == nei_1:
                    loop = True

            H = np.nan
            if loop:
                self.mg = copy.deepcopy(mg_org)
            else:
                t += 1
                self.mg['receiver'][rand_node] = rand_rec

                self.donor_find()
                self.mg['drainage_area'] = np.ones(np.prod(self.domain_size)).astype(np.float32)
                self.flow_accumulation()

                if self.mg['drainage_area'][self.mg['node_type'] == 0].sum() == np.prod(self.domain_size) and \
                        self.mg['processed_node'].sum() == len(self.mg['active_node']):
                    H = np.sum(self.mg['drainage_area'] ** (1 - self.m))

            if (min_flag is True and H < H_opt) or (min_flag is False and H > H_opt):
                H_opt = H
                t_no_imp = 0
                ph = np.where(self.mg['drainage_area'] == 1, 1, 0).sum() / len(self.mg['drainage_area'])
            else:
                t_no_imp += 1
                self.mg = copy.deepcopy(mg_org)

            self.H_opt_list.append(H_opt)
            self.H_list.append(H)
            self.Porp_head.append(ph)
            if t % 100 == 0:
                ev = self.local_network()
                most_comm_ev = ev[0][0]
                num_evs = len(ev)
                frq = ev[0][1] / len(self.mg['node_id'])

                print('Trail: ', t, 'Obj: ', int(H_opt), 'HR: ', int(ph * 100.), )
                print('MCEV: ', most_comm_ev, 'NEV: ', num_evs, 'FEV: ', int(frq * 100))

    def plot_results(self):
        plt.figure(figsize=(10, 6))

        ################
        plt.subplot(221)
        plt.title(' Log of area')
        plt.imshow(np.log(self.mg['drainage_area'].reshape(self.domain_size)))
        plt.xlabel('X')
        plt.ylabel('Y')

        ################
        plt.subplot(222)
        plt.title('Optimization')
        plt.plot(self.H_opt_list, '-k', lw=1., label='Optimal')
        plt.plot(self.H_list, '-r', lw=0.1, alpha=1., label='Tried')
        plt.xlabel('Trail')
        plt.ylabel('Objective function')
        plt.legend()

        ################
        plt.subplot(223)
        plt.title('Optimization')
        plt.plot(self.Porp_head, '-k', lw=1., label='Optimal')
        plt.xlabel('Trail')
        plt.ylabel('Head ratio')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def local_network(self):
        eigen_count = {}
        for node in self.mg['node_id']:
            adj_metrix = np.zeros((9, 9))
            neis = self.mg['neighbour'][node]
            for i, n_s in enumerate(neis):
                adj_metrix[i, i] = 1
                for j, n_d in enumerate(neis):
                    if n_s == self.mg['receiver'][n_d]:
                        adj_metrix[i, j], adj_metrix[j, i] = 1, 1

            w = np.linalg.eigvalsh(adj_metrix)
            w = sorted(list(set(np.round(w, 2))))
            w = ','.join([str(x) for x in w])
            eigen_count[w] = eigen_count.get(w, 0) + 1

        eigen_count = [(ky, vl) for ky, vl in eigen_count.items()]
        eigen_count = sorted(eigen_count, key=lambda x: -x[1])

        return eigen_count
