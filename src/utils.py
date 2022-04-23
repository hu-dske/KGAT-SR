import math
import time
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx

from kg import KGraph
from torch.utils.data import Dataset, DataLoader
import pickle

########################################## Evaluation #########################################
def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0


def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getMRR(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return 1.0 / (i + 1)
    return 0


def metrics(ranklist, targetItem):
    hr = getHitRatio(ranklist, targetItem)
    ndcg = getNDCG(ranklist, targetItem)
    mrr = getMRR(ranklist, targetItem)
    return hr, ndcg, mrr


######################################### Data Loader #########################################
class HistDataset(Dataset):
    def __init__(self, df, idx_list, attr_size, hist_max_len):
        self.data = df.values  # [user, item, timestamp]
        self.idx_list = idx_list
        self.MASK = 0
        self.attr_size = attr_size
        self.hist_max_len = hist_max_len

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        l, r, t = self.idx_list[item]
        submat = self.data[l:r]
        h_uid, h_items, h_attrs = submat.T
        uid, t_item, _ = self.data[t]
        assert np.all(h_uid == uid)

        h_attrs[-1] = [self.MASK] * self.attr_size
        h_attrs = np.array([i for i in h_attrs])
        n = len(h_items)
        if n < self.hist_max_len:
            h_items = np.pad(h_items, [0, self.hist_max_len - n], 'constant', constant_values=self.MASK)
            h_attrs = np.pad(h_attrs, [(0, self.hist_max_len - n), (0, 0)], 'constant', constant_values=self.MASK)
        return uid, np.array(list(h_items)).astype(np.long), h_attrs, t_item


class Loader():
    def __init__(self, args):
        self.dataset = args.dataset
        self.attr_size = args.attr_size
        self.neibor_size = args.neibor_size
        self.hist_min_len = args.hist_min_len
        self.hist_max_len = args.hist_max_len
        self.train_batch_size = args.batchSize
        self.eval_batch_size = args.batchSize
        self.n_workers = args.n_workers
        self.valid = args.valid
        self.n_neg = args.n_neg
        self.MASK = 0

        self.kg = KGraph(self.dataset, self.attr_size)
        self.node_neibors = self.kg.node_neibors
        self.n_entity, self.n_relation = self.kg.n_entity, self.kg.n_relation

        self.nodes_degree = self.kg.nodes_degree
        self.D_node = self.construct_D()
        self.adj_entity, self.adj_relation = self.construct_neibors_adj()
        self.all_items_list, self.train_dl, self.valid_dl = self.load_data()

    def construct_D(self):
        sorted_degree = sorted(self.nodes_degree.items(), key=lambda x: x[0])
        D_node = [i[0] for i in sorted_degree]
        return D_node

    def construct_neibors_adj(self):
        adj_entity = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)
        adj_relation = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)

        for node in range(self.n_entity):
            neighbors = self.node_neibors[node]
            n_neighbors = len(neighbors)
            # sample
            if n_neighbors >= self.neibor_size:
                sampled_indices = np.random.choice(neighbors, size=self.neibor_size, replace=False)
            else:
                sampled_indices = np.random.choice(neighbors, size=self.neibor_size, replace=True)

            adj_entity[node] = np.array([n for n in sampled_indices])
            adj_relation[node] = np.array([self.kg.G.get_edge_data(node, n)['rel'] for n in sampled_indices])
        return adj_entity, adj_relation

    def extract_subseq(self, n):
        idx_list = []
        for right in range(self.hist_min_len, n):
            left = max(0, right - self.hist_max_len)
            target = right
            idx_list.append([left, right, target])
        return np.asarray(idx_list)

    def get_idx(self, df):
        offset = 0
        train_idx_list = []
        valid_idx_list = []
        for n in df.groupby('user').size():
            train_idx_list.append(self.extract_subseq(n - 1) + offset)
            valid_idx_list.append(np.add([max(0, n - 1 - self.hist_max_len), n - 1, n - 1], offset))
            offset += n
        train_idx_list = np.concatenate(train_idx_list)
        valid_idx_list = np.stack(valid_idx_list)
        return train_idx_list, valid_idx_list

    def load_data(self):
        users, entities, attrs = list(), list(), list()
        # df = pd.read_csv(f'{self.dataset}data.tsv', header=None, names=['user', 'item', 'rating', 'timestamp'],sep='\t')
        train_data = pickle.load(open('../datasets/' + self.dataset + '/train.txt', 'rb'))

        df = pd.read_csv( train_data)
        # del df['rating']
        #df = df.sort_values(['user', 'timestamp'], ascending=[True, True])
        all_items_list = sorted(df['movieID'].unique().tolist())

        black_list = df.groupby('userID').apply(lambda subdf:
                                              [i for i in subdf.item.values]).to_dict()

        black_list_path = {u: self.kg.entity_seq_shortest_path(black_list[u])
                           for u in black_list}
        for (user, entity_list) in black_list_path.items():
            users += [user] * len(entity_list)
            entities += [e_a[0] for e_a in entity_list]
            attrs += [e_a[1] for e_a in entity_list]
        data = pd.DataFrame({'user': users, 'item': entities, 'attr': attrs})

        train_idx_list, valid_idx_list = self.get_idx(data)
        train_ds = HistDataset(data, train_idx_list, self.attr_size, self.hist_max_len)
        valid_ds = HistDataset(data, valid_idx_list, self.attr_size, self.hist_max_len)

        train_dl = DataLoader(train_ds, self.train_batch_size, pin_memory=True, shuffle=True, drop_last=True,
                              num_workers=self.n_workers)
        valid_dl = DataLoader(valid_ds, self.eval_batch_size, pin_memory=True, num_workers=self.n_workers)

        return all_items_list, train_dl, valid_dl


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            if self.method == 'ggnn':
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.zeros((max_n_node, max_n_node))
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    u_sum_in = np.sum(u_A, 0)
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)
                    u_sum_out = np.sum(u_A, 1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)

                    A_in.append(u_A_in)
                    A_out.append(u_A_out)
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]