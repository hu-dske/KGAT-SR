import numpy as np
import networkx as nx

class KGraph():
    def __init__(self, dataset, attr_size):
        self.dataset = dataset
        self.sample_attr_size = attr_size
        self.G, self.n_relation = self.get_kg()
        self.node_neibors = self.get_all_neibors()
        self.n_entity = self.G.number_of_nodes()
        self.nodes_degree = self.count_node_degree()
        self.Gadj = self.construct_Gadj()
        self.MASK = 0

    def get_kg(self):
        kg_path = '../datasets/'+ self.dataset + '/kg.txt'
        G = nx.Graph()
        all_rels = []
        with open(kg_path, 'r') as f:
            for line in f.readlines():
                # if line.strip() == '':
                #     continue

                arr = line.strip().split('\t')
                # arr = line.strip().split(' ')
                head, rel, tail = int(arr[0]), arr[1], int(arr[2])
                G.add_edge(head, tail, rel=rel)
                if rel not in all_rels:
                    all_rels.append(rel)
        return G, len(all_rels)

    def get_all_neibors(self):
        node_neibors = dict()
        for node in self.G.nodes():
            node_neibors[node] = [j for j in nx.all_neighbors(self.G, node)]
            # node_neibors[node] = [(j, self.G.get_edge_data(node,j)['rel'])
            #                     for j in nx.all_neighbors(self.G, node)]
        return node_neibors

    def count_node_degree(self):
        nodes_degree = dict()
        for node in self.G.nodes():
            nodes_degree[node] = self.G.degree(node)
        return nodes_degree

    def construct_Gadj(self):
        Gadj = dict(self.G.adj)
        for n in Gadj:
            adj_nodes = Gadj[n]
            adj_nodes = sorted(adj_nodes.items(), key=lambda x: self.nodes_degree[x[0]])
            adj_nodes = {k: v for (k, v) in adj_nodes}
            Gadj[n] = adj_nodes
        return Gadj

    def bidirectional_attribute(self, source, target):
        if source not in self.G or target not in self.G:
            msg = 'Either source {} or target {} is not in G'
            raise nx.NodeNotFound(msg.format(source, target))

        if target == source:
            return (source, sample_attr_nodes)

        inter_attrs = list(set(self.Gadj[source]).intersection(set(self.Gadj[target])))

        # sort attribute nodes by degrees
        if len(inter_attrs):
            inter_attrs = sorted(inter_attrs, key=lambda x: self.nodes_degree[x])
        else:
            inter_attrs = [self.MASK]

        n_inter_attrs = len(inter_attrs)
        if self.sample_attr_size >= n_inter_attrs:
            sample_attr_nodes = np.random.choice(inter_attrs, size=self.sample_attr_size, replace=True)
        else:
            # sample_attr_nodes = np.random.choice(inter_attrs, size = self.sample_attr_size, replace=False)
            sample_attr_nodes = inter_attrs[:self.sample_attr_size]

        return (source, sample_attr_nodes)

    def entity_seq_shortest_path(self, entity_seq):
        entity_attrs_path = list()
        for i in range(len(entity_seq) - 1):
            e1 = entity_seq[i]
            e2 = entity_seq[i + 1]
            attr_e1_e2 = self.bidirectional_attribute(e1, e2)
            entity_attrs_path.append(attr_e1_e2)
        entity_attrs_path.append((entity_seq[-1], [self.MASK] * self.sample_attr_size))
        return entity_attrs_path
