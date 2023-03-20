import math
import numpy as np
import networkx as nx
import torch
import random
from texttable import Texttable
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
from torch_geometric.data import Data
# import matplotlib.pyplot as plt

import torch_scatter

import pdb
import pdb
import copy
from itertools import repeat

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'min', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0

    return out

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """

    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation
    
def _calculate_prec_at_k(k, target):
    target_increase = np.sort(target)
    target_value_sel = (target_increase <= target_increase[k-1]).sum()

    if target_value_sel > k:
        best_k_target = target.argsort()[:target_value_sel]
    else:
        best_k_target = target.argsort()[:k]
    return best_k_target


def calculate_prec_at_k(k, prediction, target, target_ged):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = _calculate_prec_at_k(k, -target)
    best_k_target_ged = _calculate_prec_at_k(k, target_ged)

    return len(set(best_k_pred).intersection(set(best_k_target_ged))) / k

def denormalize_sim_score(g1, g2, sim_score):
    """
    Converts normalized similar into ged.
    """
    return denormalize_ged(g1, g2, -math.log(sim_score, math.e))


def denormalize_ged(g1, g2, nged):
    """
    Converts normalized ged into ged.
    """
    return round(nged * (g1.num_nodes + g2.num_nodes) / 2)


def gen_synth_data(count=200, nl=None, nu=50, p=0.5, kl=None, ku=2):
    """
    Generating synthetic data based on Erdos–Renyi model.
    :param count: Number of graph pairs to generate.
    :param nl: Minimum number of nodes in a source graph.
    :param nu: Maximum number of nodes in a source graph.
    :param p: Probability of an edge.
    :param kl: Minimum number of insert/remove edge operations on a graph.
    :param ku: Maximum number of insert/remove edge operations on a graph.
    """
    if nl is None:
        nl = nu
    if kl is None:
        kl = ku
    
    data = []
    data_new = []
    mat = torch.full((count, count), float('inf'))
    norm_mat = torch.full((count, count), float('inf'))
    
    for i in range(count):
        n = random.randint(nl, nu)
        edge_index = erdos_renyi_graph(n, p)
        x = torch.ones(n, 1)
        
        g1 = Data(x=x, edge_index=edge_index, i=torch.tensor([i]))
        g2, ged = gen_pair(g1, kl, ku)

        data.append(g1)
        data_new.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g1.num_nodes + g2.num_nodes))
            
    return data, data_new, mat, norm_mat


def gen_pairs(graphs, kl=None, ku=2):
    gen_graphs_1 = []
    gen_graphs_2 = []

    count = len(graphs)
    mat = torch.full((count, count), float('inf'))
    norm_mat = torch.full((count, count), float('inf'))
    
    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))
    
    return gen_graphs_1, gen_graphs_2, mat, norm_mat


def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def gen_pair(g, kl=None, ku=2):
    if kl is None:
        kl = ku
    
    directed_edge_index = to_directed(g.edge_index)
    
    n = g.num_nodes
    num_edges = directed_edge_index.size(1)
    to_remove = random.randint(kl, ku)
    
    edge_index_n = directed_edge_index[:,torch.randperm(num_edges)[to_remove:]]
    if edge_index_n.size(1) != 0:
        edge_index_n = to_undirected(edge_index_n)
    
    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)
    adj[row, col] = 0
    non_edge_index = adj.nonzero().t()
    
    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size(1)

    to_add = random.randint(kl, ku)
    
    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)
    
    if hasattr(g, 'i'):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)
    
    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


def aids_labels(g):
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]
    
    return [types[i] for i in g.x.argmax(dim=1).tolist()]

def feature_augmentation(dataset, feature_aug_options):
    print('dataset', dataset)
    copy_dataset = copy.deepcopy(dataset)
    temp_dataset = []

    for idx, graph_item in enumerate(copy_dataset):
        edge_index = graph_item.edge_index
        size = graph_item.x.shape[0]
        node_degree = list(repeat(0, size))
        aug_feature_list = []
        queue = []
        visited = {}

        [graph_adj1, graph_adj2] = edge_index.numpy()

        w, h = size, size
        full_adj = [[0 for x in range(w)] for y in range(h)] 
        for idx in range (0, len(graph_adj1)):
            full_adj[graph_adj1[idx]][graph_adj2[idx]] = 1

        A = np.array(full_adj)

        # number of nodes in the graph
        # n = A.shape[0]

        for node_idx in range(0, size):
            temp_list = [0 for i in range(11) ]
            aug_feature_list.append(temp_list)
        
        
        # method 1: fast identity GIN
        if feature_aug_options == 1: # save if a route exitst
        # calculate the counts of length k up to a maximum length of max_k
            max_k = min(10, size)
            for k in range(3, max_k+1):
                Ak = np.linalg.matrix_power(A, k)

                for j in range(0, len(Ak)):
                    if(Ak[j][j] > 0):
                        aug_feature_list[j][k] = 1
        
        elif feature_aug_options == 2: # save the count of routes
            max_k = min(10, size)
            for k in range(3, max_k+1):
                Ak = np.linalg.matrix_power(A, k)

                for j in range(0, len(Ak)):
                    if(Ak[j][j] > 0):
                        aug_feature_list[j][k] = float(min(round(Ak[j][j] / (2*size), 2), 1))


        elif feature_aug_options == 3: # only count the number of triangles
            max_k = 3

            for k in range(3, max_k+1):
                Ak = np.linalg.matrix_power(A, k)
                for j in range(0, len(Ak)):
                    if(Ak[j][j] > 0):
                        Ak[j][j] = Ak[j][j] // 2 
                        count_triangle = min(Ak[j][j], 10)
                        # aug_feature_list[j][k] = round(Ak[j][j] / (2*size),2)
                        aug_feature_list[j][count_triangle] = 1

        aug_feature_list = torch.tensor(aug_feature_list)

        # only do feature augmentation when self.args.feature_aug > 0
        if feature_aug_options > 0: 
            graph_item.x = torch.cat((graph_item.x, aug_feature_list), 1)

        temp_dataset.append(graph_item)
    return temp_dataset

def draw_graphs(glist, aids=False):
    ...
    # for i, g in enumerate(glist):
    #     plt.clf()
    #     G = to_networkx(g).to_undirected()
    #     if aids:
    #         label_list = aids_labels(g)
    #         labels = {}
    #         for j, node in enumerate(G.nodes()):
    #             labels[node] = label_list[j]
    #         nx.draw(G, labels=labels)
    #     else:
    #         nx.draw(G)
    #     plt.savefig('graph{}.png'.format(i))


def draw_weighted_nodes(filename, g, model):
    ...
    # """
    # Draw graph with weighted nodes (for AIDS).
    # """
    # features = model.convolutional_pass(g.edge_index, g.x)
    # coefs = model.attention.get_coefs(features)
    
    # print(coefs)
    
    # plt.clf()
    # G = to_networkx(g).to_undirected()
    
    # label_list = aids_labels(g)
    # labels = {}
    # for i, node in enumerate(G.nodes()):
    #     labels[node] = label_list[i]
    
    # vmin = coefs.min().item() - 0.005
    # vmax = coefs.max().item() + 0.005

    # nx.draw(G, node_color=coefs.tolist(), cmap=plt.cm.Reds, labels=labels, vmin=vmin, vmax=vmax)

    # # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # # sm.set_array(coefs.tolist())
    # # cbar = plt.colorbar(sm)

    # plt.savefig(filename)
