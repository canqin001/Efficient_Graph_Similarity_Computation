import torch
import torch.nn.functional as F
import torch.nn as nn

from math import ceil
from torch.nn import Linear, ReLU
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, DenseGINConv, dense_diff_pool, JumpingKnowledge
from utils import scatter_

import pdb
import random

def repeat_certain_graph( edge_index_1, features_1, batch_1):
        pick_i = random.randint(0, batch_1[-1] )
        batch_cell = batch_1[batch_1==pick_i]
        features_cell = features_1[batch_1==pick_i,:]
        
        index_cell = (batch_1==pick_i).nonzero().squeeze()
        edge_index_cell = torch.cat(( edge_index_1[0,(edge_index_1[0,:]>=index_cell[0])&(edge_index_1[0,:]<=index_cell[-1])].unsqueeze(0), \
             edge_index_1[1,(edge_index_1[1,:]>=index_cell[0])&(edge_index_1[0,:]<=index_cell[-1])].unsqueeze(0)),dim=0)
        reap_num = batch_1[-1] + 1
        batch = batch_cell.repeat(reap_num)
        batch_cell_range = torch.linspace(0,reap_num.item(), steps=reap_num.item(), dtype = batch_cell.dtype)
        for num_k in range(0, batch_cell.shape[0]):
            index_k = batch_cell_range * batch_cell.shape[0] + num_k
            batch[index_k] = batch_cell_range
        
        edge_index_cell = edge_index_cell - edge_index_cell.min()
        edge_index = edge_index_cell.repeat(1, reap_num)
        edge_index_range = batch_cell_range * (edge_index_cell.max() + 1)

        for num_k in range(0, edge_index_cell.shape[1]):
            index_k = batch_cell_range * edge_index_cell.shape[1] + num_k
            edge_index[:,index_k] = torch.cat(((edge_index_cell[0,num_k]+edge_index_range).unsqueeze(0), 
            (edge_index_cell[1,num_k]+edge_index_range).unsqueeze(0)),dim=0)
        features = features_cell.repeat(reap_num, 1)
        return edge_index, features, batch

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1) # e -> 128 * 16, e_square -> 128 * 0
    prod = e @ e.t() # prod -> 128 * 128
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps) # res -> 128 * 128
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0 # let diag elements become 0
    return res

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d #/ mean_td
        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d
        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

def tensor_match(src,tar):
        (x,y) = (src, tar) if src.shape[0] <= tar.shape[0] else (tar, src)
        size_x = x.shape[0]
        size_y = y.shape[0]
        joint_tensor = torch.zeros(size_x*size_y, x.shape[1] + y.shape[1])
        for i in range(size_x):
            x_reap = x[i,:].repeat(size_y,1)
            joint_tensor[ i * size_y : (i+1) * size_y, 0:x.shape[1]] = x_reap
        y_reap = y.repeat(size_x,1)
        joint_tensor[:,x.shape[1]:] = y_reap
        return joint_tensor


class SEAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        """
        :param args: Arguments object.
        """
        super(SEAttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        """
        Defining weights.
        """
        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

    def forward(self, x):
        x = self.fc(x)
        return x


class AttentionModule_w_SE(torch.nn.Module):
    def __init__(self, args, dim_size):
        super(AttentionModule_w_SE, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

    def forward(self, x, batch, size=None):
        size = batch[-1].item() + 1 if size is None else size # size is the quantity of batches: 128 eg
        mean = scatter_('mean', x, batch, dim_size=size) # dim of mean: 128 * 16
        transformed_global = self.fc(mean)
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1)) # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x # weighted: 1128 * 16
        
        return scatter_('add', weighted, batch, dim_size=size) # 128 * 16

class AttentionModule_fix(torch.nn.Module):
    def __init__(self, args, dim_size):
        super(AttentionModule_fix, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size)) 
        self.weight_matrix1 = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size))
        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Tanh()
                )

        self.fc1 =  nn.Linear(channel,  channel)
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        attention = self.fc(x)
        x = attention * x + x

        size = batch[-1].item() + 1 if size is None else size # size is the quantity of batches: 128 eg
        mean = scatter_('mean', x, batch, dim_size=size) # dim of mean: 128 * 16
        transformed_global = \
        torch.tanh(torch.mm(mean, self.weight_matrix)) 
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1)) # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x #+ coefs_se.unsqueeze(-1) * x # weighted: 1128 * 16
        return scatter_('add', weighted, batch, dim_size=size) # 128 * 16

class SETensorNetworkModule(torch.nn.Module):
    def __init__(self,args, dim_size):

        super(SETensorNetworkModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        # self.init_parameters()

    def setup_weights(self):
        channel = self.dim_size*2
        reduction = 4
        self.fc_se = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

        self.fc0 = nn.Sequential(
                        nn.Linear(channel,  channel),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel, channel),
                        nn.ReLU(inplace = True)
                )

        self.fc1 = nn.Sequential(
                        nn.Linear(channel,  channel),
                        nn.ReLU(inplace = True),
                         nn.Linear(channel, self.dim_size // 2), #nn.Linear(channel, self.args.tensor_neurons),
                        nn.ReLU(inplace = True)
                )
    def forward(self, embedding_1, embedding_2):
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        se_feat_coefs = self.fc_se(combined_representation)
        se_feat = se_feat_coefs * combined_representation + combined_representation #+ self.fc0(combined_representation)
        scores = self.fc1(se_feat)
        # scoring: 128 * 16, block_scoring: 128 * 16, self.bias: 16 * 1, scores: 128 * 16
        return scores

class ConfusionAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        """
        :param args: Arguments object.
        """
        super(ConfusionAttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        #self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        channel = self.dim_size*2
        reduction = 8
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, 1),
                        nn.Sigmoid()
                )
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)


    def forward(self, x_src, batch_src, x_tar, batch_tar, size=None):
        size = batch_src[-1] + 1 if batch_src[-1] == batch_tar[-1] else min(batch_src[-1], batch_tar[-1]) + 1
        score_batch = torch.zeros(size,1) # 128 * 1
        for i in range(size):
            feat_src_batch = x_src[batch_src == i,:]
            feat_tar_batch = x_tar[batch_tar == i,:]
            x_joint = torch.mm(feat_src_batch, feat_tar_batch.t()).view(-1)
            score_batch[i,:] = x_joint.mean()

        return score_batch # 128 * 1

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args, dim_size):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size)) 
        self.weight_matrix1 = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size))
        #self.weight_matrix2 = torch.nn.Parameter(torch.Tensor(int(self.dim_size), self.dim_size)) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix1)
        #torch.nn.init.xavier_uniform_(self.weight_matrix2)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param size: Dimension size for scatter_
        :param batch: Batch vector, which assigns each node to a specific example, Dim: 1128 * 0; 0...1...2...
        :return representation: A graph level representation matrix. 
        """

        size = batch[-1].item() + 1 if size is None else size # size is the quantity of batches: 128 eg
        mean = scatter_('mean', x, batch, dim_size=size) # dim of mean: 128 * 16
        
        transformed_global = \
        torch.tanh(torch.mm(mean, self.weight_matrix)) 
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1)) # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x # weighted: 1128 * 16
        
        return scatter_('add', weighted, batch, dim_size=size) # 128 * 16
        
    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class DenseAttentionModule(torch.nn.Module):
    """
    SimGNN Dense Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(DenseAttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param mask: Mask matrix indicating the valid nodes for each graph. 
        :return representation: A graph level representation matrix. 
        """
        B, N, _ = x.size()
        
        if mask is not None:
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            mean = x.sum(dim=1)/num_nodes.to(x.dtype)
        else:
            mean = x.mean(dim=1)
        
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))
        
        koefs = torch.sigmoid(torch.matmul(x, transformed_global.unsqueeze(-1)))
        weighted = koefs * x
        
        if mask is not None:
            weighted = weighted * mask.view(B, N, 1).to(x.dtype)
        
        return weighted.sum(dim=1)


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self,args, dim_size):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, self.dim_size *2))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1) # 128
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.dim_size, -1)) 
        scoring = scoring.view(batch_size, self.dim_size, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.dim_size, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super(Block, self).__init__()
        
        nn1 = torch.nn.Sequential(
                Linear(in_channels, hidden_channels), 
                ReLU(), 
                Linear(hidden_channels, hidden_channels))
        
        nn2 = torch.nn.Sequential(
                Linear(hidden_channels, out_channels), 
                ReLU(), 
                Linear(out_channels, out_channels))
        
        self.conv1 = DenseGINConv(nn1, train_eps=True)
        self.conv2 = DenseGINConv(nn2, train_eps=True)
        
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(self.jump([x1, x2]))


class DiffPool(torch.nn.Module):
    def __init__(self, args, num_nodes=10, num_layers=4, hidden=16, ratio=0.25):
        super(DiffPool, self).__init__()
        
        self.args = args
        num_features = self.args.filters_3
        
        self.att = DenseAttentionModule(self.args)
        
        num_nodes = ceil(ratio * num_nodes)
        self.embed_block1 = Block(num_features, hidden, hidden)
        self.pool_block1 = Block(num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, num_features)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for block1, block2 in zip(self.embed_blocks, self.pool_blocks):
            block1.reset_parameters()
            block2.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj, mask):
        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))

        xs = [self.att(x, mask)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)

        for i, (embed, pool) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool(x, adj)
            x = F.relu(embed(x, adj))
            xs.append(self.att(x))
            if i < (len(self.embed_blocks) - 1):
                x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
