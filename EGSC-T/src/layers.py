import torch
import torch.nn.functional as F
import torch.nn as nn

from math import ceil
from torch.nn import Linear, ReLU
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, DenseGINConv, dense_diff_pool, JumpingKnowledge
#from torch_geometric.utils import scatter_
from utils import scatter_

import pdb

def tensor_match(src,tar):

    (x,y) = (src, tar) if src.shape[0] <= tar.shape[0] else (tar, src)

    size_x = x.shape[0]
    size_y = y.shape[0]

    joint_tensor = torch.zeros(size_x*size_y, x.shape[1] + y.shape[1]) # 72 * 64, e.g.

    for i in range(size_x):
        x_reap = x[i,:].repeat(size_y,1) # 1 * 32 -> 8 * 32
        joint_tensor[ i * size_y : (i+1) * size_y, 0:x.shape[1]] = x_reap
    y_reap = y.repeat(size_x,1) # 1 * 32 -> 8 * 32
    joint_tensor[:,x.shape[1]:] = y_reap

    return joint_tensor



class ConfusionAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        """
        :param args: Arguments object.
        """
        super(ConfusionAttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim_size*2
        reduction = 8
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, 1),
                        nn.Sigmoid()
                )
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)


    def forward(self, x_src, batch_src, x_tar, batch_tar, size=None):
        size = batch_src[-1] + 1 if batch_src[-1] == batch_tar[-1] else min(batch_src[-1], batch_tar[-1]) + 1
        score_batch = torch.zeros(size,1) # 128 * 1
        for i in range(size):
            feat_src_batch = x_src[batch_src == i,:]
            feat_tar_batch = x_tar[batch_tar == i,:]

            x_joint = torch.mm(feat_src_batch, feat_tar_batch.t()).view(-1) # on dim = 0 (shu zhe)
            
            score_batch[i,:] = x_joint.mean() # weights.mean()
        
        return score_batch # 128 * 1


class SEAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        super(SEAttentionModule, self).__init__()
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
        
        return scatter_('add', weighted, batch, dim_size=size)

class AttentionModule(torch.nn.Module):
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
        weighted = coefs.unsqueeze(-1) * x 

        return scatter_('add', weighted, batch, dim_size=size) # 128 * 16
        
    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class DenseAttentionModule(torch.nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(DenseAttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, mask=None):
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

class SETensorNetworkModule(torch.nn.Module):
    def __init__(self,args, dim_size):
        super(SETensorNetworkModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

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
        se_feat = se_feat_coefs * combined_representation + combined_representation
        scores = self.fc1(se_feat)

        return scores

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self,args, dim_size,):
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

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size, self.dim_size // 2))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.dim_size // 2, self.dim_size *2))
        self.bias = torch.nn.Parameter(torch.Tensor(self.dim_size//2, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
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
