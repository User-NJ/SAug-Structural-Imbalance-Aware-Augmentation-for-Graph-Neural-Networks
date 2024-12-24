import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from torch.nn.utils import prune

# from layers import GCNConv, SAGEConv
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import subgraph, index_to_mask

import config
import torch


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, num_classes=2, multilabel=False):
        super(Net, self).__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = config.hidden_dim
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.nei_layer = nn.Sequential(
            nn.Linear(self.in_channels, config.drop_hidden),
            nn.ReLU(),
            nn.Linear(config.drop_hidden, config.drop_hidden),
            nn.ReLU()
        )
        self.self_layer = nn.Sequential(
            nn.Linear(self.in_channels, config.drop_hidden),
            nn.ReLU(),
            nn.Linear(config.drop_hidden, config.drop_hidden),
            nn.ReLU()
        )
        self.dense_layer = nn.Sequential(
            nn.Linear(config.drop_hidden * 2, 1),
            nn.ReLU()
        )
        if config.base_model == 'GCN':
            self.conv1 = GCNConv(in_channels, self.out_channels) # 第一个卷积层
            #self.conv2 = GCNConv(self.hidden_dim, out_channels) # 第二个卷积层
            self.conv4classify = GCNConv(self.out_channels, self.num_classes) # 第三个卷积层也就是分类层
        elif config.base_model == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, self.hidden_dim)
            self.conv2 = SAGEConv(self.hidden_dim, out_channels)
            self.conv4classify = SAGEConv(self.out_channels, self.num_classes)

        elif config.base_model == 'GAT':
            self.heads = config.heads # heads参数是GAT独有的，是用来设置多头注意力的
            self.conv1 = GATConv(in_channels, self.hidden_dim, heads=self.heads, concat=False)
            self.conv2 = GATConv(self.hidden_dim, out_channels, heads=self.heads, concat=False)
            self.conv4classify = GATConv(self.out_channels, self.num_classes)

        else:
            raise "base model not used!"

    def encode(self, x, edge_index):# 编码层获得节点的embedding
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index): # 解码层，只用来进行边预测
        dot = (z[edge_label_index[0]] * z[edge_label_index[1]])
        inner_dot = dot.sum(dim=-1)
        return inner_dot # 表示边存在的概率

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t() # 传播链接预测的邻接矩阵

    def node_classify(self, data, drop_edges=False):
        x, adj = data.x, data.edge_index
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv2(x, adj)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv4classify(x, adj) # 节点分类任务器
        if self.multilabel:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.sigmoid(x)



class Gen(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape, non_hub):
        super(Gen, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.non_hub = non_hub

        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_flat = nn.Linear(256, self.num_pred * self.feat_shape)

        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, feat_shape):
        super(Discriminator, self).__init__()
        self.feat_shape = feat_shape

        self.fc1 = nn.Linear(feat_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Add_Neighbor(nn.Module):
    def __init__(self, data, tails):
        super(Add_Neighbor, self).__init__()
        self.data = data
        self.tails = tails
        self.origin_data = data.clone()

    def forward(self, gen_feat):
        if isinstance(gen_feat, dict):
            edge_1 = []
            genfeat = []
            for i in gen_feat.keys():
                if gen_feat[i].size(0) == self.data.x.size(1):
                    edge_1.append(i)
                    genfeat.append(gen_feat[i].cpu().detach().numpy().tolist())
                else:
                    for j in range(gen_feat[i].size(0)):
                        edge_1.append(i)
                        genfeat.append(gen_feat[i][j].cpu().detach().numpy().tolist())
            edge_1 = np.array(edge_1, dtype=int)
            edge_2 = np.array([i for i in range(self.data.num_nodes, self.data.num_nodes + len(edge_1))])
            edge_gen = np.vstack([edge_1, edge_2])
            new_edge = torch.tensor(edge_gen, dtype=torch.int, device=config.device)
            edges = torch.hstack([self.data.edge_index, new_edge])
            genfeat = np.array(genfeat)
            genfeat = torch.tensor(genfeat, device=config.device)
            new_feat = torch.vstack([self.data.x, genfeat]).to(torch.float)
            return new_feat, edges
        else:
            edge_1 = []
            for i in range(len(self.tails)):
                for j in range(config.num_pred):
                    edge_1.append(self.tails[i])
            edge_1 = np.array(edge_1, dtype=int)
            edge_2 = np.array([i for i in range(self.data.num_nodes, self.data.num_nodes +
                                                len(self.tails) * config.num_pred)])
            edge_generate = np.vstack((edge_1, edge_2))

            new_edge = torch.tensor(edge_generate, dtype=torch.int, device=config.device)
            # self.data.edge_label_index = torch.hstack((self.data.edge_label_index, new_edge))
            new_edge = torch.hstack((self.data.edge_index, new_edge))

            new_feat = torch.vstack((self.data.x, gen_feat.view(-1, self.data.x.size(1)).detach()))
            new_edge_label = np.ones(len(self.tails) * config.num_pred, dtype=int)
            new_edge_label = torch.tensor(new_edge_label, device=config.device)
            # self.data.edge_label = torch.cat([self.data.edge_label, new_edge_label])
            self.data = self.origin_data.clone()

            return new_feat, new_edge


class NeighGen(nn.Module):
    def __init__(self, feat_shape, data, non_hub_node, neifeat):
        super(NeighGen, self).__init__()
        self.data = data
        self.non_hub_node = non_hub_node
        self.neifeat = neifeat
        self.genfeat = {}

        self.gen = Gen(latent_dim=feat_shape,
                       dropout=config.dropout,
                       num_pred=config.num_pred,
                       feat_shape=feat_shape,
                       non_hub=non_hub_node)

        self.classifier = GNN(nfeat=feat_shape,
                              nhid=config.hidden_dim,
                              nclass=1,
                              dropout=config.dropout)

        self.addneigh = Add_Neighbor(self.data, self.non_hub_node)

    def forward(self, feat):
        if isinstance(feat, dict):
            for i in feat.keys():
                gen_feat = self.gen(feat[i])
                self.genfeat[i] = gen_feat
            x, adj = self.addneigh(self.genfeat)
            class_pred = self.classifier(x, adj)
            return self.genfeat, class_pred.view(-1)

        else:
            gen_feat = self.gen(feat)
            x, adj = self.addneigh(gen_feat)
            class_pred = self.classifier(x, adj)
            return gen_feat, class_pred
