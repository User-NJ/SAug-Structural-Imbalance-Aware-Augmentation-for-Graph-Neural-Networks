import json
import math
import os.path as osp
from collections import defaultdict
from copy import copy
from copy import deepcopy
from math import ceil
from typing import Optional, Tuple, Union, List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.data.storage import NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling, mask_to_index, add_self_loops
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_networkx, index_to_mask, subgraph, degree, \
    is_undirected, dense_to_sparse

import config


class SocialNetwork(InMemoryDataset):
    datasets = ['facebook_ego', 'twitter_ego', 'youtube', 'gplus_ego']

    def __init__(self, root: str, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        assert self.name in self.datasets
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['features.txt', 'adj.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        x = np.loadtxt(self.raw_paths[0])
        x = torch.from_numpy(x).to(torch.float)
        adj = np.loadtxt(self.raw_paths[1])
        adj = torch.from_numpy(adj).to(torch.float)
        edge_index = dense_to_sparse(adj)[0]
        label = np.loadtxt(self.raw_paths[2])
        y = np.where(label != 0)[1]
        y = torch.tensor(y, dtype=torch.int)
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


class RandomLinkSplit(BaseTransform):
    def __init__(
            self,
            num_val: Union[int, float] = 0.1,
            num_test: Union[int, float] = 0.2,
            is_undirected: bool = False,
            key: str = 'edge_label',
            split_labels: bool = False,
            add_negative_train_samples: bool = True,
            neg_sampling_ratio: float = 1.0,
            disjoint_train_ratio: Union[int, float] = 0.0,
            edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
            rev_edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

        if isinstance(edge_types, list):
            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

    def __call__(self, data: Union[Data, HeteroData]):
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data, val_data, test_data = copy(data), copy(data), copy(data)

        if isinstance(data, HeteroData):
            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to"
                    "be specified when operating on 'HeteroData' objects")

            if not isinstance(edge_types, list):
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            rev_edge_types = [None]
            stores = [data._store]
            train_stores = [train_data._store]
            val_stores = [val_data._store]
            test_stores = [test_data._store]

        for item in zip(stores, train_stores, val_stores, test_stores,
                        rev_edge_types):
            store, train_store, val_store, test_store, rev_edge_type = item

            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= rev_edge_type is None

            edge_index = store.edge_index
            if is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = mask.nonzero(as_tuple=False).view(-1)
                perm = perm[torch.randperm(perm.size(0), device=perm.device)]
            else:
                device = edge_index.device
                perm = torch.randperm(edge_index.size(1), device=device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())

            num_train = perm.numel() - num_val - num_test
            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")

            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]

            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * train_edges.numel())
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(train_store, train_edges[num_disjoint:], is_undirected,
                        rev_edge_type)
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(test_store, train_val_edges, is_undirected,
                        rev_edge_type)

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:num_disjoint]
            self._create_label(
                store,
                train_edges,
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(self, store: EdgeStorage, index: Tensor, is_undirected: bool,
               rev_edge_type: EdgeType):

        for key, value in store.items():
            if key == 'edge_index':
                continue

            if store.is_edge_attr(key):
                value = value[index]
                if is_undirected:
                    value = torch.cat([value, value], dim=0)
                store[key] = value

        edge_index = store.edge_index[:, index]
        if is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = store.edge_index.flip([0])
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(self, store: EdgeStorage, index: Tensor, out: EdgeStorage):

        edge_index = store.edge_index[:, index]

        if hasattr(store, self.key):
            edge_label = store[self.key]
            assert edge_label.dtype == torch.long
            assert edge_label.size(0) == store.edge_index.size(1)
            edge_label = edge_label[index]
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if self.neg_sampling_ratio > 0:
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index

        else:
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')


class TailNodeSplit(BaseTransform):
    def __init__(
            self,
            split: str = "train_rest",
            num_splits: int = 1,
            num_train_per_class: int = 20,
            num_val: Union[int, float] = 500,
            num_test: Union[int, float] = 1000,
            key: Optional[str] = "y",
    ):
        assert split in ['train_rest', 'test_rest', 'random', 'pagerank', 'degree']
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store, data) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage, data=None) -> Tuple[Tensor, Tensor, Tensor]:
        np.random.seed(config.seed)
        num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_copy = train_mask.clone()
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)


        if isinstance(self.num_val, float):
            num_val = round(num_nodes * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_nodes * self.num_test)
        else:
            num_test = self.num_test

        if self.split == 'train_rest':
            perm = torch.randperm(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True
        elif self.split == 'pagerank':
            all_node = np.array([i for i in range(data.num_nodes)])
            num_train = num_nodes - num_test - num_val
            data_nx = to_networkx(data, to_undirected=True)
            data_nx.remove_edges_from(nx.selfloop_edges(data_nx))
            data_pagerank = nx.pagerank(data_nx)
            data_pagerank_l = list(data_pagerank.values())
            data_pagerank_l = np.array(data_pagerank_l)
            sample_node = np.random.choice(all_node,
                                           size=num_train,
                                           replace=False, p=data_pagerank_l)
            remains = np.array([i for i in all_node if i not in sample_node])
            val_ = np.random.choice(remains, size=num_val,
                                    replace=False)
            test_ = [i for i in remains if i not in val_]
            val_ = torch.tensor(val_, dtype=torch.long)
            test_ = torch.tensor(test_, dtype=torch.long)
            val_mask[val_] = True
            test_mask[test_] = True
            train_mask_copy[sample_node] = True
            # train_mask = (~train_mask).nonzero(as_tuple=False).view(-1)
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            train_set = mask_to_index(train_mask_copy)
            for c in range(num_classes):
                idx = (y[train_mask_copy] == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

        elif self.split == 'degree':
            node_degree = degree(data.edge_index[0], num_nodes=data.num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask_copy = train_mask.clone()
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            node_degree = node_degree.cpu().numpy()
            head_nodes = np.where(node_degree >= 5)[0]
            tail_nodes = np.where(node_degree < 5)[0]
            tail_num = tail_nodes.size
            num_val = tail_num // 3
            val_ = np.random.choice(tail_nodes, size=num_val,
                                    replace=False)
            test_ = [i for i in tail_nodes if i not in val_]
            train_ = torch.tensor(head_nodes)
            val_ = torch.tensor(val_)
            test_ = torch.tensor(test_)
            val_mask[val_] = True
            test_mask[test_] = True
            train_mask_copy[train_] = True
            # train_mask[train_] = True
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                idx = (y[train_mask_copy] == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

        else:
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True

            if self.split == 'test_rest':
                test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'


def link_groundtruth(adj, ratio=0.1):
    bound = [2, 10]
    rowsum = np.sum(adj, axis=1)
    select_nodes = np.where(rowsum >= bound[0])[0]
    select_nodes = np.where(rowsum[select_nodes] <= bound[1])[0]

    sum_links = np.sum(adj)
    gt_size = int(sum_links * ratio)

    select_adj = adj[select_nodes].nonzero()
    pairs = np.stack((select_adj[0], select_adj[1]), axis=1)
    # print('pair ', pairs.shape[0])

    gt = np.random.choice(pairs.shape[0], gt_size, replace=False)
    ground_truth = pairs[gt]

    remain = [pairs[i] for i in range(pairs.shape[0]) if i not in gt]
    remain = np.asarray(remain)

    # print('Edges: ', sum_links)
    # print('GT: ', ground_truth.shape[0])

    processed_adj = sp.coo_matrix((np.ones(remain.shape[0]), (remain[:, 0], remain[:, 1])),
                                  shape=(adj.shape[0], adj.shape[0]),
                                  dtype=np.float32)

    return ground_truth, processed_adj.tolil()


def split_links(gt, head, tail): # 把图的连接划分成训练、验证、测试集 链接预测
    h_h = [] # 存储hub-hub边
    t_t = [] # 存储tail-tail边
    h_t = [] # 存储hub-tail边

    for i in range(gt.shape[0]):
        if gt[i][0] in head and gt[i][1] in head:
            h_h.append(i)
        elif gt[i][0] in tail and gt[i][1] in tail:
            t_t.append(i)
        else:
            if gt[i][0] in head and gt[i][1] in tail:
                gt[i][0], gt[i][1] = gt[i][1], gt[i][0] # 确保hub在前，tail在后
            h_t.append(i)

    np.random.shuffle(h_t) # 打乱hub-tail
    half = int(len(h_t) / 2)
    h_t_train = h_t[:half]
    h_t_test = h_t[half:]

    idx_train = np.concatenate((h_h, h_t_train)) # 训练包含hh、ht
    idx_valtest = np.concatenate((t_t, h_t_test)) # 验证包含tt、ht
    np.random.shuffle(idx_valtest) # 打乱验证
    p = int(idx_valtest.shape[0] / 3)
    idx_val = idx_valtest[:p] #接着把验证划分为 真的验证
    idx_test = idx_valtest[p:] # 真的测试

    return idx_train, idx_val, idx_test


def link_dropout(adj, idx): # 生产没有idx所有节点连接信息的邻接矩阵
    # np.random.seed(seed)
    tail_adj = adj.copy()

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[1]
        tail_adj[idx[i]] = 0.0

    return tail_adj


def split_nodes(adj): #根据pagerank值划分hub和tail节点
    tail_node = []
    all_node = np.array([i for i in range(adj.shape[0])])
    graph_nx = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph())
    graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx))
    data_pagerank = nx.pagerank(graph_nx)
    data_pagerank_l = list(data_pagerank.values())
    data_pagerank_l = np.array(data_pagerank_l)
    hub_num = round(config.hub_node_ratio * all_node.shape[0])
    tail_num = round(config.tail_ratio * all_node.shape[0]) # 到这里为止是计算hub和tail的数目
    sample_num = all_node.shape[0] - tail_num # sample_num就是不是tail节点的节点个数
    hub_node = np.random.choice(all_node,
                                size=hub_num, replace=False, p=data_pagerank_l) #按照pagerank值选择前hub_num个节点作为hub节点
    samples = np.random.choice(all_node,
                               size=sample_num,
                               replace=False, p=data_pagerank_l) # 包含了hub和其他节点，这里sample主要是为了选出tail
    for i in all_node:
        if i not in samples:
            tail_node.append(i) # 选择tail节点，不在sample里面的节点就是tail
    sample_tail = np.array(tail_node)  # sample-tail就是tail
    return hub_node, sample_tail


def mutual_process(adj): # 对数据集进行预处理
    gt, new_adj = link_groundtruth(adj.tolil()) # 用于生成连接的标签对应关系，应该是在链接预测中使用

    # build symmetric adjacency matrix
    new_adj = new_adj + new_adj.T.multiply(new_adj.T > new_adj) - new_adj.multiply(new_adj.T > new_adj)
    new_adj = new_adj.tolil()
    hub, tail = split_nodes(new_adj) #划分hub和tail
    tail_adj = link_dropout(new_adj, hub) #生成没有hub结点的邻接矩阵

    idx_train, idx_val, idx_test = split_links(gt, hub, tail)
    # 好像忽略了non-hub或者non-tail的连接！！
    # print(idx_train.shape, idx_val.shape, idx_test.shape)

    return new_adj, tail_adj, gt, (idx_train, idx_val, idx_test), hub, tail


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
    dataset = './data/' + dataset_str + '/'
    adj = sp.csr_matrix(np.loadtxt(dataset + 'adj.txt', dtype=np.float32))
    features = sp.csr_matrix(np.loadtxt(dataset + 'features.txt', dtype=np.float32))
    # features=sp.csr_matrix(np.eye(adj.shape[0]),dtype=np.float32)

    origin, target, timestamp = np.loadtxt()
    labels = np.loadtxt(dataset + 'labels.txt', dtype=np.int32)
    idx_train = np.loadtxt(dataset + 'train.txt', dtype=np.int32)
    idx_val = np.loadtxt(dataset + 'val.txt', dtype=np.int32)
    idx_test = np.loadtxt(dataset + 'test.txt', dtype=np.int32)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print("adj:", adj)
    print("features:", features)
    print("y_train", y_train)
    print("y_val", y_val)
    print("train_mask", train_mask)
    print("val_mask", val_mask)
    print("test_mask", test_mask)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask # 有用到暂时先不看


def group_pagerank(data, pagerank_prob=0.85): #基于pagerank计算节点的centrality，pangerank_prob是pagerank的跳转概率
    VERY_SMALL_NUMBER = 1e-12
    num_nodes = data.num_nodes
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze() # 把edge_index转换成邻接矩阵
    adj_norm = normalize_adj_torch(adj) # 获得对称归一化的邻接矩阵，D-1/2AD-1/2
    idx_train = mask_to_index(data.train_mask) # 获得训练节点的idx，而不是n维的mask形式

    labeled_list = [0 for _ in range(num_classes)] # 用于计算每类被标记的节点的数量
    labeled_node = [[] for _ in range(num_classes)] # 用于存储已标记的属于某类别的节点的index，如labeled_node[i]表示属于i类的标记节点的所有index
    labeled_node_list = []

    for iter1 in idx_train: # 将训练节点存储在不同的类列表里面，并且计数
        iter_label = labels[iter1]
        labeled_node[iter_label].append(iter1)
        labeled_list[iter_label] += 1
        labeled_node_list.append(iter1)

    if num_nodes > 5000: #大规模图
        A = adj_norm.detach()
        A_hat = A.to(config.device) + torch.eye(A.size(0)).to(config.device) #加自环
        D = torch.sum(A_hat, 1) #加自环后的节点度
        D_inv = torch.eye(num_nodes).to(config.device)

        for iter in range(num_nodes): #构建D的逆矩阵，无穷小数保证分母不为0
            if D[iter] == 0:
                D[iter] = VERY_SMALL_NUMBER
            D_inv[iter][iter] = 1.0 / D[iter]
        D = D_inv.sqrt().to(config.device) #开平方

        A_hat = torch.mm(torch.mm(D, A_hat), D)
        temp_matrix = torch.eye(A.size(0)).to(config.device) - pagerank_prob * A_hat
        temp_matrix = temp_matrix.cpu().numpy()
        temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

        inv = torch.from_numpy(temp_matrix_inv).to(config.device)
        P = (1 - pagerank_prob) * inv # 计算pagerank
    else:
        A = adj_norm
        A_hat = A.to(config.device) + torch.eye(A.size(0)).to(config.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        P = (1 - pagerank_prob) * ((torch.eye(A.size(0)).to(config.device) - pagerank_prob * A_hat).inverse())
        # 到这里得到的P就是pagerank转移矩阵

    I_star = torch.zeros(num_nodes)

    for class_index in range(num_classes):
        Lc = labeled_list[class_index]
        Ic = torch.zeros(num_nodes)
        Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc
        if class_index == 0:
            I_star = Ic
        if class_index != 0:
            I_star = torch.vstack((I_star, Ic))

    I_star = I_star.transpose(-1, -2).to(config.device)

    Z = torch.mm(P, I_star) # Z是num_nodes * num _classes 的，表示每个节点属于每个类别的概率
    # 好像没用到，这是直接用pagerank值进行节点分类 辅助节点分类

    return Z


def neighborhood_difficulty_measurer(data):
    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    embed = torch.load(node_pretrain_path).to(config.device).detach()
    _, pred = embed.max(dim=1)
    label = deepcopy(pred)
    label[data.train_mask] = data.y[data.train_mask]
    neighbor_label, _ = add_self_loops(data.edge_index)
    neighbor_label[1] = label[neighbor_label[1]]
    neighbor_label = torch.transpose(neighbor_label, 0, 1)
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)
    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()
    neighbor_class = neighbor_class[data.train_id]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    return local_difficulty.to(config.device)


def sort_training_nodes(data, alpha=0.5):
    neighbor_difficulty = neighborhood_difficulty_measurer(data)
    pagerank = pagerank_cal(data)
    pagerank = torch.tensor(pagerank).to(config.device)
    if pagerank.size(0) != data.train_id.size(0):
        node_difficulty = alpha * neighbor_difficulty - (1 - alpha) * pagerank[data.train_id]
    else:
        node_difficulty = alpha * neighbor_difficulty - (1 - alpha) * pagerank
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def pagerank_cal(data):
    all_node = [i for i in range(data.num_nodes)]
    all_node = torch.tensor(all_node, device=config.device)
    all_train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=config.device)
    all_train_mask[data.val_mask] = False
    all_train_mask[data.test_mask] = False
    all_train_node = all_node[all_train_mask]
    train_sub = subgraph(all_train_mask, data.edge_index)[0]
    train_nx = nx.Graph()
    train_nx.add_nodes_from(all_train_node.cpu().numpy())
    train_nx.add_edges_from(train_sub.t().cpu().numpy())
    train_nx.remove_edges_from(nx.selfloop_edges(train_nx))
    data_pagerank = nx.pagerank(train_nx)
    data_pagerank_l = list(data_pagerank.values())
    data_pagerank_np = np.array(data_pagerank_l)
    data_pagerank_np *= 100
    return data_pagerank_np


def rank_group_pagerank(data, pagerank):
    # pagerank *= 100
    edge_index = remove_self_loops(data.edge_index)[0]
    train_mask = data.train_mask.cpu().numpy()[data.train_id.cpu().numpy()]
    nums = data.num_nodes
    nodes = dict()
    dist = [0] * nums
    adj = to_dense_adj(edge_index, max_num_nodes=nums).squeeze()
    adj = adj.cpu().numpy()
    for i in range(nums):
        nei = np.where(adj[:, i] != 0)[0]
        for j in nei:
            pagerank_dist = (pagerank[i] * pagerank[j].t()).sum(-1).item()
            dist[i] += pagerank_dist
        nodes[i] = dist[i] / (nei.shape[0] + 1)
    train_subset = np.array(list(nodes.items()))[data.train_id.cpu().numpy()]
    train_subset = np.column_stack((train_subset, train_mask))
    # nodes_sorted = sorted(nodes.items(), key=lambda x: x[1], reverse=False)
    nodes_sorted = sorted(train_subset, key=lambda x: x[1], reverse=False)
    nodes_sorted = np.array(nodes_sorted)
    nodes = nodes_sorted[:, 0].astype(int)
    sorted_train_mask = nodes_sorted[:, 2].astype(bool)

    return nodes, sorted_train_mask


def rank_group_pagerank_KL(data, pagerank_before, pagerank_after):
    num_nodes = data.num_nodes

    KL_A = pagerank_before[0]
    KL_B = pagerank_after

    for i in range(num_nodes):
        if i == 0:
            for j in range(num_nodes - 1):
                KL_A = torch.vstack((KL_A, pagerank_before[i]))
        else:
            for j in range(num_nodes):
                KL_A = torch.vstack((KL_A, pagerank_before[i]))

    for i in range(num_nodes - 1):
        KL_B = torch.vstack((KL_B, pagerank_after))

    pagerank_dist = F.kl_div(KL_A.softmax(dim=-1).log(), KL_B.softmax(dim=-1), reduction='none').detach()
    pagerank_dist = torch.sum(pagerank_dist, dim=1) * (-1)

    node_pair_group_pagerank_mat_list = pagerank_dist.flatten()
    index = torch.argsort(-node_pair_group_pagerank_mat_list)
    rank = torch.argsort(index)
    rank += 1
    node_pair_group_pagerank_mat = torch.reshape(rank, (num_nodes, num_nodes))

    return node_pair_group_pagerank_mat


def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def get_nei(data, index): # 主要是得到邻居掩码
    if hasattr(data, 'edge_label_index'):
        edge_index = remove_self_loops(data.edge_label_index)
    else:
        edge_index = remove_self_loops(data.edge_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=data.num_nodes)
    adj = adj.squeeze().cpu().numpy()
    if not is_undirected(data.edge_index):
        out_nei_idx = np.where(adj[:, index] != 0)[0]
        in_nei_idx = np.where(adj[index, :] != 0)[0]
        out_nei_idx = torch.tensor(out_nei_idx, device=config.device)
        in_nei_idx = torch.tensor(in_nei_idx, device=config.device)
        nei_idx = torch.hstack((out_nei_idx, in_nei_idx))
    else:
        nei_idx = torch.tensor(np.where(adj[:, index] != 0)[0])
    nei_idx_mask = index_to_mask(nei_idx, size=data.num_nodes)
    # 这里的nei_idx_mask是将邻居的idx转换为掩码的形式，在是邻居的位置是True否则是False
    return nei_idx, nei_idx_mask


def get_nei_feature4node(data, samples=None, use_fullnei=False):
    random_nonhub = []
    neifeat = []
    neilabel = []
    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    tail_ratio = data_params['tail_ratio']
    all_node = [i for i in range(data.num_nodes)]
    all_node = torch.tensor(all_node, device=config.device)
    all_train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=config.device) # 创建训练集掩码初始全为true
    all_train_mask[data.val_mask] = False
    all_train_mask[data.test_mask] = False
    all_train_node = all_node[all_train_mask] # 找到训练集
    # val_test_num = all_node[data.val_mask].size(0) + all_node[data.test_mask].size(0)
    sample_num = round(all_train_node.size(0) * (1 - tail_ratio)) # sample所有不是tail的节点数目
    train_subgraph = subgraph(all_train_mask, data.edge_index)[0] # 得到训练subgraph
    data_nx = nx.Graph()
    data_nx.add_nodes_from(all_train_node.cpu().numpy())
    data_nx.add_edges_from(train_subgraph.t().cpu().numpy())

    edge_index = remove_self_loops(data.edge_index) # 去自环
    adj = to_dense_adj(edge_index[0], max_num_nodes=data.num_nodes).cpu().numpy()
    adj = adj.reshape(data.num_nodes, data.num_nodes) # 邻接矩阵
    if use_fullnei: #根据use-fullnei的设定提取特定节点的邻居的特征和label
        neifeat = {}
        neilabel = {}
        nei_num = 0
        if samples is None: #如果没指定samples，就在这里samples
            data_nx.remove_edges_from(nx.selfloop_edges(data_nx))
            data_pagerank = nx.pagerank(data_nx)
            data_pagerank_l = list(data_pagerank.values())
            data_pagerank_l = np.array(data_pagerank_l)
            sample_node = np.random.choice(all_train_node.cpu().numpy(), size=sample_num,
                                           replace=False, p=data_pagerank_l)
            for i in all_train_node.cpu().numpy():
                if i not in sample_node:
                    random_nonhub.append(i)
            sample_tail = np.array(random_nonhub) # 和前面的逻辑差不多，都是sample出tail节点

            for i in sample_tail:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx):
                    nei_feature = data.x[nei_idx]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                    nei_num += nei_label.size(0)
                else:
                    nei_feature = data.x[i]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                    nei_num += 1
            return sample_tail, neifeat, neilabel, nei_num # 返回sample_tail和邻居特征、邻居的label、邻居的个数
        else: #传了samples的情况
            for i in samples:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx):
                    nei_feature = data.x[nei_idx]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                    nei_num += nei_label.size(0)
                else:
                    nei_feature = data.x[i]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                    nei_num += 1
            return samples, neifeat, neilabel, nei_num # 就是少了samples那一步
    else: # 不用full-nei
        if samples is None:
            data_nx.remove_edges_from(nx.selfloop_edges(data_nx))
            #data_pagerank = nx.pagerank(data_nx)
            #data_pagerank_l = list(data_pagerank.values())
            #data_pagerank_l = np.array(data_pagerank_l)
            degrees = dict(data_nx.degree())
            degrees_l = list(degrees.values())
            degrees_l = np.array(degrees_l)
            # 我去这里是根据degree采样
            # 就是对度低的节点在生成邻居的时候只采样一个邻居，根据这个邻居生成特征最相似的邻居，对应于generate模块
            #sample_node = np.random.choice(all_train_node.cpu().numpy(), size=sample_num,
            #                               replace=False, p=data_pagerank_l)
            sample_node = np.random.choice(all_train_node.cpu().numpy(), size=sample_num,
                                           replace=False, p=degrees_l)
            for i in all_train_node.cpu().numpy():
                if i not in sample_node:
                    random_nonhub.append(i)
            sample_tail = np.array(random_nonhub)

            for i in sample_tail:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx) is True:
                    nei_feature = data.x[nei_idx]
                    self_feature = data.x[i].expand_as(nei_feature)
                    cos_sim = cosine_similarity(self_feature, nei_feature, dim=1)
                    most_similar_nei = torch.argmax(cos_sim)
                    similar_nei_feat = nei_feature[most_similar_nei].cpu().numpy() #上面是采样所有的邻居
                    nei_label = data.y[most_similar_nei].cpu().numpy() # 这里是根据余弦相似度取一个最相似的
                    neifeat.append(similar_nei_feat)
                    neilabel.append(nei_label)
                else:
                    nei_feature = data.x[i]
                    neifeat.append(nei_feature.cpu().numpy())
                    nei_label = data.y[i].cpu().numpy()
                    neilabel.append(nei_label)
            nei_feat = torch.tensor(np.array(neifeat, dtype=np.float32), device=config.device)
            neilabel = torch.tensor(np.array(neilabel), dtype=torch.int, device=config.device)
            return sample_tail, nei_feat, neilabel

        else:
            for i in samples:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx) is True:
                    nei_feature = data.x[nei_idx]
                    self_feature = data.x[i].expand_as(nei_feature)
                    cos_sim = cosine_similarity(self_feature, nei_feature, dim=1)
                    most_similar_nei = torch.argmax(cos_sim)
                    similar_nei_feat = nei_feature[most_similar_nei].cpu().numpy()
                    nei_label = data.y[most_similar_nei].cpu().numpy()
                    neifeat.append(similar_nei_feat)
                    neilabel.append(nei_label)
                else:
                    nei_feature = data.x[i]
                    neifeat.append(nei_feature.cpu().numpy())
                    nei_label = data.y[i].cpu().numpy()
                    neilabel.append(nei_label)
            nei_feat = torch.tensor(np.array(neifeat, dtype=np.float32), device=config.device)

            neilabel = torch.tensor(np.array(neilabel), dtype=torch.int, device=config.device)
            return samples, nei_feat, neilabel


def get_neighbor_feature4link(data, samples=None, use_fullnei=False, tail_ratio=config.tail_ratio):
    random_nonhub = []
    neifeat = []
    neilabel = []
    parameters = json.load(open('lp_parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    tail_ratio = data_params['tail_ratio']
    all_node = np.array([i for i in range(data.num_nodes)])
    sample_num = round((1 - tail_ratio) * data.num_nodes)
    edge_index = remove_self_loops(data.edge_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=data.num_nodes).cpu().numpy()
    adj = adj.reshape(data.num_nodes, data.num_nodes)

    if use_fullnei:
        neifeat = {}
        neilabel = {}
        nei_num = 0
        if samples is None:
            data_nx = to_networkx(data, to_undirected=True)
            data_nx.remove_edges_from(nx.selfloop_edges(data_nx))
            data_pagerank = nx.pagerank(data_nx)
            data_pagerank_l = list(data_pagerank.values())
            data_pagerank_l = np.array(data_pagerank_l)
            sample_node = np.random.choice(all_node, size=sample_num,
                                           replace=False, p=data_pagerank_l)
            for i in all_node:
                if i not in sample_node:
                    random_nonhub.append(i)
            sample_tail = np.array(random_nonhub)

            for i in sample_tail:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx):
                    nei_feature = data.x[nei_idx]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                    nei_num += nei_label.size(0)
                else:
                    nei_feature = data.x[i]
                    neifeat[i] = nei_feature
                    nei_label = data.y[i]
                    neilabel[i] = nei_label
                    nei_num += 1
            return sample_tail, neifeat, neilabel, nei_num
        else:
            for i in samples:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_num += nei_idx.size(0)
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx):
                    nei_feature = data.x[nei_idx]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
                else:
                    nei_feature = data.x[i]
                    neifeat[i] = nei_feature
                    nei_label = data.y[nei_idx]
                    neilabel[i] = nei_label
            return samples, neifeat, neilabel
    else:
        if samples is None:
            all_node = np.array([i for i in range(data.num_nodes)])
            data_nx = to_networkx(data, to_undirected=True)
            data_nx.remove_edges_from(nx.selfloop_edges(data_nx))
            data_pagerank = nx.pagerank(data_nx)
            data_pagerank_l = list(data_pagerank.values())
            data_pagerank_l = np.array(data_pagerank_l)
            sample_node = np.random.choice(all_node, size=sample_num,
                                           replace=False, p=data_pagerank_l)
            for i in all_node:
                if i not in sample_node:
                    random_nonhub.append(i)
            sample_tail = np.array(random_nonhub)

            for i in sample_tail:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx) is True:
                    nei_feature = data.x[nei_idx]
                    self_feature = data.x[i].expand_as(nei_feature)
                    cos_sim = cosine_similarity(self_feature, nei_feature, dim=1)
                    most_similar_nei = torch.argmax(cos_sim)
                    similar_nei_feat = nei_feature[most_similar_nei].cpu().numpy()
                    nei_label = data.y[most_similar_nei].cpu().numpy()
                    neifeat.append(similar_nei_feat)
                    neilabel.append(nei_label)
                else:
                    nei_feature = data.x[i]
                    neifeat.append(nei_feature.cpu().numpy())
                    nei_label = data.y[i].cpu().numpy()
                    neilabel.append(nei_label)
            nei_feat = torch.tensor(np.array(neifeat, dtype=np.float32), device=config.device)
            neilabel = torch.tensor(np.array(neilabel), dtype=torch.int, device=config.device)
            return sample_tail, nei_feat, neilabel
        else:
            for i in samples:
                nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
                nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
                if any(nei_idx) is True:
                    nei_feature = data.x[nei_idx]
                    self_feature = data.x[i].expand_as(nei_feature)
                    cos_sim = cosine_similarity(self_feature, nei_feature, dim=1)
                    most_similar_nei = torch.argmax(cos_sim)
                    similar_nei_feat = nei_feature[most_similar_nei].cpu().numpy()
                    neifeat.append(similar_nei_feat)
                    nei_label = data.y[most_similar_nei].cpu().numpy()
                    neifeat.append(similar_nei_feat)
                    neilabel.append(nei_label)
                else:
                    nei_feature = data.x[i]
                    neifeat.append(nei_feature.cpu().numpy())
                    nei_label = data.y[i].cpu().numpy()
                    neilabel.append(nei_label)
            nei_feat = torch.tensor(np.array(neifeat, dtype=np.float32), device=config.device)

            neilabel = torch.tensor(np.array(neilabel), dtype=torch.int, device=config.device)
            return samples, nei_feat, neilabel


def decode_low_degree(edge_embed, full_edge_embed, node_embed, full_node_embed):
    edge_prob = edge_embed @ full_edge_embed.t() # 计算node1和node2之间的edge_embed相似度 @ 代表矩阵乘法
    edge_prob = torch.sigmoid(edge_prob) # 通过sigmoid划到0-1的范围
    label_sim = node_embed @ full_node_embed.t()
    prob = edge_prob * label_sim # 内机得到相似度
    return prob.detach()


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
