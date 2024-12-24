import json
import math
import os
import time
from collections import defaultdict
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch.autograd import Variable
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.datasets import WikipediaNetwork, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops, mask_to_index
from torch_geometric.utils import index_to_mask
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import GDC, RandomNodeSplit

import config
from models import Discriminator
from models import NeighGen
from models import Net
from utils import get_nei_feature4node, \
    decode_low_degree, get_nei, TailNodeSplit, SocialNetwork, training_scheduler, group_pagerank, rank_group_pagerank, \
    sort_training_nodes

def create_masks(data, num_train_per_class=20, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    创建训练集、验证集和测试集的掩码。

    参数：
    - data: 图数据，包含节点标签和节点数
    - num_train_per_class: 每个类别的训练集节点数
    - val_ratio: 验证集占总节点的比例
    - test_ratio: 测试集占总节点的比例
    - seed: 随机种子，保证结果可复现

    返回：
    - train_mask, val_mask, test_mask: 训练集、验证集和测试集的掩码
    """
    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 获取标签信息
    labels = data.y.cpu().numpy() # 假设data.y是节点的标签信息
    num_classes = len(np.unique(labels))

    # 创建每个类别的节点索引
    class_nodes = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # 训练集、验证集和测试集的节点索引
    train_nodes = []
    val_nodes = []
    test_nodes = []

    # 对每个类进行划分
    for i in range(num_classes):
        class_idx = class_nodes[i]
        # 打乱每一类的节点顺序
        np.random.shuffle(class_idx)

        # 选择20个节点到训练集
        train_nodes.extend(class_idx[:num_train_per_class])

    # 将训练集的节点去除，剩下的用于划分验证集和测试集
    remaining_nodes = np.setdiff1d(np.arange(data.num_nodes), train_nodes)

    # 按比例划分剩余节点
    np.random.shuffle(remaining_nodes)
    val_size = int(val_ratio * data.num_nodes)  # 验证集为总节点的val_ratio比例
    test_size = int(test_ratio * data.num_nodes)  # 测试集为总节点的test_ratio比例

    val_nodes = remaining_nodes[:val_size]
    test_nodes = remaining_nodes[val_size:val_size + test_size]

    # 确保没有交集
    assert len(set(train_nodes) & set(val_nodes)) == 0, "训练集和验证集有交集"
    assert len(set(train_nodes) & set(test_nodes)) == 0, "训练集和测试集有交集"
    assert len(set(val_nodes) & set(test_nodes)) == 0, "验证集和测试集有交集"

    # 创建掩码
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = 1
    val_mask[val_nodes] = 1
    test_mask[test_nodes] = 1

    return train_mask, val_mask, test_mask

# 计算损失
def greedy_loss(pred_feats, true_feat):
    loss = torch.zeros_like(pred_feats).to(config.device)
    pred_len = pred_feats.size(0)
    for i in range(pred_len):
        for j in range(config.num_pred):
            loss[i][j] += F.mse_loss(pred_feats[i][j].unsqueeze(0).float(),
                                     true_feat[i].unsqueeze(0).float()).squeeze(0)

    return loss.mean()

# 训练邻居生成器
def train_gen(data, sample_node=None): # 因为要重新sample节点，所以这里可以默认为None
    if config.use_fullneifeat:
        best_gen_feat = {}
        z = {}
        if sample_node is None:
            sample_tail, nei_feat, nei_label, nei_num = get_nei_feature4node(data, use_fullnei=True)
        else:
            sample_tail, nei_feat, nei_label, nei_num = get_nei_feature4node(data, sample_node, use_fullnei=True)

        feature_size = data.x.size(1)
        gen = NeighGen(feature_size, data, sample_tail, nei_feat).to(config.device)
        dis = Discriminator(feat_shape=feature_size).to(config.device)

        optimizer_gen = optim.Adam(gen.parameters(),
                                   lr=config.lr, weight_decay=config.weight_decay)

        gen_feat = []
        best_gen_feat = []
        min_loss_gen = 10
        print("Training neighbor_generator...")
        for epoch in range(config.gan_epoch):
            t = time.time()
            feat_loss = 0
            for i in sample_tail:
                z[i] = Variable(torch.FloatTensor(np.random.normal
                                                  (0, 1, size=nei_feat[i].size()))).to(config.device)
            gen.train()
            optimizer_gen.zero_grad()
            true_label = torch.ones(data.num_nodes, device=config.device)
            gen_label = torch.zeros(nei_num, device=config.device)
            gen_feat, label_predict = gen(z)
            for i in sample_tail:
                feat_loss += F.mse_loss(gen_feat[i], nei_feat[i])
            feat_loss /= nei_num
            label = torch.concat([true_label, gen_label], dim=0)
            gen_node = np.array([i for i in range(data.num_nodes,
                                                  data.num_nodes + nei_num)])
            gen_idx = np.zeros(data.num_nodes + nei_num, dtype=bool)
            gen_idx[gen_node] = True
            label_loss = F.binary_cross_entropy(label_predict[gen_idx], label[gen_idx])

            gen_loss = feat_loss + label_loss
            gen_loss.backward()
            optimizer_gen.step()
            gen.eval()
            if gen_loss.item() < min_loss_gen:
                min_loss_gen = gen_loss.item()
                best_gen_feat = gen_feat
        print(f'min_loss_gen:{min_loss_gen:.4f}')
        print("Training generator finished!")
        return best_gen_feat, sample_tail, nei_label # fullneifeat是False就先不看了

    else:
        if sample_node is None: # 就是获取tail的最相似邻居的信息 根据这个生成！！
            sample_tail, nei_feat, nei_label = get_nei_feature4node(data, use_fullnei=False)
        else:
            sample_tail, nei_feat, nei_label = get_nei_feature4node(data, sample_node, use_fullnei=False)

        feature_size = data.x.size(1) # 特征维度要想同
        gen = NeighGen(feature_size, data, sample_tail, nei_feat).to(config.device) # 生成模型

        optimizer_gen = optim.Adam(gen.parameters(),
                                   lr=config.lr, weight_decay=config.weight_decay)

        gen_feat = []
        best_gen_feat = []
        min_loss_gen = 10
        print("Training neighbor_generator...")
        for epoch in range(config.gan_epoch):
            t = time.time()
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (sample_tail.shape[0], data.x.size(1))))).to(
                config.device) # 生成噪声向量，就是用噪声来拟合特征的
            gen.train()
            optimizer_gen.zero_grad()

            gen_feat, label_predict = gen(z)
            label_predict = label_predict.squeeze()

            feat_loss = F.mse_loss(gen_feat, nei_feat)
            true_laebl = torch.ones(data.num_nodes, device=config.device)
            gen_label = torch.zeros(len(sample_tail), device=config.device)
            label = torch.concat([true_laebl, gen_label], dim=0)
            gen_node = np.array([i for i in range(data.num_nodes,
                                                  data.num_nodes + len(sample_tail))])
            gen_idx = np.zeros(data.num_nodes + len(sample_tail), dtype=bool)
            gen_idx[gen_node] = True
            label_loss = F.binary_cross_entropy(label_predict[gen_idx], label[gen_idx])

            gen_loss = feat_loss + label_loss
            gen_loss.backward()
            if gen_loss.item() < min_loss_gen:
                min_loss_gen = gen_loss.item()
                best_gen_feat = gen_feat
            optimizer_gen.step()
            gen.eval()

            """
            print(f'time:{time.time() - t},'
                  f'gen_epoch:{epoch},'
                  f'loss_gen:{gen_loss:.4f},'
                  f'feat_loss:{feat_loss:.4f},'
                  f'label:{label_loss:.4f},')
            """
        print(f'min_loss_gen:{min_loss_gen:.4f}')
        print("Training generator finished!") # 没仔细看 先跳过了
        return best_gen_feat, sample_tail, nei_label #反正这里就是训练生成器，得到最拟合的特征，tail节点组和最相似的邻居的label


def model_training_cl(data, sample_tail, use_weight=False):
    macro_results = []
    micro_results = []
    mu = config.mu
    data_copy = data.clone()
    save_path = './saved_model/' + config.dataset + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if use_weight:
        loss_function = torch.nn.CrossEntropyLoss(reduction="none")
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
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    labels = data.y.cpu().numpy()
    num_classes = int(np.max(labels) + 1)
    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    add_threshold = data_params['add_threshold']
    add_topK = data_params['add_topK']

    # edge_index = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(data.edge_index)[0]
    adj = to_dense_adj(edge_index, max_num_nodes=data.num_nodes)
    adj = adj.squeeze().cpu().numpy()

    node_embed = torch.load(node_pretrain_path).to(config.device).detach()
    link_embed = torch.load(link_pretrain_path).to(config.device).detach()
    # new_embed = link_embed.clone()
    # new_embed[high_degree_node] = 0
    prob = decode_low_degree(link_embed[sample_tail], link_embed,
                             node_embed[sample_tail], node_embed)

    index = 0
    for i in sample_tail:
        nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
        nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
        prob[index][nei_idx] = 0
        index += 1

    print("adding neighbor for low_degree_node which score > {}".format(add_threshold))
    prob_np = prob.cpu().numpy()
    origin_node_idx, predict_neighbor = np.where(prob_np >= add_threshold)
    edge_dct = defaultdict(list)
    edge_prob = []
    origin_node = np.zeros(origin_node_idx.shape[0], dtype=int)
    for i in range(origin_node_idx.shape[0]):
        origin_node[i] = sample_tail[origin_node_idx[i]]
        edge_dct[origin_node_idx[i]].append(predict_neighbor[i])
    for i in edge_dct.keys():
        for j in edge_dct[i]:
            edge_prob.append(prob_np[i][j])

    new_edge = np.vstack((origin_node, predict_neighbor, edge_prob))
    new_edges_trans = new_edge.T
    new_edge_ordered = new_edges_trans[np.argsort(-new_edges_trans[:, 2])]
    new_edges_trans = new_edge_ordered[:, 0:2].astype(int)

    for i in range(config.train_iter):
        cur_step = 0
        min_loss = 10
        best_val_f1 = 0
        start = 0
        edge_size = new_edges_trans.shape[0]
        model = Net(data.num_features, config.embed_dim,
                    config.dropout, num_classes=num_classes).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            cl_sample_ratio = mu * (1 - math.cos(epoch / config.epoch * (math.pi / 2)))
            ratio = training_scheduler(config.lam, epoch, config.epoch, config.scheduler)
            size = int(round(edge_size * ratio))
            if size == 0:
                continue
            if start + size + 1 < new_edges_trans.shape[0]:
                add_edges = torch.tensor(new_edges_trans[start:start + size + 1].T).to(config.device)
                data.edge_index = torch.cat([data.edge_index, add_edges], dim=1)
                start += size + 1
            if start < new_edges_trans.shape[0] <= start + size + 1:
                add_edges = torch.tensor(new_edges_trans[start:-1].T).to(config.device)
                data.edge_index = torch.cat([data.edge_index, add_edges], dim=1)
                start += size + 1
            # new_edges_trans = np.delete(new_edges_trans, adding_idx, axis=0)
            # index = np.arange(new_edges_trans.shape[0])

            model.train()
            optimizer.zero_grad()
            h = model.node_classify(data)
            loss = loss_function(h[data.train_mask], data.y[data.train_mask])
            if use_weight:
                data_pagerank = nx.pagerank(train_nx)
                data_pagerank_l = list(data_pagerank.values())
                data_pagerank_l = np.array(data_pagerank_l) * 100
                pagerank_weight_np = 1 / np.exp(data_pagerank_l)
                pagerank_weight = torch.tensor(pagerank_weight_np).to(config.device)
                loss *= pagerank_weight
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            if epoch % config.eval_step == 0:
                model.eval()
                val_h = model.node_classify(data)
                _, val_predict_y = val_h[data.val_mask].max(dim=1)
                val_f1 = f1_score(val_predict_y.cpu(), data.y[data.val_mask].cpu(), average="macro")
                """
                print(f'epoch:{epoch + 1}'
                      f'loss:{loss:.4f}'
                      f'val_f1:{val_f1:.4f}')
                """
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_logits = val_h
                    epoch_early_stop = epoch
                    torch.save(model, os.path.join(save_path, 'model.pt'))
                    cur_step = 0
                else:
                    cur_step += 1
                    if cur_step == config.patience:
                        early_stop = 'Early Stopping at epoch {:d}'.format(epoch)
                        print(early_stop)
                        break

                if loss.item() < min_loss:
                    min_loss = loss.item()
        embed_model = torch.load(os.path.join(save_path, 'model.pt'))
        test_h = embed_model.node_classify(data)[data.test_mask]
        _, test_predict_y = test_h.max(dim=1)
        # _, test_predict_y = best_logits[data.test_mask].max(dim=1)
        test_f1_macro = f1_score(test_predict_y.cpu(), data.y[data.test_mask].cpu(), average="macro")
        test_f1_micro = f1_score(test_predict_y.cpu(), data.y[data.test_mask].cpu(), average="micro")
        print(f'test_f1_macro:{test_f1_macro:.4f},'
              f'test_f1_micro:{test_f1_micro:.4f}')
        macro_results.append(test_f1_macro)
        micro_results.append(test_f1_micro)
        data = data_copy.clone()
    macro_result = 100 * torch.tensor(macro_results)
    micro_result = 100 * torch.tensor(micro_results)
    f1_macro_avg = macro_result.mean()
    f1_macro_std = macro_result.std()
    f1_micro_avg = micro_result.mean()
    f1_micro_std = micro_result.std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'f1_macro:{f1_macro_avg:.2f}±{f1_macro_std:.2f}'
          f'f1_micro:{f1_micro_avg:.2f}±{f1_micro_std:.2f}')


def model_training(data, val_data, with_gen=False, sample_node=None, use_weight=False):
    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    tail_ratio = data_params['tail_ratio']
    hub_times = data_params['hub_times']
    data_copy = data.clone()
    save_path = './saved_model/' + config.dataset + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path) # 储存模型
    cur_step = 0
    min_loss = 10
    best_val_f1 = 0
    train_time = 0
    labels = data.y.cpu().numpy()
    num_classes = int(np.max(labels) + 1)
    if use_weight:
        loss_function = torch.nn.CrossEntropyLoss(reduction="none") # 直接返回每个样本的单独损失，适合对损失进行加权操作
    else:
        loss_function = torch.nn.CrossEntropyLoss() # 返回平均损失
    best_logits = None
    if with_gen:
        if sample_node is None:
            gen_feat, non_hub_node, nei_label = train_gen(data)

        else:
            gen_feat, non_hub_node, nei_label = train_gen(data, sample_node)

        if isinstance(gen_feat, dict): # 判断生产的特征gen_feat是不是dict字典形式
            edge_1 = []
            genfeat = []
            neilabel = []
            for i in gen_feat.keys(): # 遍历每个生成节点的ID
                if gen_feat[i].size(0) == data.x.size(1): # gen_feat[i].size(0)==每个节点的特征维度
                    edge_1.append(i) # 这种情况下表示最好的gen_feat只有一个特征向量，可以直接加到edge里
                    genfeat.append(gen_feat[i].cpu().detach().numpy().tolist())
                    neilabel.append(nei_label[i].cpu().numpy().tolist())
                else: # 否则说明gen_feat[i]包含多个特征向量
                    for j in range(gen_feat[i].size(0)): # 逐一遍历这些生产特征并将他们存储
                        edge_1.append(i)
                        genfeat.append(gen_feat[i][j].cpu().detach().numpy().tolist())
                        neilabel.append(nei_label[i][j].cpu().numpy().tolist())
                # 允许一个生成节点有多个生成特征

            edge_1 = np.array(edge_1, dtype=int) # edge_1是目标节点的id，也就是边的起始节点
            edge_2 = np.array([i for i in range(data.num_nodes, data.num_nodes + len(edge_1))]) # edge2用来存储生产的节点id，拼在已有的id后面
            edge_gen = np.vstack([edge_1, edge_2])# 形成新边列表、只包含生成边
            new_edge = torch.tensor(edge_gen, dtype=torch.int, device=config.device)
            data.edge_index = torch.hstack((data.edge_index, new_edge)) # hstack可以认为是cat指定dim=1
            new_train_mask = torch.zeros(new_edge.size(1), dtype=torch.bool, device=config.device)
            new_val_mask = torch.zeros(new_edge.size(1), dtype=torch.bool, device=config.device)
            data.train_mask = torch.cat([data.train_mask, new_train_mask])
            data.val_mask = torch.cat([data.val_mask, new_val_mask])
            data.test_mask = torch.cat([data.test_mask, new_val_mask])

            genfeat = np.array(genfeat)
            genfeat = torch.tensor(genfeat, device=config.device)
            neilabel = np.array(neilabel)
            neilabel = torch.tensor(neilabel, device=config.device)
            data.x = torch.vstack([data.x, genfeat]).to(torch.float)
            data.y = torch.cat([data.y, neilabel])

        else:
            edge_1 = []
            for i in range(len(non_hub_node)):
                for j in range(config.num_pred):
                    edge_1.append(non_hub_node[i])

            edge_1 = np.array(edge_1, dtype=int)
            edge_2 = np.array([i for i in range(data.train_mask.size(0),
                                                data.train_mask.size(0) + len(non_hub_node) * config.num_pred)])
            edge_generate = np.vstack((edge_1, edge_2))
            data.num_nodes += len(non_hub_node) * config.num_pred

            new_edge = torch.tensor(edge_generate, dtype=torch.int, device=config.device)
            data.edge_index = torch.hstack((data.edge_index, new_edge))
            new_train_mask = torch.zeros(new_edge.size(1), dtype=torch.bool, device=config.device)
            new_val_mask = torch.zeros(new_edge.size(1), dtype=torch.bool, device=config.device)

            data.train_mask = torch.cat([data.train_mask, new_train_mask])
            data.val_mask = torch.cat([data.val_mask, new_val_mask])

            device = new_val_mask.device
            data.test_mask = data.test_mask.to(device)
            data.test_mask = torch.cat([data.test_mask, new_val_mask])
            data.x = torch.vstack((data.x, gen_feat.view(-1, data.x.size(1)).detach()))
            data.y = torch.cat([data.y, nei_label])

        # 这里还在with_gen内部
        model_withgan = Net(data.num_features, config.embed_dim,
                            config.dropout, num_classes=num_classes).to(config.device)
        loss_function = torch.nn.NLLLoss() # 负对数似然损失函数，比较适合用在节点分类任务
        optimizer = optim.Adam(model_withgan.parameters(), lr=config.lr)

        for epoch in range(config.epoch):
            model_withgan.train()
            optimizer.zero_grad()
            h = model_withgan.node_classify(data) # Net没有forward，直接调用节点分类就进行节点分类，灵活度更高
            loss = loss_function(h[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % config.eval_step == 0:
                model_withgan.eval()
                val_h = model_withgan.node_classify(data)[data.val_mask]
                _, val_predict_y = val_h.max(dim=1)
                val_f1 = f1_score(val_predict_y.cpu(), data.y[data.val_mask].cpu(), average="macro")
                """
                print(f'epoch:{epoch + 1}'
                      f'loss:{loss:.4f}'
                      f'val_f1:{val_f1:.4f}'
                      f'test_f1:{test_f1:.4f}')
                """
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epoch_early_stop = epoch
                    torch.save(model_withgan, os.path.join(save_path, 'model_withgan.pt'))
                    cur_step = 0
                else:
                    cur_step += 1
                    if cur_step == config.patience:
                        early_stop = 'Early Stopping at epoch {:d}'.format(epoch)
                        print(early_stop)
                        break

                if loss.item() < min_loss:
                    min_loss = loss.item()


        embed_model = torch.load(os.path.join(save_path, 'model_withgan.pt')) # 加载最佳模型
        test_h = embed_model.node_classify(data) #
        _, test_predict_y = test_h.max(dim=1)


        test_f1_macro = f1_score(test_predict_y[data.test_mask].cpu(), data.y[data.test_mask].cpu(), average="macro")
        test_f1_micro = f1_score(test_predict_y[data.test_mask].cpu(), data.y[data.test_mask].cpu(), average="micro")
        print(f'test_f1_macro:{test_f1_macro:.4f},'
              f'test_f1_micro:{test_f1_micro:.4f}')


        data = data_copy.clone()

        return test_f1_macro,test_f1_micro
    else:
        if use_weight:
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
            data_pagerank_l = np.array(data_pagerank_l) * 100
            pagerank_weight_np = 1 / np.exp(data_pagerank_l)
            pagerank_weight = torch.tensor(pagerank_weight_np).to(config.device)
        model = Net(data.num_features, config.embed_dim,
                    config.dropout, num_classes=num_classes).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            h = model.node_classify(data)
            loss = loss_function(h[data.train_mask], data.y[data.train_mask]) # 如果没有生成器就是交叉熵损失函数
            if use_weight:
                loss *= pagerank_weight
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            train_time += end_time - start_time
            if epoch % config.eval_step == 0:
                model.eval()
                val_h = model.node_classify(data)
                _, val_predict_y = val_h[data.val_mask].max(dim=1)
                val_f1 = f1_score(val_predict_y.cpu(), data.y[data.val_mask].cpu(), average="macro")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_logits = val_h
                    epoch_early_stop = epoch
                    torch.save(model, os.path.join(save_path, 'model.pt'))
                    cur_step = 0
                else:
                    cur_step += 1
                    if cur_step == config.patience:
                        early_stop = 'Early Stopping at epoch {:d}'.format(epoch)
                        print(early_stop)
                        break

                if loss.item() < min_loss:
                    min_loss = loss.item()


        avg_train_time = train_time / epoch
        embed_model = torch.load(os.path.join(save_path, 'model.pt'))
        #test_h = embed_model.node_classify(data)[data.test_mask]
        test_h = embed_model.node_classify(data)
        _, test_predict_y = test_h.max(dim=1)
        # _, test_predict_y = best_logits[data.test_mask].max(dim=1)


        test_f1_macro = f1_score(test_predict_y[data.test_mask].cpu(), data.y[data.test_mask].cpu(), average="macro")
        test_f1_micro = f1_score(test_predict_y[data.test_mask].cpu(), data.y[data.test_mask].cpu(), average="micro")
        print(f'test_f1_macro:{test_f1_macro:.4f},'
              f'test_f1_micro:{test_f1_micro:.4f}')


        print("average training time: " + str(avg_train_time))
        data = data_copy.clone()

        #return test_f1_macro, test_f1_micro, test_f1_macro_hub ,test_f1_micro_hub, test_f1_macro_tail, test_f1_micro_tail
        return test_f1_macro,test_f1_micro

def drop_edges(data, hub_node): # 为什么说只在训练集上进行！是因为hub_nodes是根据训练集选出来的，测试集没有分
    data_copy = data.clone()
    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    drop_thres = data_params['drop_threshold']
    node_embed = torch.load(node_pretrain_path).to(config.device)
    link_embed = torch.load(link_pretrain_path).to(config.device)
    macro_results = []
    micro_results = []

    edge_index = remove_self_loops(data.edge_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=data.num_nodes).squeeze()
    # edge_index[0]表示edge_index中所有起点节点的索引
    # 第二个参数指定邻接矩阵的形状，data.num_nodes * data.num_nodes
    for i in hub_node:
        hub_nei_idx, hub_nei_mask = get_nei(data, i) # 获取hub_nodes的邻居和其掩码
        hub_prob = decode_low_degree(link_embed[i], link_embed[hub_nei_mask],
                                     node_embed[i], node_embed[hub_nei_mask])
        hub_prob_array = hub_prob.cpu().numpy()
        if np.any(hub_prob_array < drop_thres):
            dissim_nei = np.where(hub_prob_array < config.dissim)[0]
            dissim_nei_idx = hub_nei_idx[dissim_nei]
            for j in dissim_nei_idx:
                adj[i][j] = 0
        else:
            continue
    data.edge_index = dense_to_sparse(adj)[0]

    for i in range(config.train_iter):
        print('itertion:' + str(i + 1))
        #f1_macro, f1_micro, f1_macro_hub,f1_micro_hub,f1_macro_tail,f1_micro_tail  = model_training(data, val_data=data_copy, with_gen=False)
        f1_macro, f1_micro = model_training(data, val_data=data_copy, with_gen=False)
        macro_results.append(f1_macro)
        micro_results.append(f1_micro)


    macro_result = 100 * torch.tensor(macro_results)
    micro_result = 100 * torch.tensor(micro_results)


    macro_f1_avg = macro_result.mean()
    macro_f1_std = macro_result.std()
    micro_f1_avg = micro_result.mean()
    micro_f1_std = micro_result.std()


    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'f1_macro:{macro_f1_avg:.2f}±{macro_f1_std:.2f}'
          f'f1_micro:{micro_f1_avg:.2f}±{micro_f1_std:.2f}')
    #print('average results after' + str(config.train_iter) + 'iteration:')
    #print(f'f1_macro_hub:{f1_macro_avg_hub:.2f}±{f1_macro_std_hub:.2f} '
    #      f'f1_micro_hub:{f1_micro_avg_hub:.2f}±{f1_micro_std_hub:.2f} ')
    #print('average results after' + str(config.train_iter) + 'iteration:')
    #print(f'f1_macro_tail:{f1_macro_avg_tail:.2f}±{f1_macro_std_tail:.2f} '
    #      f'f1_micro_tail:{f1_micro_avg_tail:.2f}±{f1_micro_std_tail:.2f} ')


def adding_edges(data, sample_tail, use_weight=False): # adding_edges只针对于训练集也是因为sample_tail是在训练集选出来的
    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    data_copy = data.clone()
    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    add_threshold = data_params['add_threshold']
    add_topK = data_params['add_topK']

    edge_index = remove_self_loops(data.edge_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=data.num_nodes)
    adj = adj.squeeze().cpu().numpy()

    node_embed = torch.load(node_pretrain_path).to(config.device).detach()
    link_embed = torch.load(link_pretrain_path).to(config.device).detach()
    # new_embed = link_embed.clone()
    # new_embed[high_degree_node] = 0
    prob = decode_low_degree(link_embed[sample_tail], link_embed,
                             node_embed[sample_tail], node_embed)

    index = 0
    for i in sample_tail:
        nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
        nei_idx = index_to_mask(nei_idx, size=data.num_nodes)
        prob[index][nei_idx] = 0
        index += 1

    # 把现在所有有边的概率都置为0
    print("adding neighbor for low_degree_node which score > {}".format(add_threshold))

    prob_np = prob.cpu().numpy()
    origin_node_idx, predict_neighbor = np.where(prob_np >= add_threshold)
    origin_node = np.zeros(origin_node_idx.shape[0], dtype=int)
    for i in range(origin_node_idx.shape[0]):
        origin_node[i] = sample_tail[origin_node_idx[i]]

    new_edge = np.vstack((origin_node, predict_neighbor))
    new_edge = torch.tensor(new_edge, dtype=torch.int, device=config.device)
    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
    data_adding_edges = data.clone()
    macro_results = []
    micro_results = []

    for i in range(config.train_iter):
        print('itertion:' + str(i + 1))
        if use_weight:
            f1_macro, f1_micro = model_training(data, val_data=data_copy, with_gen=False, use_weight=use_weight)
        else:
            #f1_macro, f1_micro, f1_macro_hub,f1_micro_hub,f1_macro_tail,f1_micro_tail = model_training(data, val_data=data_copy, with_gen=False)
            f1_macro, f1_micro = model_training(data, val_data=data_copy, with_gen=False)
        # f1_macro_withgan, f1_micro_withgan = model_training(data, val_data=data_copy, with_gen=True)
        # macro_results.append([f1_macro, f1_macro_withgan])
        # micro_results.append([f1_micro, f1_micro_withgan])
        macro_results.append(f1_macro)
        micro_results.append(f1_micro)


        data = data_adding_edges.clone()

    macro_result = 100 * torch.tensor(macro_results)
    micro_result = 100 * torch.tensor(micro_results)

    f1_macro_avg = macro_result.mean()
    f1_micro_avg = micro_result.mean()
    f1_macro_std = macro_result.std()
    f1_micro_std = micro_result.std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'f1_macro:{f1_macro_avg:.2f}±{f1_macro_std:.2f} '
          f'f1_micro:{f1_micro_avg:.2f}±{f1_micro_std:.2f} ')

    data = data_copy.clone()


def main():
    name = config.dataset
    if config.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./temp/' + name + '/', name=name)
    elif config.dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./temp/' + config.dataset + '/', name=config.dataset)
    elif config.dataset in ['actor']:
        dataset = Actor(root='./temp/actor/')
    else:
        raise "dataset not used"

    parameters = json.load(open('parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    tail_ratio = data_params['tail_ratio']
    hub_times = data_params['hub_times'] # 直接读参数这样好！针对每个数据集都有不同的参数
    data = dataset[0]
    data = data.to(config.device)

    if not config.overall:
        transform = TailNodeSplit(config.split, num_splits=1, num_val=0.1, num_test=0.2) # 如果不是overall的设置就按照这个划分hub和tail
        data = transform(data).to(config.device)

    else: # 如果是overall的设置
        if config.dataset in ['chameleon', 'squirrel', 'actor']:  # 会有10组不同的划分，划分都是48/32/20
            data.train_mask, data.val_mask, data.test_mask = create_masks(data, seed=config.seed)

    print("Train Mask:", data.train_mask.sum().item())
    print("Validation Mask:", data.val_mask.sum().item())
    print("Test Mask:", data.test_mask.sum().item())


    data.y = data.y.to(torch.int64)
    data_copy = data.clone()
    macro_results = []
    micro_results = []

    tail_node = []
    all_node = [i for i in range(data.num_nodes)]
    all_node = torch.tensor(all_node, device=config.device)
    all_train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=config.device)
    all_train_mask[data.val_mask] = False
    all_train_mask[data.test_mask] = False
    all_train_node = all_node[all_train_mask]
    data.all_train_mask = all_train_mask
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    np.random.seed(config.seed) # 这里是获取所有训练节点，那么掩码信息还是在加载器加载了的
    train_id = data.train_id.to(config.device)

    tail_num = round(tail_ratio * all_train_node.size(0)) # 获取尾结点的数目
    sample_num = all_train_node.size(0) - tail_num

    train_sub = subgraph(all_train_mask, data.edge_index)[0]
    train_nx = nx.Graph()
    train_nx.add_nodes_from(all_train_node.cpu().numpy())
    train_nx.add_edges_from(train_sub.t().cpu().numpy())
    train_nx.remove_edges_from(nx.selfloop_edges(train_nx)) # 生产训练子图
    data_pagerank = nx.pagerank(train_nx)
    data_pagerank_l = list(data_pagerank.values())
    data_pagerank_l = np.array(data_pagerank_l)
    avg_pagerank = data_pagerank_l.sum(-1) / data_pagerank_l.shape[0]

    hub_num = np.where(data_pagerank_l >= avg_pagerank * hub_times)[0] # 只对训练集划分hub和samples
    hub_node = np.random.choice(all_train_node.cpu().numpy(),
                               size=hub_num.shape[0], replace=False, p=data_pagerank_l)

    samples = np.random.choice(all_train_node.cpu().numpy(),
                               size=sample_num,
                               replace=False, p=data_pagerank_l)
    for i in all_train_node.cpu().numpy():
        if i not in samples:
            tail_node.append(i)
    sample_tail = np.array(tail_node) # 对训练集选出tail nodes


    for i in range(config.train_iter): # 进行10次训练迭代，如果这么写的话直接就是训练了10次不用自己手动了
        print("iteation:" + str(i + 1))
        f1_macro, f1_micro = model_training(data,val_data=data_copy,with_gen=False)

        data = data_copy.clone()

        macro_results.append(f1_macro)
        micro_results.append(f1_micro)


    macro_result = 100 * torch.tensor(macro_results)
    micro_result = 100 * torch.tensor(micro_results)

    f1_macro_avg = macro_result.mean()
    f1_micro_avg = micro_result.mean()
    f1_macro_std = macro_result.std()
    f1_micro_std = micro_result.std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'f1_macro:{f1_macro_avg:.2f}±{f1_macro_std:.2f} '
          f'f1_micro:{f1_micro_avg:.2f}±{f1_micro_std:.2f} ')
    data = data_copy.clone()
    print("==================================")
    print("training GNN after dropping edges:")
    drop_edges(data, hub_node)
    # data.all_train_mask = all_train_mask
    # data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    data = data_copy.clone()
    print("==================================")
    print("training GNN after adding edges without pagerank weight:")
    adding_edges(data, train_id)
    print("training finish")
    exit(0)


if __name__ == "__main__":
    main()
