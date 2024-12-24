import json
import math
import os.path
import time
from typing import Union
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.datasets import AttributedGraphDataset, Planetoid
from torch_geometric.datasets import Twitch, WikipediaNetwork, Actor, WikiCS, SNAPDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch import optim
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_networkx, is_undirected
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from torch.nn.functional import cosine_similarity
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import index_to_mask
import numpy as np
from models import NeighGen
from models import Discriminator
from models import Net
from torch.autograd import Variable
import config
from utils import get_neighbor_feature4link, decode_low_degree, get_nei, RandomLinkSplit
from utils import SocialNetwork


def train(train_data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    neg_edge_index = negative_sampling(train_data.edge_label_index,
                                       num_nodes=train_data.num_nodes,
                                       num_neg_samples=train_data.edge_label_index.size(1))

    z = model.encode(train_data.x, train_data.edge_label_index)
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1
    )

    edge_label = torch.cat(
        [train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))],
        dim=0
    )

    out = model.decode(z, edge_label_index).view(-1)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = criterion(out, edge_label)
    loss.backward()

    optimizer.step()

    return loss, z


def test(data, model):
    model.eval()

    z = model.encode(data.x, data.edge_label_index)

    edge_label = torch.ones(data.edge_label_index.size(1))
    neg_edges = negative_sampling(data.edge_label_index, num_nodes=data.num_nodes,
                                  num_neg_samples=data.edge_label_index.size(1))
    edge_index = torch.cat([data.edge_label_index, neg_edges], dim=-1)
    neg_label = torch.zeros(neg_edges.size(1))
    edge_label = torch.cat([edge_label, neg_label])

    out = model.decode(z, edge_index).view(-1).sigmoid()
    auc = roc_auc_score(edge_label.cpu().numpy(), out.cpu().detach().numpy())

    return auc


def greedy_loss(pred_feats, true_feat):
    loss = torch.zeros_like(pred_feats).to(config.device)
    pred_len = pred_feats.size(0)
    for i in range(pred_len):
        for j in range(config.num_pred):
            loss[i][j] += F.mse_loss(pred_feats[i][j].unsqueeze(0).float(),
                                     true_feat[i].unsqueeze(0).float()).squeeze(0)

    return loss.mean()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx - yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def train_gen(data, sample_node=None):
    min_loss_gen = 10
    if config.use_fullneifeat:
        best_gen_feat = {}
        z = {}
        if sample_node is None:
            sample_tail, nei_feat, _, nei_num = get_neighbor_feature4link(data, use_fullnei=True)
        else:
            sample_tail, nei_feat, _, nei_num = get_neighbor_feature4link(data, sample_node, use_fullnei=True)
        feature_size = data.x.size(1)
        gen = NeighGen(feature_size, data, sample_tail, nei_feat).to(config.device)
        optimizer_gen = optim.Adam(gen.parameters(),
                                   lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.gan_epoch):
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
            if gen_loss.item() < min_loss_gen:
                min_loss_gen = gen_loss.item()
                best_gen_feat = gen_feat
            optimizer_gen.step()
            gen.eval()
        print(f'min_loss_gen:{min_loss_gen:.4f}')
        print("Training generator finished!")

        return best_gen_feat, sample_tail

    else:
        if sample_node is None:
            sample_tail, nei_feat, _ = get_neighbor_feature4link(data)
        else:
            sample_tail, nei_feat, _ = get_neighbor_feature4link(data, sample_node)

        feature_size = data.x.size(1)
        gen = NeighGen(feature_size, data, sample_tail, nei_feat).to(config.device)
        dis = Discriminator(feat_shape=feature_size).to(config.device)

        optimizer_gen = optim.Adam(gen.parameters(),
                                   lr=config.lr, weight_decay=config.weight_decay)

        optimizer_dis = optim.Adam(dis.parameters(),
                                   lr=config.lr, weight_decay=config.weight_decay)

        dis_real_label = Variable(torch.ones(sample_tail.shape[0])).to(config.device)
        real_label = Variable(torch.ones(sample_tail.shape[0] * config.num_pred)).to(config.device)
        fake_label = Variable(torch.zeros(sample_tail.shape[0] * config.num_pred)).to(config.device)

        gen_feat = []
        best_gen_feat = []
        min_loss_gen = 10
        print("Training neighbor_generator...")
        for epoch in range(config.gan_epoch):
            t = time.time()
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (sample_tail.shape[0], data.x.size(1))))).to(
                config.device)
            if config.use_dis is True:
                dis.train()
                gen.train()
                optimizer_dis.zero_grad()
                optimizer_gen.zero_grad()
                real_out = dis(nei_feat)
                real_out = real_out.squeeze()
                d_real_loss = F.binary_cross_entropy(real_out, dis_real_label)
                gen_feat, _ = gen(z)
                output_feat = gen_feat.view(-1, data.x.size(1))

                fake_out = dis(output_feat)
                fake_out = fake_out.squeeze()
                d_fake_loss = F.binary_cross_entropy(fake_out, fake_label)

                loss_dis = d_fake_loss + d_real_loss
                loss_dis.backward()
                optimizer_dis.step()

                gen_feat, _ = gen(z)
                gen_feat = gen_feat.view(-1, data.x.size(1))
                output = dis(gen_feat)
                output = output.squeeze()
                loss_gen = F.binary_cross_entropy(output, real_label)

                if loss_gen.item() < min_loss_gen:
                    min_loss_gen = loss_gen.item()
                    best_gen_feat = gen_feat

                loss_gen.backward()
                optimizer_gen.step()
                gen.eval()
                dis.eval()
                """
                print(f'time:{time.time() - t},'
                      f'gen_epoch:{epoch},'
                      f'loss_gen:{loss_gen:.4f},'
                      f'loss_dis:{loss_dis:.4f},'
                      f'd_real_loss:{d_real_loss:.4f},'
                      f'd_fake_loss:{d_fake_loss:.4f}')
                """
            else:
                gen.train()
                optimizer_gen.zero_grad()

                gen_feat, label_predict = gen(z)
                label_predict = label_predict.squeeze()

                feat_loss = F.mse_loss(gen_feat, nei_feat)
                true_label = torch.ones(data.num_nodes, device=config.device)
                gen_label = torch.zeros(len(sample_tail), device=config.device)
                label = torch.concat([true_label, gen_label], dim=0)
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
        print("Training generator finished!")

        return best_gen_feat, sample_tail


def model_training(train_data, val_data, test_data, with_gen=False, sample_node=None):
    save_path = './saved_model/' + config.dataset + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_data_copy = train_data.clone()
    best_val_auc = 0
    cur_step = 0
    min_loss = 10
    if with_gen:
        model_withgan = Net(train_data.num_features, config.embed_dim,
                            dropout=config.dropout).to(config.device)
        optimizer = optim.Adam(model_withgan.parameters(), lr=config.lr)
        if sample_node is None:
            gen_feat, sample_tail = train_gen(train_data)

        else:
            gen_feat, sample_tail = train_gen(train_data, sample_node)
        if isinstance(gen_feat, dict):
            edge_1 = []
            genfeat = []
            for i in gen_feat.keys():
                if gen_feat[i].size(0) == train_data.x.size(1):
                    edge_1.append(i)
                    genfeat.append(gen_feat[i].cpu().detach().numpy().tolist())
                else:
                    for j in range(gen_feat[i].size(0)):
                        edge_1.append(i)
                        genfeat.append(gen_feat[i][j].cpu().detach().numpy().tolist())
            edge_1 = np.array(edge_1, dtype=int)
            edge_2 = np.array([i for i in range(train_data.num_nodes, train_data.num_nodes + len(edge_1))])
            edge_gen = np.vstack([edge_1, edge_2])
            new_edge = torch.tensor(edge_gen, dtype=torch.int, device=config.device)
            train_data.edge_label_index = torch.hstack((train_data.edge_label_index, new_edge))
            train_data.edge_index = torch.hstack((train_data.edge_index, new_edge))
            genfeat = np.array(genfeat)
            genfeat = torch.tensor(genfeat, device=config.device)
            train_data.x = torch.vstack([train_data.x, genfeat]).to(torch.float)
            new_edge_label = torch.ones(new_edge.size(1), dtype=torch.int, device=config.device)
            train_data.edge_label = torch.cat([train_data.edge_label, new_edge_label])
        else:
            edge_1 = []
            for i in range(len(sample_tail)):
                for j in range(config.num_pred):
                    edge_1.append(sample_tail[i])

            edge_1 = np.array(edge_1, dtype=int)
            edge_2 = np.array([i for i in range(train_data.num_nodes,
                                                train_data.num_nodes + len(sample_tail) * config.num_pred)])
            edge_generate = np.vstack((edge_1, edge_2))
            train_data.num_nodes += len(sample_tail) * config.num_pred

            new_edge = torch.tensor(edge_generate, dtype=torch.int, device=config.device)
            train_data.edge_label_index = torch.hstack((train_data.edge_label_index, new_edge))
            train_data.edge_index = torch.hstack((train_data.edge_index, new_edge))

            train_data.x = torch.vstack((train_data.x, gen_feat.view(-1, train_data.x.size(1)).detach()))
            new_edge_label = np.ones(len(sample_tail) * config.num_pred, dtype=int)
            new_edge_label = torch.tensor(new_edge_label, device=config.device)
            train_data.edge_label = torch.cat([train_data.edge_label, new_edge_label])

        for epochs in range(config.epoch):
            loss, embedding = train(train_data=train_data, model=model_withgan, optimizer=optimizer)
            val_auc = test(val_data, model_withgan)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epoch_early_stop = epochs
                torch.save(model_withgan, os.path.join(save_path, 'model_withgan.pt'))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == config.patience:
                    early_stop = 'Early Stopping at epoch {:d}'.format(epochs)
                    print(early_stop)
                    break

            if loss.item() < min_loss:
                min_loss = loss.item()

            """
            print(f'epochs:{epochs + 1},'
                  f'loss:{loss:.4f},'
                  f'val_auc:{val_auc:.4f},'
                  f'test_auc:{test_auc:.4f}')

            with open(result_path, 'a') as f:
                f.writelines(f'epochs:{epochs + 1}, '
                             f'loss:{loss:.4f}, '
                             f'val_auc:{val_auc:.4f}, '
                             f'test_auc:{test_auc:.4f}')
                f.close()
            """

        embed_model = torch.load(os.path.join(save_path, 'model_withgan.pt'))
        test_auc = test(test_data, embed_model)
        print(f'min_loss:{min_loss:.4f}, '
              f'test_auc_withgan:{test_auc:.4f}')

        train_data = train_data_copy.clone()
        return best_val_auc, test_auc

    else:
        model = Net(train_data.num_features, config.embed_dim,
                    config.dropout).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            loss, _ = train(train_data=train_data, model=model, optimizer=optimizer)
            val_auc = test(val_data, model)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
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
            """
            print(f'epochs:{epoch + 1}, '
                  f'loss:{loss:.4f}, '
                  f'val_auc:{val_auc:.4f}, '
                  f'test_auc:{test_auc:.4f}')
            """
        embed_model = torch.load(os.path.join(save_path, 'model.pt'))
        test_auc = test(test_data, embed_model)
        print(f'min_loss:{min_loss:.4f}, '
              f'test_auc:{test_auc:.4f}')

        return best_val_auc, test_auc


def drop_edges(train_data, val_data, test_data, hub_node):
    data_copy = train_data.clone()
    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    parameters = json.load(open('lp_parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    drop_thres = data_params['drop_threshold']
    node_embed = torch.load(node_pretrain_path).to(config.device)
    link_embed = torch.load(link_pretrain_path).to(config.device)
    val_results = []
    test_results = []

    edge_index = remove_self_loops(train_data.edge_label_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=train_data.num_nodes).squeeze()
    for i in hub_node:
        hub_nei_idx, hub_nei_mask = get_nei(train_data, i)
        hub_prob = decode_low_degree(link_embed[i], link_embed[hub_nei_mask],
                                     node_embed[i], node_embed[hub_nei_mask])
        hub_prob_array = hub_prob.cpu().numpy()
        if np.any(hub_prob_array < config.dissim):
            dissim_nei = np.where(hub_prob_array < drop_thres)[0]
            dissim_nei_idx = hub_nei_idx[dissim_nei]
            for j in dissim_nei_idx:
                adj[i][j] = 0
        else:
            continue
    train_data.edge_label_index = dense_to_sparse(adj)[0]
    edge_label = torch.ones(train_data.edge_label_index.size(1), device=config.device)
    train_data.edge_label = edge_label
    for i in range(config.train_iter):
        print('itertion:' + str(i + 1))
        val_auc, test_auc = model_training(train_data, val_data, test_data, with_gen=False)
        val_results.append(val_auc)
        test_results.append(test_auc)
    test_result = 100 * torch.tensor(test_results)
    test_auc_avg = test_result.mean()
    test_auc_std = test_result.std()
    print('average results after'+str(config.train_iter)+'iteration:')
    print(f'test_auc_avg:{test_auc_avg:.2f}±{test_auc_std:.2f}')


def adding_edges(train_data, val_data, test_data, tail_node):
    train_data_copy = train_data.clone()
    parameters = json.load(open('lp_parameters.json', 'r'))
    data_params = parameters[config.dataset][config.base_model]
    add_threshold = data_params['add_threshold']
    add_topK = data_params['add_topK']

    edge_index = remove_self_loops(train_data.edge_label_index)
    adj = to_dense_adj(edge_index[0], max_num_nodes=train_data.num_nodes).cpu().numpy()
    adj = adj.reshape(train_data.num_nodes, train_data.num_nodes)

    node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'
    node_embed = torch.load(node_pretrain_path).to(config.device)
    link_embed = torch.load(link_pretrain_path).to(config.device)
    # new_embed = link_embed.clone()
    # new_embed[high_degree_node] = 0
    prob = decode_low_degree(link_embed[tail_node], link_embed,
                             node_embed[tail_node], node_embed)
    # low_degree_prob = prob[low_degree_node]
    index = 0
    for i in tail_node:
        nei_idx = torch.tensor(np.where(adj[:, i] != 0)[0])
        nei_idx = index_to_mask(nei_idx, size=train_data.num_nodes)
        prob[index][nei_idx] = 0.0
        index = index + 1

    print("adding neighbor for low_degree_node which score > {}".format(add_threshold))
    prob_np = prob.cpu().numpy()
    origin_node_idx, predict_neighbor = np.where(prob_np >= add_threshold)
    origin_node = np.zeros(origin_node_idx.shape[0], dtype=np.int)
    for i in range(origin_node_idx.shape[0]):
        origin_node[i] = tail_node[origin_node_idx[i]]

    new_edge = np.vstack((origin_node, predict_neighbor))
    new_edge = torch.tensor(new_edge, dtype=torch.int, device=config.device)
    train_data.edge_label_index = torch.cat([train_data.edge_label_index, new_edge], dim=1)
    train_data.edge_index = torch.cat([train_data.edge_index, new_edge], dim=1)
    new_edge_label = torch.ones(new_edge.size(1), dtype=torch.float, device=config.device)
    train_data.edge_label = torch.cat([train_data.edge_label, new_edge_label])

    train_data_adding_edges = train_data.clone()
    val_results = []
    test_results = []
    for i in range(config.train_iter):
        print('itertion:' + str(i + 1))
        val_auc, test_auc = model_training(train_data, val_data, test_data)
        val_auc_withgan, test_auc_withgan = model_training(train_data, val_data,
                                                           test_data, with_gen=True)
        val_results.append([val_auc, val_auc_withgan])
        test_results.append([test_auc, test_auc_withgan])
        train_data = train_data_adding_edges.clone()
    test_result = 100 * torch.tensor(test_results)
    test_auc_avg = test_result[:, 0].mean()
    test_auc_std = test_result[:, 0].std()
    test_auc_withgan_avg = test_result[:, 1].mean()
    test_auc_withgan_std = test_result[:, 1].std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'test_auc_avg:{test_auc_avg:.2f}±{test_auc_std:.2f}')
    print(f'test_auc_withgan_avg:{test_auc_withgan_avg:.2f}±{test_auc_withgan_std:.2f}')
    train_data = train_data_copy.clone()

    print("traing GNN with adding {} predict edges which score is max:".format(add_topK))
    for i in range(add_topK):
        predict_neighbor = torch.argmax(prob, dim=1).cpu().numpy()
        new_edge = np.vstack((tail_node, predict_neighbor))
        new_edge = torch.tensor(new_edge, dtype=torch.int).to(config.device)
        train_data.edge_label_index = torch.cat([train_data.edge_label_index, new_edge], dim=1)
        train_data.edge_index = torch.cat([train_data.edge_index, new_edge], dim=1)
        new_edge_label = torch.ones(new_edge.size(1), dtype=torch.float, device=config.device)
        train_data.edge_label = torch.cat([train_data.edge_label, new_edge_label])
        for j in range(predict_neighbor.shape[0]):
            prob[j][predict_neighbor[j]] = 0.0

    train_data_adding_edges = train_data.clone()
    val_results = []
    test_results = []
    for i in range(config.train_iter):
        print('itertion:' + str(i + 1))
        val_auc, test_auc = model_training(train_data, val_data, test_data, with_gen=False)
        val_auc_withgan, test_auc_withgan = model_training(train_data, val_data,
                                                           test_data, with_gen=True)
        val_results.append([val_auc, val_auc_withgan])
        test_results.append([test_auc, test_auc_withgan])
        train_data = train_data_adding_edges.clone()
    test_result = 100 * torch.tensor(test_results)
    test_auc_avg = test_result[:, 0].mean()
    test_auc_std = test_result[:, 0].std()
    test_auc_withgan_avg = test_result[:, 1].mean()
    test_auc_withgan_std = test_result[:, 1].std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'test_auc_avg:{test_auc_avg:.2f}±{test_auc_std:.2f}')
    print(f'test_auc_withgan_avg:{test_auc_withgan_avg:.2f}±{test_auc_withgan_std:.2f}')
    train_data = train_data_copy.clone()


def main():
    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, add_negative_train_samples=False)

    name = config.dataset
    if config.dataset in ['facebook_ego', 'twitter_ego', 'gplus_ego']:
        dataset = SocialNetwork(root='./temp/' + name + '/', name=name)
    elif config.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./temp/' + name + '/', name=name)
    elif config.dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./temp/' + config.dataset + '/', name=config.dataset)
    elif config.dataset in ['actor']:
        dataset = Actor(root='./temp/actor/')
    else:
        raise "dataset not used!"
    data = dataset[0]
    data.num_nodes = data.y.size(0)
    train_data, val_data, test_data = transform(data)
    train_data = train_data.to(config.device)
    val_data = val_data.to(config.device)
    test_data = test_data.to(config.device)
    train_data_copy = train_data.clone()

    tail_node = []
    all_node = [i for i in range(data.num_nodes)]
    all_node = torch.tensor(all_node, device=config.device)
    tail_num = round(config.tail_ratio * all_node.size(0))
    sample_num = train_data.num_nodes - tail_num
    train_nx = to_networkx(train_data, to_undirected=True)
    train_nx.remove_edges_from(nx.selfloop_edges(train_nx))
    data_pagerank = nx.pagerank(train_nx)
    data_pagerank_l = list(data_pagerank.values())
    data_pagerank_l = np.array(data_pagerank_l)
    avg_pagerank = data_pagerank_l.sum(-1) / data_pagerank_l.shape[0]
    hub_num = np.where(data_pagerank_l >= avg_pagerank * 2)[0]
    hub_node = np.random.choice(all_node.cpu().numpy(),
                                size=hub_num.shape[0], replace=False, p=data_pagerank_l)
    samples = np.random.choice(all_node.cpu().numpy(),
                               size=sample_num,
                               replace=False, p=data_pagerank_l)
    for i in all_node.cpu().numpy():
        if i not in samples:
            tail_node.append(i)
    sample_tail = np.array(tail_node)
    val_results = []
    test_results = []
    for i in range(config.train_iter):
        print("iteation:" + str(i + 1))
        val_auc, test_auc = model_training(train_data, val_data, test_data)
        val_auc_withgan, test_auc_withgan = model_training(train_data, val_data,
                                                           test_data, with_gen=True)
        val_results.append([val_auc, val_auc_withgan])
        test_results.append([test_auc, test_auc_withgan])
        train_data = train_data_copy.clone()
    test_result = 100 * torch.tensor(test_results)
    test_auc_avg = test_result[:, 0].mean()
    test_auc_std = test_result[:, 0].std()
    test_auc_withgan_avg = test_result[:, 1].mean()
    test_auc_withgan_std = test_result[:, 1].std()
    print('average results after' + str(config.train_iter) + 'iteration:')
    print(f'test_auc_avg:{test_auc_avg:.2f}±{test_auc_std:.2f}')
    print(f'test_auc_withgan_avg:{test_auc_withgan_avg:.2f}±{test_auc_withgan_std:.2f}')
    train_data = train_data_copy.clone()
    # print("training GNN after adding edges and without dropping edges:")
    # train_data = train_data_copy.clone()
    print("training GNN after dropping edges:")
    drop_edges(train_data, val_data, test_data, hub_node)
    print("training GNN after adding edges:")
    adding_edges(train_data, val_data, test_data, sample_tail)
    print("training finish")


if __name__ == '__main__':
    main()
