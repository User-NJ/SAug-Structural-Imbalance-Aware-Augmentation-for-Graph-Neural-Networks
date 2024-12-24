import json
import math
import os
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch.autograd import Variable
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork, Actor
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops, mask_to_index
from torch_geometric.utils import index_to_mask
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_dense_adj

import config
from models import Discriminator
from models import NeighGen
from models import Net
from utils import get_nei_feature4node, \
    decode_low_degree, get_nei, TailNodeSplit, SocialNetwork, training_scheduler, group_pagerank, rank_group_pagerank, \
    sort_training_nodes


def cl_training(data, sorted_node, lam=config.lam, T_grow=config.T_grow):
    macro_results = []
    micro_results = []
    data_copy = data.clone()
    save_path = './saved_model/' + config.dataset + '/'
    loss_function = torch.nn.CrossEntropyLoss()
    labels = data.y.cpu().numpy()
    num_classes = int(np.max(labels) + 1)
    for i in range(config.train_iter):
        cur_step = 0
        min_loss = float("inf")
        best_val_f1 = 0
        model = Net(data.num_features, config.embed_dim,
                    config.dropout, num_classes=num_classes).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            # cl_sample_ratio = mu * (1 - math.cos(epoch / config.epoch * (math.pi / 2)))
            ratio = training_scheduler(lam, epoch, T_grow, config.scheduler)
            # new_edges_trans = np.delete(new_edges_trans, adding_idx, axis=0)
            # index = np.arange(new_edges_trans.shape[0])
            train_size = int(ratio * sorted_node.shape[0])
            training_subset = sorted_node[:train_size]
            model.train()
            optimizer.zero_grad()
            h = model.node_classify(data)
            loss = loss_function(h[training_subset], data.y[training_subset])
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
                """
                if loss.item() < min_loss:
                    min_loss = loss.item()
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model, os.path.join(save_path, 'model.pt'))
        embed_model = torch.load(os.path.join(save_path, 'model.pt'))
        test_h = embed_model.node_classify(data)[data.test_mask]
        _, test_predict_y = test_h.max(dim=1)
        # _, test_predict_y = best_logits[data.test_mask].max(dim=1)
        test_f1_macro = f1_score(test_predict_y.cpu(), data.y[data.test_mask].cpu(), average="macro")
        test_f1_micro = f1_score(test_predict_y.cpu(), data.y[data.test_mask].cpu(), average="micro")

        macro_results.append(test_f1_macro)
        micro_results.append(test_f1_micro)
        data = data_copy.clone()
    micro_result = 100 * torch.tensor(micro_results)
    f1_micro_avg = micro_result.mean()
    return f1_micro_avg


'''def lam_T_tuning(data, sorted_node):
    lam = np.arange(0.1, 1, 0.1)
    T_grow = np.arange(50, 400, 50)
    lam_T_sens = []
    for T in T_grow:
        for lam_0 in lam:
            lam_T_sens.append((T, lam_0, cl_training(data, sorted_node, lam_0, T)))
    return lam_T_sens'''

def alpha_tuning(data):
    alpha = np.arange(0.1, 1, 0.05)
    res = []
    for a in alpha:
        sorted_nodes = sort_training_nodes(data, a)
        res.append([a, cl_training(data, sorted_nodes)])
    return res


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

    save_path = "./saves/actor_alpha.csv"

    data = dataset[0]
    data = data.to(config.device)
    if not config.overall:
        transform = TailNodeSplit(config.split, num_splits=1, num_val=0.1, num_test=0.2)
        data = transform(data).to(config.device)

    else:
        if config.dataset in ['chameleon', 'squirrel', 'actor']:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
    data.y = data.y.to(torch.int64)
    all_train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=config.device)
    all_train_mask[data.val_mask] = False
    all_train_mask[data.test_mask] = False
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    data.all_train_mask = all_train_mask
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    sorted_trainset = sort_training_nodes(data)

    print("tuning for alpha")
    alpha_res = alpha_tuning(data)
    alpha_res = np.array(alpha_res)
    alpha_res = pd.DataFrame(alpha_res, columns=["alpha", "Micro-F1(%)"])
    if not os.path.exists(save_path):
        alpha_res.to_csv(save_path, index=False, header=True, sep=",")
        print("save finish")

    """
    lam_T_res = lam_T_tuning(data, sorted_trainset)
    lam_T_res = np.array(lam_T_res)
    lam_T_res = pd.DataFrame(lam_T_res, columns=["T_grow", "lam_0", "res"])

    if not os.path.exists(save_path):
        lam_T_res.to_csv(save_path, index=False, header=True, sep=",")
        print("save finish")
    """


if __name__ == '__main__':
    main()
