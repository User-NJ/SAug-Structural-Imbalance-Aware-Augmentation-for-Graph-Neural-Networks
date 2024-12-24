from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import AttributedGraphDataset, Planetoid, Airports, WebKB
from torch_geometric.datasets import Twitch, WikiCS, Actor, WikipediaNetwork
from sklearn.metrics import f1_score
from torch_geometric.utils import degree, negative_sampling
import torch
import numpy as np
from models import Net
from torch.autograd import Variable
from torch_geometric.utils import index_to_mask, subgraph

from node_classification import create_masks
from utils import TailNodeSplit, SocialNetwork
import config

def train_lp(train_data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    neg_edge_index = negative_sampling(train_data.edge_index,
                                       num_nodes=train_data.num_nodes,
                                       num_neg_samples=train_data.edge_index.size(1))

    z = model.encode(train_data.x, train_data.edge_index)
    edge_label_index = torch.cat(
        [train_data.edge_index, neg_edge_index],
        dim=-1
    )
    pos_label = torch.ones(train_data.edge_index.size(1), device=config.device)
    neg_label = torch.zeros(neg_edge_index.size(1), device=config.device)
    edge_label = torch.cat([pos_label, neg_label], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = criterion(out, edge_label)
    loss.backward()

    optimizer.step()

    return loss, z


name = config.dataset
if config.dataset in ['facebook_ego', 'twitter_ego', 'youtube', 'gplus_ego']:
    dataset = SocialNetwork(root='./temp/' + name + '/', name=name)
elif config.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='./temp/' + name + '/', name=name)
elif config.dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root='./temp/' + config.dataset + '/', name=config.dataset)
elif config.dataset in ['actor']:
    dataset = Actor(root='./temp/actor/')
else:
    raise "dataset not used"

transform = TailNodeSplit("pagerank", num_splits=1, num_val=0.1, num_test=0.2)
data = dataset[0]
if config.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    data = data.to(config.device)
if config.dataset in ['chameleon', 'squirrel', 'actor']:
    data.train_mask, data.val_mask, data.test_mask = create_masks(data, seed=config.seed)
data.y = data.y.squeeze().to(torch.int64)
labels = data.y.cpu().numpy()

num_classes = int(np.max(labels) + 1)

all_node = np.array([i for i in range(data.num_nodes)])
all_node = torch.tensor(all_node, device=config.device)
test_node = all_node[data.test_mask]
test_sub = subgraph(data.test_mask, data.edge_index, num_nodes=data.num_nodes)[0]

node_pretrain_path = './pretrain/node-classify-embedding/' + config.base_model + '/' + config.dataset + '.pt'
link_pretrain_path = './pretrain/link-predict-embedding/' + config.base_model + '/' + config.dataset + '.pt'

# pretrain for node_classification embedding
if config.dataset in ['facebook', 'twitter']:
    model4classify = Net(data.num_features, config.embed_dim,
                         config.dropout, num_classes=num_classes,
                         multilabel=True).to(config.device)
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
else:
    model4classify = Net(data.num_features, config.embed_dim,
                         config.dropout, num_classes=num_classes).to(config.device)
    loss_function = torch.nn.CrossEntropyLoss()

opti = torch.optim.Adam(model4classify.parameters(),
                        lr=config.lr, weight_decay=config.weight_decay)
min_loss = 10
max_f1 = 0
best_embed = None
for epoch in range(config.epoch):
    model4classify.train()
    opti.zero_grad()
    out = model4classify.node_classify(data)
    _, pred = out.max(dim=1)
    loss = loss_function(out, data.y)
    loss.backward()
    opti.step()
    model4classify.eval()
    f1 = f1_score(labels, pred.detach().cpu(), average='micro')
    if min_loss > loss:
        min_loss = loss

    if max_f1 < f1:
        max_f1 = f1
        best_embed = out
    print(f'epoch:{epoch + 1}'
          f'loss:{loss:.4f}'
          f'f1:{f1:.4f}')

print(f'min_loss:{min_loss}\n'
      f'max_f1:{max_f1}\n'
      f'best_embed:{best_embed}')

torch.save(best_embed, f=node_pretrain_path)
print("pretrain node-classifier finish")

label_sim = -best_embed @ best_embed.t()

# pretrain for link prediction
model4link = Net(data.num_features, config.embed_dim,
                 dropout=config.dropout).to(config.device)
opti4link = torch.optim.Adam(model4link.parameters(),
                             lr=config.lr, weight_decay=config.weight_decay)

best_val_auc = best_test_auc = 0
min_loss = 10
best_embed4link = None

for epoch2 in range(config.epoch):
    loss, embedding = train_lp(data, model4link, opti4link)

    if loss.item() < min_loss:
        min_loss = loss.item()
        best_embed4link = embedding
    print(f'epoch:{epoch2 + 1}'
          f'loss:{loss:.4f}')

torch.save(best_embed4link, f=link_pretrain_path)
