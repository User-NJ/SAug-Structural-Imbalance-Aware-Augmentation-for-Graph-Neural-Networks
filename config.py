import torch

# 设置
seed = 2023 # 随机种子
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
base_model = 'GCN'  # GCN, GraphSAGE, GAT #backbone模型
dataset = 'Cora'  # Cora, CiteSeer,PubMed, chameleon, squirrel, actor
# #train的时候的默认数据集
heads = 3
hidden_dim = 32
embed_dim = 16
steps = 10
lr = 0.01
weight_decay = 5e-4
gan_epoch = 200
epoch = 500
patience = 300
train_iter = 10
num_pred = 1
drop_hidden = 16
dropout = 0.5
hidden_portion = 0.5
eval_step = 1
gamma = -0.0
zeta = 1.01
eps = 1e-7
lambda1 = 0.01

hub_node_ratio = 0.1
tail_ratio = 0.3
dissim = 0.1
add_neighbor_num = 3
add_threshold = 0.95
mu = 0.5
prune_epoch = 20
prune_ratio = 0.05
T_grow = 300
lam = 0.4
split = "pagerank" # pagerank,degree
scheduler = "geom"

threshold = True
overall = True
use_dis = True
use_fullneifeat = False
