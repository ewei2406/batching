import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from tools.dataset import Dataset
from tools.loadGraph import loadGraph
from tools.GCN import GCN
from tools import metrics
from tools import utils

import sample

################################################
# Args
################################################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--reg_epochs', type=int, default=50, help='Epochs to train models')
parser.add_argument('--numSplits', type=int, default=4, help='Number of splits')
parser.add_argument('--splitSize', type=int, default=0.75, help='Size of splits')

parser.add_argument('--dataset', type=str, default='cora', help='dataset')
args = parser.parse_args()

################################################
# Environment
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

print('==== Environment ====')
print(f'  torch version: {torch.__version__}')
print(f'  device: {device}')
print(f'  torch seed: {args.seed}')

################################################
# Load Data
################################################

graphData = sample.Graph(*loadGraph('./datasets', args.dataset, 'gcn', args.seed, device))
graphData.summarize()

################################################
# Baseline
################################################

baseline = GCN(
    input_features=graphData.features.shape[1],
    output_classes=graphData.labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name=f"baseline"
)

baseline.fit(graphData, args.reg_epochs)

################################################
# Batched
################################################

batchModel = GCN(
    input_features=graphData.features.shape[1],
    output_classes=graphData.labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name=f"batch"
)

splits = sample.getSplits(graphData, args.splitSize * graphData.numNodes(), args.numSplits)

for epoch in range(args.reg_epochs // args.numSplits):
    for graph in splits:
        loss = batchModel.train1epoch(graph)
        print(f"LOSS: {loss:.2f}     \r", end="")

surrogate = GCN(
    input_features=graphData.features.shape[1],
    output_classes=graphData.labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name=f"surrogate"
)

surrogate.fit(graphData, args.reg_epochs)

for epoch in range(10):
    surrogate.eval()

################################################
# Evaluation
################################################

pred = baseline(graphData.features, graphData.adj)
r = metrics.acc(pred, graphData.labels)
print(f"BASELINE: {r:0.4f}")
pred = batchModel(graphData.features, graphData.adj)
r = metrics.acc(pred, graphData.labels)
print(f"BATCHES: {r:0.4f}")



# indices = (torch.bernoulli(torch.empty(1, adj.shape[0])[0].uniform_(0,1))) > 0.5
# maskA = indices.nonzero().t()[0]
# maskB = (~indices).nonzero().t()[0]

# tensor1 = adj[maskA].t()[maskA].t()
# print(tensor1.shape)

# tensor3 = adj[maskB].t()[maskA].t()
# print(tensor3.shape)



