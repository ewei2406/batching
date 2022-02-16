import torch
import math

def getSplits(graph, splitSize, numSplits):
    indices = torch.randperm(graph.adj.shape[0])

    res = []

    factor = (graph.adj.shape[0] - splitSize) / numSplits
    for i in range(0, numSplits):
        lowerBound = math.floor(factor*i)
        upperBound = math.floor(factor*i + splitSize)
        graphIdx = indices[torch.arange(lowerBound, upperBound)]

        res.append(graph.getSubgraph(graphIdx))

    return res

class Graph:
    def __init__(self, adj, labels, features, idx_train, idx_val, idx_test, nodeid=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        if nodeid != None:
            self.nodeid = nodeid
        else:
            self.nodeid = torch.arange(0,self.adj.shape[0])

    def __repr__(self):
        return f"<Graph {self.adj.shape[0]}x{self.adj.shape[1]}>"

    def summarize(self, name=""):
        print()
        print(f'==== Dataset Summary: {name} ====')
        print(f'adj shape: {list(self.adj.shape)}')
        print(f'feature shape: {list(self.features.shape)}')
        print(f'num labels: {self.labels.max().item()+1}')
        print(
            f'train|val|test: {self.idx_train.sum()}|{self.idx_val.sum()}|{self.idx_test.sum()}')
    
    def split(self, nsplits):
        indices = torch.zeros(10, dtype=torch.bool)
        return None

    def numEdges(self):
        return self.adj.sum() / 2

    def numNodes(self):
        return self.adj.shape[0]

    def getSample(self, size):
        indices = (torch.bernoulli(torch.empty(1, size)[0].uniform_(0,1))) > 0.5
        maskA = indices.nonzero().t()[0]
        maskB = (~indices).nonzero().t()[0]

        return maskA, maskB

    def getSubgraph(self, indices):
        return Graph(
            adj=self.adj[indices].t()[indices].t(),
            features=self.features[indices],
            labels=self.labels[indices],
            idx_train=self.idx_train[indices],
            idx_val=self.idx_val[indices],
            idx_test=self.idx_test[indices],
            nodeid=indices
        )
    