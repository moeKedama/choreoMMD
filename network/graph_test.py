import torch
from torch_geometric.data import Data
# from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)  # 01 10 12 21 进行连接
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index)

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index.t().contiguous())  # 作用相同
#
# print(data.keys)
# print(data['x'])
# for key, item in data:
#     print(f'{key} found in data')
# print('edge_attr' in data)
# print(data.num_nodes)
# print(data.num_edges)
# print(data.num_node_features)
# print(data.has_isolated_nodes())
# print(data.has_self_loops())
# print(data.is_directed())

device = torch.device('cuda')
# data = data.to(device)

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# print(len(dataset))
#
# print(dataset.num_classes)
#
# print(dataset.num_node_features)
#
# data = dataset[0]
#
# print(data.is_undirected())
#
# train_dataset = dataset[:540]
#
# test_dataset = dataset[540:]

# dataset = dataset.shuffle()

# for batch in loader:
#     print(batch)
#     print(batch.num_graphs)

# data = next(iter(loader))
# print(data.num_graphs)
# x = scatter(data.x, data.batch, dim=0, reduce='mean')
# print(x.size())

# transform
# from torch_geometric.datasets import ShapeNet
#
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
#
# dataset[0]

# import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet
#
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6))
#
# dataset[0]

# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6),
#                     transform=T.RandomJitter(0.01))
#
# dataset[0]


dataset = Planetoid(root='/tmp/Cora', name='Cora')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
