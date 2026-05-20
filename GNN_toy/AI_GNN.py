import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import random_split

# =========================
# Dataset (your class)
# =========================
from torch_geometric.data import InMemoryDataset

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = None  # <-- you already handle filling this

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

# =========================
# Load dataset
# =========================
root = '~/GNN_graphs/second'
dataset = MyOwnDataset(root)

# =========================
# Train/Test split (80/20)
# =========================
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =========================
# RMSE Loss
# =========================
def rmse_loss(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))

# =========================
# GNN Model (3 conv layers)
# =========================
class GNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.lin = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x.view(-1)

# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# Training function
# =========================
def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        loss = rmse_loss(out, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# Testing function
# =========================
def test():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)

            loss = rmse_loss(out, batch.y.view(-1))

            total_loss += loss.item()

    return total_loss / len(test_loader)

# =========================
# Epoch loop
# =========================
num_epochs = 50

for epoch in range(0, num_epochs ):
    train_loss = train()
    test_loss = test()

    print(f"Epoch {epoch:03d} | Train RMSE: {train_loss:.5f} | Test RMSE: {test_loss:.5f}")