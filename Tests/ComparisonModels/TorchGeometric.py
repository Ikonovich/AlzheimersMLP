import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch
import torch.nn.functional as func
from torch_geometric.data import Data, DataLoader

from DataHandling.YouChooseClickDataset import YouChooseDataset
from Tests.ComparisonModels.SageConvGraphLayer import SageConvGraphLayer

# Test data
samples = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
labels = torch.tensor([0, 1, 0, 1], dtype=torch.float)

# Contains the mapping from source nodes (list one) to target nodes (list two).
edges = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.float)

dataset = YouChooseDataset()
dataset = dataset.shuffle()


train_dataset = dataset[:800000]
val_dataset = dataset[800000:900000]
test_dataset = dataset[900000:]

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Need to read in the data files to let us get a few parameters from the data frame
click_df = pd.read_csv("C:\\Users\\evanh\\Documents\\Datasets\\You-choose\\youchoose-clicks.dat", header=None)
click_df.columns = ['session_id', 'timestamp', 'item_id', 'category']

# Transform the item_ids so that they start at 0
label_encoder = LabelEncoder()
click_df['item_id'] = label_encoder.fit_transform(click_df.item_id)


class GeometricNet(torch.nn.Module):

    def __init__(self):
        super(GeometricNet, self).__init__()
        embed_dim = 128

        self.conv1 = SageConvGraphLayer(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SageConvGraphLayer(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SageConvGraphLayer(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=click_df.item_id.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x - func.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = func.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = func.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = func.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x