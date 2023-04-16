import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size
from torch_geometric.utils import add_self_loops, remove_self_loops


class SageConvGraphLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(SageConvGraphLayer, self).__init__(aggr='max')
        self.linear = torch.nn.Linear(in_features=in_channels + out_channels, out_features=out_channels, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear_update = torch.nn.Linear(in_features=in_channels + out_channels, out_features=out_channels,
                                             bias=False)
        self.update_activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        pass

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.linear(x_j)
        x_j = self.activation(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        embedding = torch.cat([aggr_out, x], dim=1)
        embedding = self.update_linear(embedding)
        embedding = self.update_activation(embedding)

        return embedding
