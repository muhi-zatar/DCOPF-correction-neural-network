import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv

class ACOPFPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        for _ in range(2):  # 2 layers
            conv = HeteroConv({
                ('bus', 'ac_line', 'bus'): GCNConv(-1, hidden_channels),
                ('bus', 'transformer', 'bus'): GCNConv(-1, hidden_channels),
                ('generator', 'generator_link', 'bus'): SAGEConv((-1, -1), hidden_channels),
                ('bus', 'generator_link', 'generator'): SAGEConv((-1, -1), hidden_channels),
                ('load', 'load_link', 'bus'): SAGEConv((-1, -1), hidden_channels),
                ('bus', 'load_link', 'load'): SAGEConv((-1, -1), hidden_channels),
                ('shunt', 'shunt_link', 'bus'): SAGEConv((-1, -1), hidden_channels),
                ('bus', 'shunt_link', 'shunt'): SAGEConv((-1, -1), hidden_channels),
            })
            self.convs.append(conv)

        self.lin = torch.nn.Linear(hidden_channels, 4)  # 4 outputs: P, Q, V, theta

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        out_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return out_dict