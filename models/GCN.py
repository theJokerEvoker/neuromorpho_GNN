# benchmark sgcn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter
from torch_geometric.utils import softmax

    
class GCNNet(torch.nn.Module):

    def __init__(self,input_channels_node=1, hidden_channels=128, output_channels=1, readout='add', num_layers=3):
        super(GCNNet, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        
        self.readout = readout
        self.node_lin = Sequential(
            Linear(input_channels_node, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        self.interactions = ModuleList()
        for _ in range(num_layers):
            block = GCNConv(hidden_channels, hidden_channels)
            self.interactions.append(block)
            
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):            
        torch.nn.init.xavier_uniform_(self.node_lin[0].weight)
        self.node_lin[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.node_lin[2].weight)
        self.node_lin[2].bias.data.fill_(0)
        
    def forward(self, x, pos, edge_index, batch):
        
        x = self.node_lin(x)
        for block in self.interactions:
            x = block(x, edge_index)
            x = x.relu()
            
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        out = scatter(x, batch, dim=0, reduce=self.readout)        
    
        return out  