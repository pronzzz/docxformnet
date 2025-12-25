import torch
import torch.nn as nn
import math

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x, adj):
        """
        Args:
            x: [Batch, NumNodes, InFeatures]
            adj: [Batch, NumNodes, NumNodes] Adjacency matrix (usually normalized)
            
        Returns:
            out: [Batch, NumNodes, OutFeatures]
        """
        # X' = A * X * W
        # x is [B, N, D]
        # adj is [B, N, N]
        
        # Support = X * W
        support = self.linear(x) # [B, N, Out]
        
        # Output = A * Support
        output = torch.bmm(adj, support) # [B, N, Out]
        
        output = self.act(output)
        output = self.dropout(output)
        return output

class GCNModule(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super(GCNModule, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
            
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        """
        Args:
            x: [Batch, NumNodes, HiddenDim]
            adj: [Batch, NumNodes, NumNodes]
        """
        # Residual connection over the whole GCN block or per layer?
        # Usually per layer or dense. Let's do simple residual block around the whole thing or iterative.
        # GraphSAGE/GCN usually just stack.
        # Let's add residual from input if shape matches
        residual = x
        for layer in self.layers:
            x = layer(x, adj)
        
        return self.norm(residual + x)
