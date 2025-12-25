import torch
import torch.nn as nn
import math

class SpatialEmbedding(nn.Module):
    def __init__(self, hidden_dim=128, max_len=1000):
        super(SpatialEmbedding, self).__init__()
        # We process [x0, y0, x1, y1, w, h] or similar.
        # Simple approach: Learnable embeddings for each coordinate + linear projection
        # Or: Sinusoidal embeddings (like Positional Encoding)
        # Let's use Linear projection of 4 normalized coords for simplicity/flexibility + ReLU
        
        self.input_dim = 4 # [x0, y0, x1, y1]
        self.hidden_dim = hidden_dim
        
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Alternative: Embedding layer for discretized coordinates (e.g. 0-1000 buckets)
        # self.x_emb = nn.Embedding(max_len + 1, hidden_dim // 4)
        # self.y_emb = nn.Embedding(max_len + 1, hidden_dim // 4)
        # self.w_emb = nn.Embedding(max_len + 1, hidden_dim // 4)
        # self.h_emb = nn.Embedding(max_len + 1, hidden_dim // 4)
        # But linear is often sufficient for normalized float coords in new architectures

    def forward(self, bboxes):
        """
        Args:
            bboxes: [Batch, SeqLen, 4] normalized coordinates (0-1000)
            
        Returns:
            embeddings: [Batch, SeqLen, HiddenSize]
        """
        # Normalize to 0-1 range for numerical stability if they are 0-1000
        x = bboxes.float() / 1000.0 
        return self.linear(x)

    def get_embedding_dim(self):
        return self.hidden_dim
