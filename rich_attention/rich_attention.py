import torch
import torch.nn as nn
import math

class RichAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(RichAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable spatial bias (simplified)
        # In a full implementation, we'd compute relative distances and lookup in a bias table
        # bias = MLP(relative_x, relative_y)
        # Here we implement a simplified version where we expect an attention_bias tensor passed in
        # or we implement a basic relative position bias
        
    def forward(self, hidden_states, attention_mask=None, spatial_bias=None):
        """
        Args:
            hidden_states: [Batch, SeqLen, HiddenDim]
            attention_mask: [Batch, SeqLen] (0 for padding)
            spatial_bias: [Batch, NumHeads, SeqLen, SeqLen] (Optional precomputed bias)
            
        Returns:
            output: [Batch, SeqLen, HiddenDim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: Q * K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add spatial bias if provided (Rich Attention part)
        if spatial_bias is not None:
             # Ensure shape broadcasting
             scores = scores + spatial_bias
             
        # Apply mask (mask entries with 0 should be -inf in scores)
        if attention_mask is not None:
            # attention_mask is [B, S], expand to [B, 1, 1, S] for broadcasting
            # we want to mask out padded tokens in Key
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v) # [B, H, S, D_h]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.out_proj(context)
        return output

class RichAttLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(RichAttLayer, self).__init__()
        self.attention = RichAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, attention_mask=None, spatial_bias=None):
        attn_out = self.attention(x, attention_mask, spatial_bias)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
