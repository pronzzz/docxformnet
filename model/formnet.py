import torch
import torch.nn as nn
from tokenization.combined_embedding import CombinedEmbedding
from rich_attention.rich_attention import RichAttLayer
from gcn_super_tokens.gcn import GCNModule
from preprocessing.pipeline import Preprocessor # Helper for graph construction if needed

class FormNet(nn.Module):
    def __init__(self, num_labels, hidden_dim=768, num_attention_layers=4, num_gcn_layers=2, dropout=0.1):
        super(FormNet, self).__init__()
        
        self.embedding = CombinedEmbedding(hidden_dim=hidden_dim)
        
        # Rich Attention Encoder
        self.rich_att_layers = nn.ModuleList([
            RichAttLayer(hidden_dim, num_heads=8, dropout=dropout) 
            for _ in range(num_attention_layers)
        ])
        
        # GCN Super-Token Refinement
        self.gcn = GCNModule(hidden_dim, num_layers=num_gcn_layers, dropout=dropout)
        
        # Serialization/Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, bboxes, adj_matrix=None):
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            bboxes: [B, S, 4]
            adj_matrix: [B, S, S] (Optional, if not provided we might need to construct it or skip GCN)
            
        Returns:
            logits: [B, S, NumLabels]
        """
        # 1. Embedding
        x = self.embedding(input_ids, attention_mask, bboxes) # [B, S, H]
        
        # 2. Rich Attention Encoder
        # Create extended attention mask
        # Mask [B, S] -> [B, 1, 1, S] handled inside RichAtt usually or here
        # RichAttLayer expects simple mask or handled internally? 
        # My implementation of RichAttLayer takes attention_mask [B, S] directly.
        for layer in self.rich_att_layers:
            x = layer(x, attention_mask=attention_mask)
            
        # 3. GCN Refinement
        if adj_matrix is not None:
            x = self.gcn(x, adj_matrix)
            
        # 4. Classification
        logits = self.classifier(x)
        
        return logits
