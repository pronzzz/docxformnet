import torch
import torch.nn as nn
from .text_embedding import TextEmbedding
from .spatial_embedding import SpatialEmbedding

class CombinedEmbedding(nn.Module):
    def __init__(self, hidden_dim=768, spatial_dim=128):
        super(CombinedEmbedding, self).__init__()
        self.text_emb = TextEmbedding()
        self.spatial_emb = SpatialEmbedding(hidden_dim=spatial_dim)
        
        text_dim = self.text_emb.get_embedding_dim()
        spatial_out_dim = self.spatial_emb.get_embedding_dim()
        
        # Fuse via concatenation + projection + LayerNorm
        self.projection = nn.Linear(text_dim + spatial_out_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, bboxes):
        """
        Args:
            input_ids: [Batch, SeqLen]
            attention_mask: [Batch, SeqLen]
            bboxes: [Batch, SeqLen, 4]
            
        Returns:
            fused_embeddings: [Batch, SeqLen, HiddenSize]
        """
        text_feat = self.text_emb(input_ids, attention_mask) # [B, S, 768]
        spatial_feat = self.spatial_emb(bboxes)              # [B, S, 128]
        
        combined = torch.cat([text_feat, spatial_feat], dim=-1)
        projected = self.projection(combined)
        
        return self.dropout(self.layer_norm(projected))
        
    def get_embedding_dim(self):
        return self.projection.out_features
