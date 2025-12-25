import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEmbedding(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', freeze_layers=False):
        super(TextEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.hidden_size = self.model.config.hidden_size
        
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [Batch, SeqLen]
            attention_mask: [Batch, SeqLen]
            
        Returns:
            embeddings: [Batch, SeqLen, HiddenSize]
        """
        # DistilBERT outputs tuple, first element is last hidden state
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state

    def get_embedding_dim(self):
        return self.hidden_size
