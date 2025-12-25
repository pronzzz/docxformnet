import torch
from torch.nn.utils.rnn import pad_sequence
from preprocessing.pipeline import Preprocessor

class FormCollator:
    def __init__(self, tokenizer=None, pad_token_id=0):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.preprocessor = Preprocessor()

    def __call__(self, batch):
        # batch is a list of dicts from Dataset.__getitem__
        # Keys: input_ids, attention_mask, bbox, labels
        
        # Extract fields
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        bboxes = [item['bbox'] for item in batch] # [L, 4]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences
        # input_ids: [B, MaxLen]
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Pad bboxes: [B, MaxLen, 4]
        # We need a custom pad for bboxes usually or just convert to tensor first if list
        # Since they are tensors of [L, 4]
        max_len = padded_input_ids.size(1) 
        padded_bboxes = torch.zeros((len(batch), max_len, 4), dtype=torch.long)
        for i, box in enumerate(bboxes):
            length = box.size(0)
            padded_bboxes[i, :length, :] = box
            
        # Construct Adjacency Matrices
        # [B, MaxLen, MaxLen]
        adj_matrices = []
        for i, box in enumerate(bboxes):
            # box is the unpadded tensor [L, 4]
            # normalized 0-1000
            length = box.size(0)
            adj = self.preprocessor.construct_adjacency_matrix(box, token_count=length, mode='knn', k=5) 
            
            # Pad adj to [MaxLen, MaxLen]
            padded_adj = torch.zeros((max_len, max_len), dtype=torch.float32)
            padded_adj[:length, :length] = adj
            adj_matrices.append(padded_adj)
            
        padded_adj_matrices = torch.stack(adj_matrices)
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'bboxes': padded_bboxes,
            'labels': padded_labels,
            'adj_matrix': padded_adj_matrices
        }
