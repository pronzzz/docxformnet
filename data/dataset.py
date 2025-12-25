import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class FormDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer=None, max_length=512):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g., 'data/CORD').
            split (str): 'train', 'test', or 'validation'.
            tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to use for subword tokenization.
            max_length (int): Maximum sequence length.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        json_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(json_path):
             raise FileNotFoundError(f"Dataset file not found: {json_path}")
             
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Standard NER tags (placeholder map, ideally should be built from dataset)
        self.label2id = {'O': 0} 
        self.id2label = {0: 'O'}

        self._build_label_map()

    def _build_label_map(self):
        """Builds label map from the dataset."""
        unique_labels = set()
        for item in self.data:
            for label in item['ner_tags']:
                unique_labels.add(label)
        
        # Sort for determinism
        sorted_labels = sorted(list(unique_labels))
        for i, label in enumerate(sorted_labels):
            # 0 is usually reserved for padding or O, but let's just map everything simply for now
            # In BIO schemes, 'O' is common.
            self.label2id[label] = i
            self.id2label[i] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['words']
        bboxes = item['bboxes']
        ner_tags = item['ner_tags']
        image_path = item.get('image_path')
        
        # Basic check for empty words
        if len(words) == 0:
            # Handle empty example gracefully? Or just return the next one?
            # For simplicity, returning empty tensors or skipping should be handled in collator or preprocessing.
             pass

        # If tokenizer is provided, we perform tokenization here
        # For now, we return raw data to be collated/batched later
        # In a real pipeline, we'd handle subword tokenization alignment here.
        
        # Normalize bboxes if not already normalized (0-1000 is standard)
        # Assuming input is already normalized from download script (HF datasets usually are)
        
        input_ids = []
        attention_mask = []
        token_boxes = []
        labels = []
        
        if self.tokenizer:
            # Tokenize word-by-word to preserve label alignment
            for word, box, label in zip(words, bboxes, ner_tags):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    continue
                word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                
                input_ids.extend(word_ids)
                attention_mask.extend([1] * len(word_ids))
                token_boxes.extend([box] * len(word_ids)) # Expand box for all subwords
                
                # Label alignment: Usually first subword gets label, others get ignored (-100) or 'I-' tag
                # Here assigning same label to all for simplicity, or we can use -100 for subsequent tokens
                label_id = self.label2id.get(label, 0) # Default to 0 if unknown
                labels.extend([label_id] * len(word_ids))

            # Truncation / Padding
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                token_boxes = token_boxes[:self.max_length]
                labels = labels[:self.max_length]
            else:
                # Padding will be handled by collator usually, but can do here too
                padding_length = self.max_length - len(input_ids)
                input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                token_boxes.extend([[0, 0, 0, 0]] * padding_length)
                labels.extend([-100] * padding_length) # -100 is ignored by CrossEntropyLoss

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'bbox': torch.tensor(token_boxes, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'original_words': words,
                'original_bboxes': bboxes
            }

        else:
             # Return raw lists if no tokenizer yet
             return {
                'words': words,
                'bboxes': bboxes,
                'labels': ner_tags,
                'image_path': image_path
            }

if __name__ == "__main__":
    # Simple test
    dataset = FormDataset("data/CORD", "train")
    print(f"Loaded {len(dataset)} examples.")
    print(f"Sample keys: {dataset[0].keys()}")
