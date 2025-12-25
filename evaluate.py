import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import os
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from data.dataset import FormDataset
from utils.collator import FormCollator
from model.formnet import FormNet

def evaluate_checkpoint(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = FormDataset(args.data_dir, 'test', tokenizer=tokenizer)
    collator = FormCollator(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    num_labels = len(dataset.label2id)
    model = FormNet(num_labels=num_labels, hidden_dim=768)
    
    # Load checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, using random initialization (for testing pipeline)")
        
    model.to(device)
    model.eval()
    
    preds_list = []
    labels_list = []
    id2label = dataset.id2label
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bboxes = batch['bboxes'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, bboxes, adj_matrix)
            preds = torch.argmax(logits, dim=2)
            
            for i in range(len(labels)):
                true_labels = []
                pred_labels = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        true_labels.append(id2label[labels[i][j].item()])
                        pred_labels.append(id2label[preds[i][j].item()])
                labels_list.append(true_labels)
                preds_list.append(pred_labels)
                
    print("\nClassification Report:\n")
    print(classification_report(labels_list, preds_list))
    
    f1 = f1_score(labels_list, preds_list)
    print(f"Overall F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/CORD")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    args = parser.parse_args()
    evaluate_checkpoint(args)
