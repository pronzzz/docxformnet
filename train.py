import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse
import os
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from data.dataset import FormDataset
from utils.collator import FormCollator
from model.formnet import FormNet

def train(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Datasets
    train_dataset = FormDataset(args.data_dir, 'train', tokenizer=tokenizer)
    # Using 'validation' or 'test' depending on availability, script downloaded both
    # Usually CORD has train/dev/test.
    val_dataset = FormDataset(args.data_dir, 'test', tokenizer=tokenizer) 
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Dataloaders
    collator = FormCollator(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    # Model
    num_labels = len(train_dataset.label2id) # Assuming dynamic label map
    print(f"Num labels: {num_labels}")
    
    model = FormNet(
        num_labels=num_labels,
        hidden_dim=768, 
        num_attention_layers=2, # Small for demo
        num_gcn_layers=1
    )
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bboxes = batch['bboxes'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, bboxes, adj_matrix) # [B, S, NumLabels]
            
            # Flatten for loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = labels.view(-1)
            
            loss = criterion(active_logits, active_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        evaluate(model, val_loader, device, train_dataset.id2label)
        
        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt"))

def evaluate(model, dataloader, device, id2label):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bboxes = batch['bboxes'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, bboxes, adj_matrix)
            preds = torch.argmax(logits, dim=2)
            
            # Convert to list of lists of tags for seqeval
            for i in range(len(labels)):
                true_labels = []
                pred_labels = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        true_labels.append(id2label[labels[i][j].item()])
                        pred_labels.append(id2label[preds[i][j].item()])
                labels_list.append(true_labels)
                preds_list.append(pred_labels)
                
    f1 = f1_score(labels_list, preds_list)
    print(f"Validation F1: {f1:.4f}")
    # print(classification_report(labels_list, preds_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/CORD")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    train(args)
