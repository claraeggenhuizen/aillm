#This file contains the code to fine-tune and apply a BERT model for our antisemitism classification
#importing libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

#set random seeds for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(42)

#i wrote a simpler version based on this based on the labs and then had Claude fix it as there were some issues with the integration into the other functions
class AntisemitismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        #tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, save_path=None):
    """
    this plots  a confusion matrix for evaluation purposes
    """
    #create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #set default class names 
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    #normalize for better comparisons 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'
        title = "Confusion Matrix"
    
    #plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    #save
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

#function to fine-tune BERT
def train_bert_model(train_data, val_data, model_save_path, 
                     num_labels=3, 
                     max_length=256, 
                     batch_size=8, 
                     epochs=3, 
                     learning_rate=2e-5,
                     model_name='bert-base-uncased',
                     plot_cm=False,
                     cm_save_path=None,
                     class_names=None):
    """
    BERT fine-tuning
    """
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    model = model.to(device)
    
    #prepare datasets 
    train_texts = train_data['text'].values
    train_labels = train_data['category'].values - 1
    
    val_texts = val_data['text'].values
    val_labels = val_data['category'].values - 1
    
    train_dataset = AntisemitismDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = AntisemitismDataset(val_texts, val_labels, tokenizer, max_length)
    
    #create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    #prepare optimizer #ChatGPT suggestion
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    #create directory for saving model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    #training
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0
        
        #add progress bar for training
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            #update progress bar with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Train loss: {avg_train_loss:.4f}")
        
        #evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        #add progress bar for validation
        val_progress = tqdm(val_dataloader, desc="Validation")
        
        for batch in val_progress:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                #store predictions and labels
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        #calculate validation accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    #plot confusion matrix using function
    if plot_cm:
        if class_names is None:
            class_names = [
                "Antisemitic no Israel", 
                "Israel critique no antisemitism", 
                "Israel critique with antisemitism"
            ]
        
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(
            all_labels, 
            all_preds, 
            class_names=class_names,
            save_path=cm_save_path
        )
        
        #also show normalized version
        plot_confusion_matrix(
            all_labels, 
            all_preds, 
            class_names=class_names,
            normalize=True,
            save_path=cm_save_path.replace('.png', '_normalized.png') if cm_save_path else None
        )
    
    #save the final model
    print("Saving model...")
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(os.path.join(os.path.dirname(model_save_path), 'tokenizer'))
    
    return model, tokenizer

#function to make predictions with the fine-tuned model
def predict_with_bert(texts, model, tokenizer, device=None, batch_size=8, max_length=256):
    """
    Maks predictions using fine-tuned BERT model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #set model to evaluation mode
    model.eval()
    
    #create dummy dataset for prediction
    dummy_labels = [0] * len(texts)
    dataset = AntisemitismDataset(texts, dummy_labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    #make predictions with progress bar
    all_preds = []
    all_probs = []
    
    progress_bar = tqdm(dataloader, desc="Making predictions")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            #convert to numpy and store
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    #convert predictions back to 1-indexed categories
    all_preds = [pred + 1 for pred in all_preds]
    
    #calculate confidence as the maximum probability
    confidences = [float(np.max(prob)) for prob in all_probs]
    
    #create results
    results = pd.DataFrame({
        'text': texts,
        'category': all_preds,
        'confidence': confidences
    })
    
    return results

#usage
if __name__ == "__main__":
    #load gold standard data
    gold_df = pd.read_csv("gold.csv")
    
    #split data
    train_df, val_df = train_test_split(gold_df, test_size=0.2, 
                                       stratify=gold_df['category'], 
                                       random_state=42)    
    
    #define category names for readability
    category_names = [
        "Antisemitic no Israel", 
        "Israel critique no antisemitism", 
        "Israel critique with antisemitism"
    ]
    
    #model save path
    model_save_path = "models/bert_antisemitism_classifier.pt"
    
    #train model
    model, tokenizer = train_bert_model(
        train_df, 
        val_df, 
        model_save_path,
        plot_cm=True,
        cm_save_path="confusion_matrix_bert.png",
        class_names=category_names
    )
    
    #load test data and make predictions
    test_df = pd.read_csv("test_posts.csv") 
    test_texts = test_df['text'].tolist()
    
    predictions = predict_with_bert(test_texts, model, tokenizer)
    
    #save predictions
    predictions.to_csv("bert_predictions.csv", index=False)