# -*- coding: utf-8 -*-
"""

New attempt at fewshot classification with Mistral - asked ChatGPT for suggestions to load bigger model in Colab rather than small models that don't work
"""

!pip install -q accelerate einops

!pip install -U bitsandbytes

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Dict, Tuple, Union
import pandas as pd
import json
import re
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import accelerate
import bitsandbytes
import einops

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Load model in 16-bit (fp16) without quantization if CUDA is not available
# Check if CUDA is available
if torch.cuda.is_available():
    # Load 4-bit quantized model if CUDA is available
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True  # Enable 4-bit quantization
    )
else:
    # Load model in 16-bit (fp16) without quantization if CUDA is not available
    print("Warning: CUDA not available. Loading model in 16-bit (fp16) without quantization.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Simple inference function
def generate_classification(text, fewshot_examples):
    prompt = build_prompt(text, fewshot_examples)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Use model.device instead of "cuda"
    output = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Category:")[-1].strip().split()[0]  # Returns only the number

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Prompt Template ---
def build_prompt(text, examples):
    prompt = "You are an expert content classifier. Classify the following Reddit posts into one of these categories:\n\n" \
             "1 = Traditional antisemitism (antisemitic slurs/tropes without Israel reference)\n" \
             "2 = Critique of Israel without antisemitism\n" \
             "3 = Critique of Israel that includes antisemitic tropes\n" \
             "0 = None of the above\n\n" \
             "Format: <category number>\n\n"

    prompt += "Examples:\n"
    for ex in examples:
        prompt += f"---\nPost: {ex['text']}\nCategory: {ex['category']}\nRationale: {ex['rationale']}\n"

    prompt += "---\nPost: " + str(text) + "\nCategory:"
    return prompt

#---Inference---
def classify_posts_mistral(posts, model, tokenizer, fewshot_examples):
    predictions = []
    for post in posts:
        prompt = build_prompt(post, fewshot_examples)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        output = model.generate(**inputs, max_new_tokens=10)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = decoded.split("Category:")[-1].strip().split()[0]
        predictions.append(prediction)
    return predictions

def evaluate_classification(preds, labels, texts, output_file="predictions.csv", title="Classification Confusion Matrix"):
    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds, average='macro', zero_division=0))
    print("Recall:", recall_score(labels, preds, average='macro', zero_division=0))
    print("F1 Score:", f1_score(labels, preds, average='macro', zero_division=0))

    # Save to CSV with original index
    df_out = pd.DataFrame({
        "index": range(len(texts)),
        "text": texts,
        "gold_label": labels,
        "predicted_label": preds
    })
    df_out.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

    # Get unique label values (sorted)
    unique_labels = sorted(list(set(labels + preds)))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=unique_labels)

    # Map labels to category names
    categories = []
    for label in unique_labels:
        if label == "1" or label == 1:
            categories.append("Traditional antisemitism")
        elif label == "2" or label == 2:
            categories.append("Israel critique")
        elif label == "3" or label == 3:
            categories.append("Israel-based antisemitism")
        elif label == "0" or label == 0 or label == "4" or label == 4:
            categories.append("None of the above")
        else:
            categories.append(f"Unknown ({label})")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
df = pd.read_csv("/content/drive/MyDrive/AI_gold_standard.csv", sep=";")

# FEWSHOT EXAMPLES
fewshot_examples = [
    {
        "text": "it was an act of piracy, you have shit for brains, you half-penised kike.",
        "category": "1",
        "confidence": 1.0,
        "rationale": "Contains antisemitic slur WITHOUT ISRAEL REFERENCE"
    },
    {
        "text": "Trump Torches Globalist Cabal at Davos.",
        "category": "1",
        "confidence": 1.0,
        "rationale": "Contains classic antisemitic trope WITHOUT ISRAEL REFERENCE"
    },
    {
        "text": "I think this is why the jewish freemasons are obsessed with death, cause the creator of this world is satan. The jews belive that lucifer IS light",
        "category": "1",
        "confidence": 1.0,
        "rationale": "Contains classic antisemitic trope WITHOUT ISRAEL REFERENCE"
    },
    {
        "text": "President Isaac Herzog said that all citizens of Gaza are responsible.  This is an ethnic retaliation being inflicted by Israel on a nation where half of the population are minors.",
        "category": "2",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITHOUT ANTISEMITISM"
    },
    {
        "text": "Israel won't let you hear the end of it as an excuse to why they're somehow justified in indiscriminately bombing residential apartment buildings...",
        "category": "2",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITHOUT ANTISEMITISM"
    },
    {
        "text": "We're at the point where the most extreme zionists just want the Palestinians to live in apartheid forever...",
        "category": "2",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITHOUT ANTISEMITISM"
    },
    {
        "text": "I don’t understand why Zionists & Israelis say “it’s antisemitic to say jews are colonizers”...",
        "category": "3",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITH ANTISEMITISM"
    },
    {
        "text": "Take for example the killing of 50 children in Waco Texas... Does it sound like Gaza?",
        "category": "3",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITH ANTISEMITISM"
    },
    {
        "text": "Israeli Prime Minister Benjamin Netanyahu’s father admitted that Sephardic Jews are the only ones with Israelite blood...",
        "category": "3",
        "confidence": 1.0,
        "rationale": "Critiques Israel WITH ANTISEMITISM"
    }
]

#Convert binary label columns to a single class string: "1", "2", "3"
def infer_gold_label(row):
    if row['traditional_antisemitism'] == 1:
        return "1"
    elif row['israel_critique'] == 1:
        return "2"
    elif row['israel_based_antisemitism'] == 1:
        return "3"
    else:
        return "0"  # Optional: could be skipped in evaluation if 'none' means irrelevant

df['gold_label'] = df.apply(infer_gold_label, axis=1)

# Extract text and labels
texts = df['text'].tolist()
gold_labels = df['gold_label'].tolist()

###SPLITTING FOR BERT COMPARISON
from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'],
    df['gold_label'],
    test_size=0.2,
    stratify=df['gold_label'],  # Maintain class balance
    random_state=42
)
# Run prediction and evaluation
mistral_preds = classify_posts_mistral(test_texts.tolist(), model, tokenizer, fewshot_examples)
evaluate_classification(mistral_preds, test_labels.tolist(), test_texts.tolist(),
                        output_file="mixtral_predictions.csv",
                        title="Mixtral Classification Confusion Matrix")

from google.colab import drive
drive.mount('/content/drive')

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import os
import numpy as np

model_path = "/content/drive/MyDrive/models/bert_antisemitism_classifier.pt"
tokenizer_path = "/content/drive/MyDrive/models/tokenizer"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print("Loaded saved tokenizer")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
print("Successfully loaded model weights")

model.to(device)
model.eval()

# Print test set label distribution
test_label_counts = pd.Series(test_labels).value_counts().sort_index()
print("\nTest set label distribution:")
for label, count in test_label_counts.items():
    print(f"Label '{label}': {count} examples")

# Create dataset class matching the training setup
class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.original_labels = labels

        # Map string labels to adjusted integer labels for the model
        self.adjusted_labels = []
        for label in labels:
            # Convert to integer (handling string labels)
            try:
                label_int = int(label)
                # Convert to 0-indexed for model
                if label_int == 0:
                    self.adjusted_labels.append(3)  # Maps to "None of the above"
                else:
                    self.adjusted_labels.append(label_int - 1)
            except ValueError:
                print(f"Warning: Couldn't convert label: {label}")
                self.adjusted_labels.append(0)  # Default fallback

        # Use the same tokenizer settings as in training
        self.encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.adjusted_labels[idx], dtype=torch.long),
            'original_labels': self.original_labels[idx]
        }
        if 'token_type_ids' in self.encodings:
            item['token_type_ids'] = self.encodings['token_type_ids'][idx]
        return item

    def __len__(self):
        return len(self.original_labels)

# Create dataset and dataloader
test_dataset = TestDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# Run predictions
all_preds = []
all_true_labels = []
all_original_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        true_labels = batch['labels'].to(device)
        original_labels = batch['original_labels']

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        # Store predictions and true labels
        all_preds.extend(preds)
        all_true_labels.extend(true_labels.cpu().tolist())
        all_original_labels.extend(original_labels)

# Calculate accuracy on model's internal representation
accuracy = accuracy_score(all_true_labels, all_preds)
print(f"\nBERT model accuracy (internal): {accuracy:.4f}")

# Convert prediction indices back to the original category format (1-indexed)
pred_categories = []
for pred in all_preds:
    if pred == 0:
        pred_categories.append("1")  # Traditional antisemitism
    elif pred == 1:
        pred_categories.append("2")  # Israel critique
    elif pred == 2:
        pred_categories.append("3")  # Israel-based antisemitism
    elif pred == 3:
        pred_categories.append("0")  # None of the above

# Category names (for our reference)
category_names = [
    "Traditional antisemitism",
    "Israel critique (not antisemitic)",
    "Israel-based antisemitism",
    "None of the above"
]

# Print internal prediction distribution
pred_count = Counter(all_preds)
print("\nModel prediction distribution (internal):")
for i in range(4):
    name = category_names[i]
    count = pred_count.get(i, 0)
    percentage = (count/len(all_preds)*100) if len(all_preds) > 0 else 0
    print(f"{name}: {count} predictions ({percentage:.1f}%)")

# Sample predictions
print("\nSample predictions (first 5):")
for i in range(min(5, len(test_texts))):
    category_idx = all_preds[i]
    category_name = category_names[category_idx]
    print(f"Text: {test_texts.iloc[i][:100]}...")
    print(f"True label: {all_original_labels[i]} | BERT prediction: {pred_categories[i]} ({category_name})")
    print("-" * 80)

# Evaluate using your fixed evaluation function
evaluate_classification(pred_categories, all_original_labels, test_texts.tolist(),
                        output_file="bert_predictions.csv",
                        title="BERT Classification Confusion Matrix")
