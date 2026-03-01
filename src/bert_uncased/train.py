import os
import sys
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import clean_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class HTTPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    processed_dir = f"{PROJECT_ROOT}/data/processed"
    models_dir = f"{PROJECT_ROOT}/models/bert_uncased"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    
    # Subsample to keep training fast
    print("Subsampling data for speed...")
    train_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(2000, len(val_df)), random_state=42)
    
    def extract_text(row):
        fields = ['path', 'query', 'headers', 'body']
        vals = []
        for f in fields:
            v = str(row.get(f, '')).strip()
            if v and v.lower() != 'nan':
                vals.append(v)
        return " ".join(vals)

    print("Cleaning text...")
    train_texts = train_df.apply(extract_text, axis=1).apply(clean_text).tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = val_df.apply(extract_text, axis=1).apply(clean_text).tolist()
    val_labels = val_df['label'].tolist()
    
    print("Tokenizing...")
    # Use DistilBert which is smaller and faster, adhering closely to bert-uncased requirements
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = HTTPDataset(train_encodings, train_labels)
    val_dataset = HTTPDataset(val_encodings, val_labels)
    
    print("Training model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(models_dir, 'results'),
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(models_dir, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    print("Saving model...")
    model.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)
    print(f"Model saved to {models_dir}")

if __name__ == "__main__":
    train_model()
