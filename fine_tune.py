import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


LABELS = {"benign": 0, "trafficking": 1, "BENIGN": 0, "TRAFFICKING": 1, 0: 0, 1: 1}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def read_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        frame = pd.read_json(path)
    else:
        frame = pd.read_csv(path)

    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Dataset must contain text and label columns.")

    frame = frame[["text", "label"]].dropna()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].map(lambda value: LABELS.get(value, LABELS.get(str(value), None)))
    frame = frame.dropna()
    frame["label"] = frame["label"].astype(int)
    return frame


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default="distilroberta-base")
    parser.add_argument("--output", default="models/trafficking-roberta")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    frame = read_dataset(args.data)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        frame["text"].tolist(),
        frame["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=frame["label"].tolist() if frame["label"].nunique() > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "BENIGN", 1: "TRAFFICKING"},
        label2id={"BENIGN": 0, "TRAFFICKING": 1},
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Fine-tuned model saved to {args.output}")


if __name__ == "__main__":
    main()
