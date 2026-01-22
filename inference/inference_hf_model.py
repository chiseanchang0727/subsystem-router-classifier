"""
Inference script for subsystem router classifier (Hugging Face).

Reads from data/test.jsonl and saves results to data/result/predictions.jsonl
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.data_loader import load_dataset
from evaluation.f1 import evaluate_multilabel_f1

HF_REPO_ID = "SeanCCC666/subsystem-classifier"


def load_model(repo_id: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    model.to(device)
    model.eval()

    labels = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]

    return model, tokenizer, labels, device

def predict_batch(
    model,
    tokenizer,
    queries: list[str],
    labels: list[str],
    device: str,
    threshold: float = 0.5,
    batch_size: int = 32,
):
    all_predictions = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
        batch_queries = queries[i : i + batch_size]

        encodings = tokenizer(
            batch_queries,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encodings).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()

        for prob_row in probabilities:
            preds = {}
            for j, label in enumerate(labels):
                preds[label] = {
                    "predicted": bool(prob_row[j] >= threshold),
                    "probability": float(prob_row[j]),
                }
            all_predictions.append(preds)

    return all_predictions


def main():
    test_file = "data/test.jsonl"
    output_dir = "data/result"
    threshold = 0.5
    batch_size = 32

    print(f"Loading model from Hugging Face: {HF_REPO_ID}")
    model, tokenizer, labels, device = load_model(HF_REPO_ID)
    print("Model loaded successfully!\n")

    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: Test file not found at {test_file}")
        return

    print(f"Loading test data from {test_file}...")
    test_data = load_dataset(test_file)
    print(f"Loaded {len(test_data)} test samples\n")

    queries = [item["query"] for item in test_data]

    print("Running inference...")
    predictions = predict_batch(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        labels=labels,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
    )

    metrics = evaluate_multilabel_f1(
        test_data=test_data,
        predictions=predictions,
        labels=labels,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "predictions.jsonl"
    print(f"Saving results to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for item, pred in zip(test_data, predictions):
            out = {
                "query": item["query"],
                "subsystems": {
                    label: int(pred[label]["predicted"]) for label in labels
                },
                "probabilities": {
                    label: pred[label]["probability"] for label in labels
                },
            }
            if "subsystems" in item:
                out["original_subsystems"] = item["subsystems"]

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")
    print(f"Processed {len(predictions)} samples")


if __name__ == "__main__":
    main()
