"""
Inference script for subsystem router classifier.

Reads from data/test/test.jsonl and saves results to data/test/result/predictions.jsonl
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from train.classifier import EncoderForMultiLabel
from train.data_preparation import LABELS
from utils.config_loader import load_training_config
from utils.data_loader import load_dataset


def load_model(model_path: str, device: str = None):
    """
    Load trained model and tokenizer from saved path.
    
    Args:
        model_path: Path to the saved model directory
        device: Device to load model on (auto-detect if None)
    
    Returns:
        model, tokenizer, model_config
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = Path(model_path)
    
    # Load model config
    with open(model_path / "model_config.json", "r", encoding="utf-8") as f:
        model_config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Initialize model
    model = EncoderForMultiLabel(
        model_name=model_config["model_name"],
        num_labels=model_config["num_labels"],
        drop_out=model_config["drop_out"],
        pooling=model_config["pooling"]
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, model_config


def predict_batch(model, tokenizer, queries: list, device: str = None, threshold: float = 0.5, batch_size: int = 32):
    """
    Predict subsystems for a batch of queries.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        queries: List of query strings
        device: Device to run inference on (auto-detect if None)
        threshold: Probability threshold for binary classification
        batch_size: Batch size for processing
    
    Returns:
        List of prediction dictionaries
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
        batch_queries = queries[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_queries,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        
        # Process each prediction in the batch
        for prob_row in probabilities:
            predictions = {}
            for j, label in enumerate(LABELS):
                predictions[label] = {
                    "predicted": bool(prob_row[j] >= threshold),
                    "probability": float(prob_row[j])
                }
            all_predictions.append(predictions)
    
    return all_predictions


def main():
    # Configuration
    test_file = "data/test.jsonl"
    output_dir = "data/result"
    threshold = 0.5
    batch_size = 32
    
    # Load model path from config
    config = load_training_config("train/training_config.yml")
    model_path = config.model_path
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: uv run -m train.main")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer, model_config = load_model(model_path)
    print("Model loaded successfully!")
    print()
    
    # Check if test file exists
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: Test file not found at {test_file}")
        return
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    test_data = load_dataset(test_file)
    print(f"Loaded {len(test_data)} test samples")
    print()
    
    # Extract queries
    queries = [item['query'] for item in test_data]
    
    # Predict
    print("Running inference...")
    predictions = predict_batch(model, tokenizer, queries, threshold=threshold, batch_size=batch_size)
    
    # Prepare output data
    output_data = []
    for item, pred in zip(test_data, predictions):
        output_item = {
            "query": item["query"],
            "subsystems": {
                label: 1 if pred[label]["predicted"] else 0
                for label in LABELS
            },
            "probabilities": {
                label: pred[label]["probability"]
                for label in LABELS
            }
        }
        # Preserve original subsystems if present (for comparison)
        if "subsystems" in item:
            output_item["original_subsystems"] = item["subsystems"]
        output_data.append(output_item)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_file = output_path / "predictions.jsonl"
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {output_file}")
    print(f"Processed {len(output_data)} samples")


if __name__ == "__main__":
    main()
