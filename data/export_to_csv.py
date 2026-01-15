"""
Export JSONL files to CSV format.

Combines train.jsonl and test.jsonl into a single CSV file with a dataset label column.
"""

import json
import csv
from pathlib import Path
from train.data_preparation import LABELS


def load_jsonl(jsonl_path: str):
    """
    Load samples from JSONL file.
    
    Args:
        jsonl_path: Path to input JSONL file
    
    Returns:
        List of sample dictionaries
    """
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Warning: File not found: {jsonl_path}")
        return []
    
    samples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    return samples


def export_combined_to_csv(synthetic_jsonl: str, test_jsonl: str, csv_path: str):
    """
    Combine JSONL files and export to a single CSV file with dataset labels.
    
    Args:
        synthetic_jsonl: Path to train.jsonl
        test_jsonl: Path to test.jsonl
        csv_path: Path to output CSV file
    """
    # Load samples from both files
    print(f"Loading {synthetic_jsonl}...")
    train_samples = load_jsonl(synthetic_jsonl)
    print(f"Loaded {len(train_samples)} samples from train.jsonl")
    
    print(f"Loading {test_jsonl}...")
    test_samples = load_jsonl(test_jsonl)
    print(f"Loaded {len(test_samples)} samples from test.jsonl")
    
    if not train_samples and not test_samples:
        print("No valid samples found in either file")
        return
    
    # Write combined CSV with UTF-8 BOM
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write UTF-8 BOM first
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        # Define CSV columns: dataset label, query, and all subsystem labels
        fieldnames = ['dataset', 'query'] + LABELS
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write train samples
        for sample in train_samples:
            row = {
                'dataset': 'train',
                'query': sample['query']
            }
            # Add subsystem labels
            for label in LABELS:
                row[label] = sample['subsystems'].get(label, 0)
            writer.writerow(row)
        
        # Write test samples
        for sample in test_samples:
            row = {
                'dataset': 'test',
                'query': sample['query']
            }
            # Add subsystem labels
            for label in LABELS:
                row[label] = sample['subsystems'].get(label, 0)
            writer.writerow(row)
    
    total_samples = len(train_samples) + len(test_samples)
    print(f"Exported {total_samples} samples ({len(train_samples)} train + {len(test_samples)} test) to {csv_path}")


def main():
    # Get the data directory (parent of this script)
    data_dir = Path(__file__).parent
    
    # Paths
    synthetic_jsonl = data_dir / "train.jsonl"
    test_jsonl = data_dir / "test.jsonl"
    combined_csv = data_dir / "combined_data.csv"
    
    # Export combined CSV
    print("Combining JSONL files into CSV...")
    export_combined_to_csv(str(synthetic_jsonl), str(test_jsonl), str(combined_csv))
    print()
    print("Export completed!")


if __name__ == "__main__":
    main()

