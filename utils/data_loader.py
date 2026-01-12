"""
Data loading utilities for JSONL files.
"""

from typing import List, Dict, Any
import json


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file, preserving line numbers.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of sample dictionaries, each with a '_line_number' field
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                sample['_line_number'] = line_num
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    return samples

