"""
Sample size calculation formula for validation sampling.

This module implements the statistical formula for calculating required sample size
with finite population correction.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import csv
import random


def count_samples_in_jsonl(file_path: str) -> int:
    """Count the number of samples in a JSONL file."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                count += 1
    return count


def calculate_sample_size(
    population_size: Optional[int] = None,
    dataset_file: str = "data/synthetic_data.jsonl",
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    expected_correctness: float = 0.5
) -> int:
    """
    Calculate required sample size using proportion estimation formula.
    
    Formula:
        n = (z^2 * p(1-p)) / e^2
    
    With finite population correction:
        n_adjusted = n / (1 + (n-1)/N)
    
    Args:
        population_size: Total number of samples in population (N). If None, reads from dataset_file.
        dataset_file: Path to JSONL file to count samples from (default: "data/synthetic_data.jsonl")
        confidence_level: Confidence level (default 0.95 for 95%)
        margin_of_error: Desired margin of error (e, default 0.05 for ±5%)
        expected_correctness: Expected correctness rate (p, default 0.5 for worst-case)
    
    Returns:
        Adjusted sample size (n_adjusted)
    """
    # Get population size from file if not provided
    if population_size is None:
        population_size = count_samples_in_jsonl(dataset_file)
    
    # Z-score for confidence level
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z = z_scores.get(confidence_level, 1.96)
    
    # Initial sample size calculation
    p = expected_correctness
    e = margin_of_error
    n = (z ** 2 * p * (1 - p)) / (e ** 2)
    
    # Finite population correction
    n_adjusted = n / (1 + (n - 1) / population_size)
    
    return int(round(n_adjusted))


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file, preserving line numbers."""
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


def sample_dataset(
    dataset_file: str = "data/synthetic_data.jsonl",
    sample_size: Optional[int] = None,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    expected_correctness: float = 0.5,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample data from the dataset.
    
    Args:
        dataset_file: Path to JSONL file
        sample_size: Number of samples to select. If None, calculates using formula.
        confidence_level: Confidence level for sample size calculation (if sample_size is None)
        margin_of_error: Margin of error for sample size calculation (if sample_size is None)
        expected_correctness: Expected correctness rate for sample size calculation (if sample_size is None)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of sampled items with '_line_number' field preserved
    """
    # Load dataset
    all_samples = load_dataset(dataset_file)
    
    if len(all_samples) == 0:
        raise ValueError(f"No samples found in {dataset_file}")
    
    # Calculate sample size if not provided
    if sample_size is None:
        sample_size = calculate_sample_size(
            dataset_file=dataset_file,
            confidence_level=confidence_level,
            margin_of_error=margin_of_error,
            expected_correctness=expected_correctness
        )
    
    # Ensure sample size doesn't exceed population
    sample_size = min(sample_size, len(all_samples))
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
    
    # Sample randomly
    sampled = random.sample(all_samples, sample_size)
    
    return sampled


def export_sampled_data(
    sampled: List[Dict[str, Any]],
    output_file: str = "synthetic_data_for_validation.csv"
) -> None:
    """
    Export sampled data to CSV file.
    
    Args:
        sampled: List of sampled items
        output_file: Path to output CSV file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all subsystem names for CSV columns
    import sys
    # Add parent directory to path to import subsystems
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from subsystems import get_all_subsystems
    subsystems = get_all_subsystems()
    
    # Define CSV columns
    fieldnames = ['line_number', 'query'] + subsystems
    
    # Write samples to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in sampled:
            row = {
                'line_number': sample.get('_line_number', 0),
                'query': sample.get('query', ''),
            }
            # Add subsystem labels
            subsystems_dict = sample.get('subsystems', {})
            for subsystem in subsystems:
                row[subsystem] = subsystems_dict.get(subsystem, 0)
            
            writer.writerow(row)
    
    print(f"Exported {len(sampled)} samples to {output_file}")


# Example usage
if __name__ == '__main__':
    # Example 1: Calculate sample size
    print("Sample Size Calculation:")
    print("=" * 50)
    n = calculate_sample_size(
        dataset_file="data/synthetic_data.jsonl",
        confidence_level=0.95,
        margin_of_error=0.05,
        expected_correctness=0.5
    )
    population_size = count_samples_in_jsonl("data/synthetic_data.jsonl")
    print(f"Population size: {population_size} (from data/synthetic_data.jsonl)")
    print(f"Confidence level: 95%")
    print(f"Margin of error: ±5%")
    print(f"Required sample size: {n}")
    print()
    
    # Example 2: Sample the dataset
    print("Sampling Dataset:")
    print("=" * 50)
    sampled = sample_dataset(
        dataset_file="data/synthetic_data.jsonl",
        sample_size=n,
        random_seed=42
    )
    print(f"Sampled {len(sampled)} items")
    print(f"First sample (line {sampled[0]['_line_number']}):")
    print(f"  Query: {sampled[0]['query']}")
    print(f"  Subsystems: {sampled[0]['subsystems']}")
    print()
    
    # Example 3: Export sampled data
    print("Exporting Sampled Data:")
    print("=" * 50)
    export_sampled_data(
        sampled,
        output_file="synthetic_validation/synthetic_data_for_validation.csv"
    )
