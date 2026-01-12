"""
Dataset distribution analysis module.

This module provides functions to analyze the distribution of samples
by subsystem types in the dataset.
"""

from typing import Dict, Any
from pathlib import Path

from utils.data_loader import load_dataset


def analyze_dataset_distribution(
    dataset_file: str = "data/synthetic_data.jsonl"
) -> Dict[str, Any]:
    """
    Analyze the distribution of samples by subsystem types.
    
    Args:
        dataset_file: Path to JSONL file
    
    Returns:
        Dictionary with statistics about the dataset distribution
    """
    samples = load_dataset(dataset_file)
    
    if len(samples) == 0:
        return {"total_samples": 0, "subsystem_counts": {}, "multi_subsystem_count": 0}
    
    # Get all subsystems
    from subsystems import get_all_subsystems
    subsystems = get_all_subsystems()
    
    # Count samples for each subsystem
    subsystem_counts = {subsystem: 0 for subsystem in subsystems}
    multi_subsystem_count = 0
    single_subsystem_count = 0
    
    for sample in samples:
        labels = sample.get('subsystems', {})
        active_count = sum(1 for v in labels.values() if v == 1)
        
        if active_count > 1:
            multi_subsystem_count += 1
        elif active_count == 1:
            single_subsystem_count += 1
        
        # Count each subsystem
        for subsystem in subsystems:
            if labels.get(subsystem, 0) == 1:
                subsystem_counts[subsystem] += 1
    
    return {
        "total_samples": len(samples),
        "subsystem_counts": subsystem_counts,
        "single_subsystem_count": single_subsystem_count,
        "multi_subsystem_count": multi_subsystem_count,
        "subsystem_percentages": {
            subsystem: (count / len(samples) * 100) 
            for subsystem, count in subsystem_counts.items()
        }
    }


def print_dataset_distribution(dataset_file: str = "data/synthetic_data.jsonl") -> None:
    """
    Print a formatted report of dataset distribution by subsystem types.
    
    Args:
        dataset_file: Path to JSONL file
    """
    stats = analyze_dataset_distribution(dataset_file)
    
    print("Dataset Distribution Analysis")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print()
    print("Subsystem Distribution:")
    print("-" * 60)
    for subsystem, count in stats['subsystem_counts'].items():
        percentage = stats['subsystem_percentages'][subsystem]
        print(f"  {subsystem:30s}: {count:4d} ({percentage:5.1f}%)")
    print()
    print("Sample Type Distribution:")
    print("-" * 60)
    print(f"  Single subsystem: {stats['single_subsystem_count']:4d}")
    print(f"  Multi-subsystem:  {stats['multi_subsystem_count']:4d}")
    print("=" * 60)


# Example usage
if __name__ == '__main__':
    print_dataset_distribution("data/synthetic_data.jsonl")

