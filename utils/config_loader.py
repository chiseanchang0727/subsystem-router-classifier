from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import BaseModel


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration values
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


class TrainingConfig(BaseModel):
    """Training configuration model."""
    data_path: str
    model_name: str
    num_labels: int
    pooling: str
    drop_out: float
    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    gradient_accumulation_steps: int
    save_steps: int
    save_total_limit: int


def load_training_config(config_path: str = "train/training_config.yml") -> TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to training configuration YAML file
    
    Returns:
        TrainingConfig instance with configuration values
    """
    config = load_config(config_path)
    training_dict = config.get('training', {})
    return TrainingConfig(**training_dict)

