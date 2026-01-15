import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import TrainingConfig, load_training_config, load_dataset
from train.data_preparation import BuildDataset
from train.classifier import EncoderForMultiLabel
from sklearn.model_selection import train_test_split

def train_step(model, batch, optimizer, loss_fn, device):
    model.train()

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop("labels")

    logits = model(**batch)
    loss = loss_fn(logits, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

@torch.no_grad()
def val_step(model, batch, loss_fn, device):
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop("labels")

    logits = model(**batch)
    loss = loss_fn(logits, labels)

    return loss.item()


def main(config: TrainingConfig, train_data, val_data   ):
    # Load training configuration


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = BuildDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.max_length
    )

    val_dataset = BuildDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=config.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,   # IMPORTANT
        num_workers=2,
        pin_memory=True,
    )

    model = EncoderForMultiLabel(
        model_name=config.model_name,
        num_labels=config.num_labels,
        drop_out=config.drop_out,
        pooling=config.pooling
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        train_loss = 0.0

        for batch in tqdm(train_loader):
            loss = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device
            )
            train_loss += loss
    
        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        val_loss = 0.0

        for batch in tqdm(val_loader):
            loss = val_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                device=device
            )
            val_loss += loss

        val_loss /= len(val_loader)

        
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # Save model and tokenizer
    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), model_path / "pytorch_model.bin")
    
    # Save tokenizer
    tokenizer.save_pretrained(model_path)
    
    # Save model config (for inference)
    import json
    with open(model_path / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model_name,
            "num_labels": config.num_labels,
            "pooling": config.pooling,
            "drop_out": config.drop_out,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Model saved to {model_path}")
    
    return model, tokenizer

if __name__ == "__main__":
    config: TrainingConfig = load_training_config("train/training_config.yml")

    raw_data = load_dataset(config.data_path)

    train_data, val_data = train_test_split(
        raw_data,
        test_size=0.1,        # 10% for validation
        random_state=42,
        shuffle=True
    )
    print(f"Loaded {len(train_data)} training samples")
    
    model, tokenizer = main(config, train_data, val_data)