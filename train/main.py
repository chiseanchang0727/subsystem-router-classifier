import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import TrainingConfig, load_training_config, load_dataset
from train.data_preparation import BuildDataset
from train.classifier_pt import EncoderForMultiLabel
from train.classifier_hf import build_classifier
from sklearn.model_selection import train_test_split


def train_step(model, batch, optimizer, loss_fn, device, backend):
    model.train()

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop("labels")

    if backend == "hf":
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
    else:
        logits = model(**batch)
        loss = loss_fn(logits, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


@torch.no_grad()
def val_step(model, batch, loss_fn, device, backend):
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop("labels")

    if backend == "hf":
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
    else:
        logits = model(**batch)
        loss = loss_fn(logits, labels)

    return loss.item()


def main(config: TrainingConfig, train_data, val_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = BuildDataset(train_data, tokenizer, config.max_length)
    val_dataset = BuildDataset(val_data, tokenizer, config.max_length)

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
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------- MODEL --------
    if config.backend == "hf":
        model = build_classifier(
            model_name=config.model_name,
            num_labels=config.num_labels,
        )
    else:
        model = EncoderForMultiLabel(
            model_name=config.model_name,
            num_labels=config.num_labels,
            pooling=config.pooling,
            drop_out=config.drop_out,
        )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_fn = nn.BCEWithLogitsLoss() if config.backend == "pt" else None

    # -------- TRAIN --------
    for epoch in range(config.epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            train_loss += train_step(
                model, batch, optimizer, loss_fn, device, config.backend
            )
        train_loss /= len(train_loader)

        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validate"):
            val_loss += val_step(
                model, batch, loss_fn, device, config.backend
            )
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # -------- SAVE --------
    model_path = Path(config.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    if config.backend == "hf":
        HF_REPO_ID = "SeanCCC666/subsystem-classifier"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        tokenizer.push_to_hub(HF_REPO_ID)
        model.push_to_hub(HF_REPO_ID)
        print(
            f"Successfully pushed model and tokenizer to Hugging Face Hub: {HF_REPO_ID}"
        )
        print(  
            f"https://huggingface.co/{HF_REPO_ID}"
        )

    else:
        torch.save(model.state_dict(), model_path / "pytorch_model.bin")
        tokenizer.save_pretrained(model_path)

        import json
        with open(model_path / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": config.model_name,
                    "num_labels": config.num_labels,
                    "pooling": config.pooling,
                    "drop_out": config.drop_out,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Model saved to {model_path}")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier model")
    parser.add_argument(
        "backend",
        type=str,
        choices=["hf", "pt"],
        help="Backend to use: 'hf' for Hugging Face or 'pt' for PyTorch",
    )
    args = parser.parse_args()

    config_path = f"train/training_config_{args.backend}.yml"
    config: TrainingConfig = load_training_config(config_path)

    raw_data = load_dataset(config.data_path)

    train_data, val_data = train_test_split(
        raw_data, test_size=0.1, random_state=42, shuffle=True
    )

    print(f"Loaded {len(train_data)} training samples")
    main(config, train_data, val_data)
