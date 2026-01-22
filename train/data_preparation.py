from torch.utils.data import Dataset
import torch
from train.labels import LABELS

NUM_LABELS = len(LABELS)

class BuildDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64) :
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # EX: "subsystems": {"material_knowledge": 1, "causal_regulation_lookup": 1, "pure_regulation_lookup": 0, "acquire_image_example": 0}
        labels = [item['subsystems'][l] for l in LABELS]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }


