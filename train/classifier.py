import torch
import torch.nn as nn
from transformers import AutoModel

class EncoderForMultiLabel(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        num_labels: int, 
        drop_out: float = 0.1,
        pooling: str = 'auto', # auto | cls| mean
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop_out= nn.Dropout(drop_out)
        self.pooling = pooling

        hidden_size = self.model.config.hidden_size

        # final layer to output
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqeeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _pool(self, outputs, attention_mask):
        if self.pooling == 'cls':
            return outputs.last_hidden_state[:, 0, :]
        
        if self.pooling == 'mean':
            return self._mean_pool(outputs.last_hidden_state, attention_mask)
        
        # For BertBase
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # Fallback
        return self._mean_pool(outputs.last_hidden_state, attention_mask)


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self._pool(outputs, attention_mask)# CLS. Need this if the model is BertBase   
        pooled = self.drop_out(pooled)
        logits = self.classifier(pooled)

        # loss = None
        # if labels is not None:
        #     loss_fn = nn.BCEWithLogitsLoss()
        #     loss = loss_fn(logits, labels)

        return logits
