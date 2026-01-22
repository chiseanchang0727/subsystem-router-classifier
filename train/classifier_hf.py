# train/classifier.py

from transformers import AutoModelForSequenceClassification
from train.labels import label2id, id2label

def build_classifier(
    model_name: str,
    num_labels: int,
):
    """
    Build a Hugging Face multi-label classifier.

    Uses BCEWithLogitsLoss internally.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
    )
    return model
