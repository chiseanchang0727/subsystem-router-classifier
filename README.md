# Subsystem Router Classifier

A lightweight classification model that predicts which subsystem should handle a user query, optimized for low latency and high routing accuracy in agentic systems.

This repository focuses only on the routing classification problem, not orchestration logic or downstream execution.


## Goal

Given a **single user query**, predict the **most appropriate subsystem** to handle the request, such as:

### Example Queries

| User Query | Expected Subsystem(s) |
|-----------|----------------------|
| 不鏽鋼跟鑄鐵鋼的差異是什麼 | material_knowledge |
| 排水規範有哪些 | regulation_lookup |
| 排水管用 PVC 跟鑄鐵，法規上有什麼差異？ | material_knowledge, regulation_lookup |

### Example Model Output

```json
{
  "material_knowledge": 0.82,
  "regulation_lookup": 0.91,
  "acquire_image_example": 0.05
}


The model is designed to be:
- **Fast** (sub-100ms inference)
- **Accurate** on domain-specific queries


---

## 🧠 Design Philosophy

- This is **pure classification**
- The model outputs **labels + confidence**, nothing else

> Models extract signals.  
> Orchestrators apply policy.


## Model Scope

- SLM or other text-classification model


## Evaluation Metrics

### Accuracy
- Top-1 routing accuracy: 
- Confusion matrix (per subsystem)

### Latency
- End-to-end inference time  
- P50 / P95 latency

### Confidence Quality
- Logit margin (top-1 vs top-2)  
- Accuracy vs confidence curve

---

## Experiments

This repo is designed to answer:

- Which **model size** gives the best **latency × accuracy** tradeoff?
- How much **training data** is actually needed?
- How stable are predictions across **paraphrases**?
- Where do **misroutes** occur?
