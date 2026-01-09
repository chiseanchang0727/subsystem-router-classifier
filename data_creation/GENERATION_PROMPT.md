# Training Data Generation Prompt

This document contains the prompt template used to generate training data for the subsystem router classifier.

## Prompt

You are generating training data for a subsystem router classifier. Generate diverse, natural Chinese queries with their corresponding subsystem labels.

### Task

For each example, create a JSON object with:
- `query`: A natural, casual Chinese query (user question)
- `subsystems`: A dictionary mapping each subsystem to 0 or 1 (binary label)

### Subsystems

The available subsystems are:

1. **material_knowledge** (1 or 0)
   - Use when query asks about material properties, differences, characteristics, comparisons
   - Examples: "不鏽鋼跟鑄鐵鋼有什麼差別啊", "銅管跟鋁管哪個導熱比較好"

2. **causal_regulation_lookup** (1 or 0)
   - Use when query asks about regulations/standards WITH specific context (design phase, construction phase, review/inspection phase, specific scenarios)
   - Context indicators: "設計", "施工", "審查"
   - Examples: "設計時給排水要用什麼管材？法規有要求嗎", "施工時接地要怎麼做？有什麼規範要遵守嗎"

3. **pure_regulation_lookup** (1 or 0)
   - Use when query asks about regulations/standards WITHOUT specific contextual information
   - Examples: "排水有什麼規範嗎", "電線規格法規有規定嗎"

4. **acquire_image_example** (1 or 0)
   - Use when query explicitly requests images, photos, diagrams, examples, or visual references
   - Indicators: "有圖嗎", "可以看一下", "圖片", "示意圖", "範例", "長什麼樣子" or similar descriptions
   - Examples: "可以看一下不鏽鋼水管長什麼樣子嗎", "有圖可以看嗎"

### Important Rules

1. **Multi-label is allowed**: A query can have multiple subsystems set to 1
2. **Causal vs Pure regulation**: 
   - If query mentions specific stage/context (設計/施工/審查) → use `causal_regulation_lookup`
   - If query asks about regulations in general → use `pure_regulation_lookup`
   - Do NOT set both to 1 for the same query
3. **Natural, casual tone**: Use conversational Chinese with particles like "啊", "嗎", "呢"
4. **Domain coverage**: Include queries about:
   - Plumbing (給排水, 排水, 給水, 熱水管, etc.)
   - Electricity (電氣, 電線, 配線, 接地, 弱電, etc.)
   - Fire safety (消防, 消防水系統, etc.)
   - HVAC (空調, 冷媒管, 排煙, etc.)
   - Structural (結構, 鋼材, etc.)
   - Cross-field queries (combining multiple systems)

### Stage Coverage

Ensure queries cover three key stages:
- **設計階段** (Design phase): "設計時", "設計階段"
- **施工階段** (Construction phase): "施工時", "施工階段"
- **審查階段** (Review/Inspection phase): "審查時", "審查階段"

### Output Format

Each line should be a valid JSON object in JSONL format:

```json
{"query": "自然的中文查詢", "subsystems": {"material_knowledge": 1, "causal_regulation_lookup": 0, "pure_regulation_lookup": 0, "acquire_image_example": 0}}
```

### Example Output

```jsonl
{"query": "不鏽鋼跟鑄鐵鋼有什麼差別啊", "subsystems": {"material_knowledge": 1, "causal_regulation_lookup": 0, "pure_regulation_lookup": 0, "acquire_image_example": 0}}
{"query": "排水有什麼規範嗎", "subsystems": {"material_knowledge": 0, "causal_regulation_lookup": 0, "pure_regulation_lookup": 1, "acquire_image_example": 0}}
{"query": "排水管用 PVC 還是鑄鐵比較好？法規上有差嗎", "subsystems": {"material_knowledge": 1, "causal_regulation_lookup": 0, "pure_regulation_lookup": 1, "acquire_image_example": 0}}
{"query": "設計時給排水要用什麼管材？法規有要求嗎", "subsystems": {"material_knowledge": 1, "causal_regulation_lookup": 1, "pure_regulation_lookup": 0, "acquire_image_example": 0}}
{"query": "施工時接地要怎麼做？有什麼規範要遵守嗎", "subsystems": {"material_knowledge": 0, "causal_regulation_lookup": 1, "pure_regulation_lookup": 0, "acquire_image_example": 1}}
```

### Diversity Requirements

- Mix of single-label and multi-label queries
- Mix of simple and complex queries
- Cross-field queries (combining multiple systems/subfields)
- Coverage of all three stages (設計/施工/審查)
- Natural variations in phrasing and question structure

