# Implementation Plan: Data Creation & Validation Pipeline

This document outlines the code architecture and implementation plan for the data creation and validation pipeline.

## Overview

The pipeline consists of three main components:

1. **Data Generation** - Generate synthetic training data and append to existing dataset
2. **Sampling & Validation Export** - Sample data for human validation and export to CSV
3. **Statistical Testing** - Read validated data and perform distribution comparison tests

---

## Component 1: Data Generation & Append

### Purpose
Generate new training samples using AI and append them to the existing dataset.

### Key Features
- Generate data based on configurable target size
- Append to existing `data/train.jsonl`
- Validate format and ensure all subsystems are present
- Handle incremental/batch generation

### Configuration (YAML)
```yaml
data_generation:
  target_size: 1000  # Total samples desired
  batch_size: 100     # Generate in batches
  output_file: "data/train.jsonl"
  prompt_file: "data_creation/GENERATION_PROMPT.md"
  subsystems_file: "subsystems.py"
```

### Implementation Details

**Functions/Modules Needed:**
- `load_existing_data()` - Read current train.jsonl, return count
- `load_subsystems()` - Import from subsystems.py to get all subsystem IDs
- `load_generation_prompt()` - Read prompt template
- `generate_batch()` - Call LLM API to generate batch of samples
- `validate_sample_format()` - Check JSON structure, all subsystems present, valid query
- `deduplicate_queries()` - Check for exact/fuzzy duplicates (optional)
- `append_to_dataset()` - Write new samples to JSONL file
- `backup_dataset()` - Create backup before appending (safety)

**Output:**
- Updated `data/train.jsonl` with new samples
- Log of generation stats (count, timestamp, parameters)

**Error Handling:**
- Invalid JSON format
- Missing subsystems in labels
- Empty queries
- File I/O errors

---

## Component 2: Sampling & Validation Export

### Purpose
Stratify and sample data for human validation, export to CSV with metadata.

### Key Features
- Stratified sampling based on difficulty factors
- Calculate sample weights for oversampled strata
- Export to CSV with all necessary metadata for validators
- Track validation status

### Configuration (YAML)
```yaml
sampling:
  total_samples: 280
  confidence_level: 0.95
  margin_of_error: 0.05
  population_size: 1000
  
  stratification:
    oversample_multi_subsystem: true
    oversample_regulation: true
    oversample_image: true
    oversample_complex: true
    
  complexity_heuristics:
    length_threshold: 50
    conjunction_keywords: ["和", "跟", "還有", "以及"]
    question_markers: ["嗎", "呢", "？", "?"]
    
  output:
    csv_file: "data/validation_samples.csv"
    include_metadata: true
    include_weights: true
```

### Implementation Details

**Functions/Modules Needed:**
- `load_dataset()` - Read train.jsonl
- `calculate_complexity()` - Score query complexity based on heuristics
- `identify_strata()` - Classify each sample into strata:
  - Single vs multi-subsystem
  - Has regulation-related subsystems
  - Has image-related subsystems
  - Complexity level
- `calculate_stratum_weights()` - Compute weights for each stratum
- `stratified_sample()` - Sample from each stratum with oversampling
- `generate_sample_ids()` - Assign unique IDs to sampled items
- `export_to_csv()` - Write CSV with columns:
  - `sample_id` - Unique identifier
  - `line_number` - Original line in JSONL
  - `query` - The query text
  - `original_labels` - JSON string of original subsystem labels
  - `corrected_labels` - Empty column for validator
  - `notes` - Column for validator notes
  - `validator` - Who validated (if multiple)
  - `stratum` - Which stratum this belongs to
  - `weight` - Sample weight
  - `complexity_score` - Calculated complexity
  - `validation_status` - pending/validated/skipped
- `export_config()` - Save sampling config to YAML for reference

**Output:**
- `data/validation_samples.csv` - CSV for human validators
- `data/validation_config.yaml` - Sampling parameters and metadata

**Error Handling:**
- Insufficient samples in a stratum
- Invalid stratification criteria
- CSV write errors

---

## Component 3: Statistical Testing

### Purpose
Read validated CSV, compare distributions between validated subset and remaining synthetic data.

### Key Features
- Import validated data and merge with original
- Run multiple statistical tests (Chi-square, KS-test, Jaccard, proportion tests)
- Generate reports with visualizations
- Calculate weighted correctness rates
- Provide remediation recommendations

### Configuration (YAML)
```yaml
statistical_testing:
  validated_csv: "data/validation_samples.csv"
  original_dataset: "data/train.jsonl"
  
  test_thresholds:
    chi_square_alpha: 0.05
    ks_test_alpha: 0.05
    jaccard_threshold: 0.85
    proportion_test_alpha: 0.01
    bonferroni_correction: true
    
  drift_detection:
    min_tests_significant: 2
    strong_drift_p_value: 0.001
    practical_impact_threshold: 0.10  # 10% absolute difference
    
  output:
    report_file: "data/validation_report.html"
    results_json: "data/validation_results.json"
    include_visualizations: true
```

### Implementation Details

**Functions/Modules Needed:**
- `load_validated_data()` - Read CSV, parse corrected labels
- `load_original_dataset()` - Read full train.jsonl
- `split_validated_remaining()` - Separate validated vs remaining samples
- `calculate_weighted_correctness()` - Implement weighted evaluation formula
- `calculate_inter_annotator_agreement()` - Cohen's kappa if multiple validators
- `chi_square_test()` - Test subsystem label frequencies
- `kolmogorov_smirnov_test()` - Test query complexity distributions
- `jaccard_similarity()` - Compare co-occurrence patterns
- `proportion_difference_test()` - Test single vs multi-subsystem ratios
- `detect_drift()` - Apply drift detection logic (combine test results)
- `identify_problematic_strata()` - Find which categories show drift
- `generate_report()` - Create HTML/Markdown report with:
  - Test results (p-values, statistics)
  - Distribution comparisons
  - Visualizations (plots, charts)
  - Recommendations for remediation
- `export_results()` - Save test results as JSON for programmatic use

**Output:**
- `data/validation_report.html` - Human-readable report
- `data/validation_results.json` - Machine-readable results
- Console output with summary

**Error Handling:**
- Missing validated samples
- Invalid label formats in CSV
- Statistical test failures
- Report generation errors

---

## Cross-Cutting Concerns

### Configuration Management
- Single YAML config file: `config/data_pipeline.yaml`
- Sections for each component
- Environment-specific overrides

### Logging
- Track all operations (generation, sampling, testing)
- Log errors and warnings
- Timestamp all operations

### Data Versioning
- Track dataset versions
- Track validation rounds
- Link validation results to dataset versions

### CLI Interface
- `python -m data_creation generate` - Run data generation
- `python -m data_creation sample` - Create validation samples
- `python -m data_creation test` - Run statistical tests
- `python -m data_creation validate` - Full pipeline

---

## Workflow Integration

### Step 1 → Step 2
- After generation, automatically check if sampling is needed
- Use current dataset size for sampling calculations

### Step 2 → Step 3
- Validators fill CSV and return it
- Import script reads CSV and validates format
- Merge with original data for testing

### Step 3 → Step 1 (Feedback Loop)
- Test results identify problematic strata
- Generate remediation recommendations
- Can trigger selective regeneration of specific categories

---

## File Structure

```
data_creation/
  __init__.py
  generate.py          # Component 1
  sample.py             # Component 2
  test.py               # Component 3
  utils/
    __init__.py
    data_loader.py
    format_validator.py
    complexity.py
    statistics.py
    report_generator.py

config/
  data_pipeline.yaml    # Main config file

docs/
  IMPLEMENTATION_PLAN.md  # This file
```

---

## Dependencies

- `pandas` - Data manipulation (CSV, JSONL)
- `numpy` - Statistical calculations
- `scipy` - Statistical tests
- `pyyaml` - Configuration management
- `matplotlib` / `plotly` - Visualizations (for reports)
- LLM API client (OpenAI, Anthropic, etc.) - For data generation


