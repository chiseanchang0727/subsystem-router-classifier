# Data Creation

This folder contains scripts and utilities for creating datasets for the subsystem router classifier.

## Structure

- Scripts for generating training/validation/test data
- Utilities for data validation and formatting
- Tools for creating paraphrases and augmentations

## Data Generation

Training data is generated using the prompt template in [GENERATION_PROMPT.md](./GENERATION_PROMPT.md). This prompt defines the task, subsystem definitions, labeling rules, and output format.

## Validation Strategy

After generating training data, human validation is performed using a sampling-based approach. See [VALIDATION.md](./VALIDATION.md) for details on the validation methodology.

## Usage

TBD - scripts will be added here for data creation.

