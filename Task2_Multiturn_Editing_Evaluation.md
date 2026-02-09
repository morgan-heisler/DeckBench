# Task 2: Multi-Turn Slide Editing — Evaluation Guide

This README explains how to run the evaluation for **Task 2: Multi-Turn Slide Editing**, which evaluates automatically simulated and edited slide decks against their corresponding research papers.

This document is intentionally scoped **only to Task 2** and is separate from the main repository README.

## What This Evaluation Does

The evaluation compares:
- **Multi-turn edited slide decks** (PDF format)
- **Reference research slide decks** (PDF format)
- **Reference research papers** (PDF format)

It computes multiturn editing metrics by extracting and comparing content between slides and papers.
> ⚠️ PowerPoint files (`.pptx`) are **not supported**. Slides must be exported to PDF before running evaluation.

## Input Preparation

### 1. Multi-Turn Edited Slides

- Format: **PDF**
- One slide deck per paper
- File names must match the corresponding paper with prefix **slide_** and **paper ID**.

Example:
```
sim_slides/
├── 001
  ├── slide_001.json
  ├── slide_001_turn0.pdf
  ├── slide_001_turn1.pdf
  ├── slide_001_turn2.pdf
  ├── slide_001_turn3.pdf
  ├── slide_001_turn4.pdf
  └── slide_001_turn5.pdf
```

### 2. Reference Papers

- Format: **PDF**
- One paper per slide deck
- File names must exactly match the paper IDs.

Example:
```
papers/
├── 001.pdf
├── 002.pdf
└── 003.pdf
```

### 3. Reference Slides

- Format: **PDF**
- One slide deck per paper
- File names must match the corresponding paper IDs.To distinguish from paper file names, the file name includes suffix **_1**.

Example:
```
ref_slides/
├── 001_1.pdf
├── 002_1.pdf
└── 003_1.pdf
```

## Expected Directory Structure

By default, the evaluation assumes the following layout:

```
/root/data/
├── sim_slides/ # Edited slide PDFs
├── ref_slides/ # Reference slide PDFs
└── papers/ # Reference paper PDFs
```

Paths can be changed using command-line arguments.

## Running the Evaluation

Once the PDFs are prepared, run:

```bash
python multiturn_evaluation.py \
  --data_path.papers_root /root/data/papers \ #Directory containing reference paper PDFs
  --data_path.gt_slides_root /root/data/ref_slides \ #Directory containing reference slide deck PDFs
  --data_path.deck_list_path /root/data/sim_slides #Directory containing edited slide deck PDFs
  --output_folder /root/data/sim_eval_output #Directory to save evaluation output files for all decks (json file per deck)
  --config evaluation_config.yaml #Configuration YAML for evaluation
  --save_analysis_output # if set, output final summary result file(baseline_relative_rate_summary.csv) under output_folder/analysis
```
## Arguments
Argument	Description
- --data_path.papers_root	: Directory containing reference paper PDFs
- --data_path.gt_slides_root	: Directory containing reference slide deck PDFs
- --data_path.deck_list_path :	Directory containing generated slide deck PDFs
- --output_folder : Directory to save evaluation output files for all decks (json file per deck)
- --config evaluation_config.yaml : Configuration YAML for evaluation
- --save_analysis_output : if set, output final summary result file(generation_metrics.csv) under output/analysis

## Outputs

After completion, the output directory will contain json files corresponding to the final metric outputs.

Example:
```
sim_eval_output/
├── 001_multiturn_results.json
├── 002_multiturn_results.json
```
## Assumptions & Notes

- Slides and papers are matched one-to-one by filename/paper identifier
- PDFs should be text-readable (not scanned images)
- Large PDFs may increase runtime due to text extraction

## Common Issues

File not found errors
→ Check if file names match exactly between slides and papers.

Empty or missing metrics
→ Ensure PDFs contain extractable text.

Path errors
→ Use absolute paths and verify directory existence.
