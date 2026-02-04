# Task 1: Generation — Evaluation Guide

This README explains how to run the evaluation for **Task 1: Generation**, which evaluates automatically generated slide decks against their corresponding research papers.

This document is intentionally scoped **only to Task 1** and is separate from the main repository README.

## What This Evaluation Does

The generation evaluation compares:
- **Generated slide decks** (PDF format)
- **Reference research slide decks** (PDF format)
- **Reference research papers** (PDF format)

It computes generation metrics by extracting and comparing content between slides and papers.
> ⚠️ PowerPoint files (`.pptx`) are **not supported**. Slides must be exported to PDF before running evaluation.

## Input Preparation

### 1. Generated Slides

- Format: **PDF**
- One slide deck per paper
- Filenames must match the corresponding paper

Example:
```
gen_slides/
├── slide_001.pdf
├── slide_002.pdf
└── slide_003.pdf
```

### 2. Reference Papers

- Format: **PDF**
- One paper per slide deck
- Filenames must exactly match the paper IDs

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
- Filenames must match the corresponding paper

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
├── gen_slides/ # Generated slide PDFs
├── ref_slides/ # Reference slide PDFs
└── papers/ # Reference paper PDFs
```

Paths can be changed using command-line arguments (see below).

## Running the Evaluation

Once the PDFs are prepared, run:

```bash
python generation_evaluation.py \
  --slides_root /root/data/ref_slides \
  --papers_root /root/data/papers \
  --pipeline_root /root/data/gen_slides
```
## Arguments
Argument	Description
--slides_root	Directory containing reference slide deck PDFs
--papers_root	Directory containing reference paper PDFs
--pipeline_root	Directory containing generated slide deck PDFs and outputs

## Outputs

After completion, the pipeline output directory will contain json files corresponding to the final metric outputs.

Example:
```
gen_slides/
├── slide_001.pdf
├── slide_001_similarity_results.json
├── slide_002.pdf
└── slide_002_similarity_results.json
```
## Assumptions & Notes

- Slides and papers are matched one-to-one by filename/paper identifier
- PDFs should be text-readable (not scanned images)
- Large PDFs may increase runtime due to text extraction

## Common Issues

File not found errors
→ Check that filenames match exactly between slides and papers.

Empty or missing metrics
→ Ensure PDFs contain extractable text.

Path errors
→ Use absolute paths and verify directory existence.
