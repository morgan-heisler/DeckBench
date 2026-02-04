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
slides/
├── paper_001.pdf
├── paper_002.pdf
└── paper_003.pdf
```

### 2. Reference Papers

- Format: **PDF**
- One paper per slide deck
- Filenames must exactly match the slide PDFs

Example:
```
papers/
├── paper_001.pdf
├── paper_002.pdf
└── paper_003.pdf
```

## Expected Directory Structure

By default, the evaluation assumes the following layout:

```
/root/data/Conference_Papers_Slides/eval/
├── slides/ # Generated slide PDFs
└── papers/ # Reference paper PDFs
```

Evaluation outputs will be written to: `/root/data/daesik/generated_eval_pdf/`


Paths can be changed using command-line arguments (see below).

## Running the Evaluation

Once the PDFs are prepared, run:

```bash
python generation_evaluation.py \
  --slides_root /root/data/Conference_Papers_Slides/eval/slides \
  --papers_root /root/data/Conference_Papers_Slides/eval/papers \
  --pipeline_root /root/data/daesik/generated_eval_pdf
```
## Arguments
Argument	Description
--slides_root	Directory containing generated slide deck PDFs
--papers_root	Directory containing reference paper PDFs
--pipeline_root	Output directory for evaluation artifacts and results

## Outputs

After completion, the pipeline output directory will contain:

- Parsed slide representations
- Parsed paper representations
- Intermediate evaluation artifacts
- Final generation metrics

Example:
```
generated_eval_pdf/
├── parsed_slides/
├── parsed_papers/
├── metrics.json
└── logs/
```
## Assumptions & Notes

- Slides and papers are matched one-to-one by filename
- PDFs should be text-readable (not scanned images)
- Large PDFs may increase runtime due to text extraction

## Common Issues

File not found errors
→ Check that filenames match exactly between slides and papers.

Empty or missing metrics
→ Ensure PDFs contain extractable text.

Path errors
→ Use absolute paths and verify directory existence.

##Task Scope

This README applies only to Task 1: Generation.
See the main repository README for:
- Overall benchmark description
- Other tasks
- Dataset details
- Submission format

