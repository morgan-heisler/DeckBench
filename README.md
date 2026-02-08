# DECKBench: A Benchmark for Multi-Agent Slide Generation and Editing
[![Hugging Face](https://img.shields.io/badge/huggingface-Dataset-FFD21E?logo=huggingface)](https://huggingface.co/datasets/mheisler/DeckBench)

This repository contains the official benchmark and evaluation code for **DECKBench**, a reproducible benchmark for **academic paper–to–slide generation and multi-turn slide editing**.

DECKBench evaluates the *full presentation workflow*, from converting long research papers into slide decks to iteratively refining those decks through natural-language editing instructions. The benchmark is designed for evaluating **LLM- and Agent-based systems** under realistic, multi-turn conditions.

📄 **Paper**: *DECKBench: Benchmarking Multi-Agent Slide Generation and Editing from Academic Papers*  
🧪 **Status**: KDD 2026 submission  
📦 **Release**: Post-submission / arXiv

---

## Overview

Preparing academic slide decks is an iterative, design-intensive process. Existing benchmarks typically evaluate either:
- one-shot text-to-slide generation, or
- isolated slide editing tasks.

DECKBench unifies these perspectives by introducing benchmark for two tightly coupled tasks:

1. **Slide Generation**  
   Generate a complete academic slide deck from a full research paper.

2. **Multi-Turn Slide Editing**  
   Iteratively refine an existing slide deck in response to natural-language editing instructions.

The benchmark includes:
- curated paper–slide pairs as url links
- initial generated slide decks to reproduce gneraiton and multi-turn evaluation
- simulation pipeline code to generate multi-turn editing trajectories
- reference-free and reference-based evaluation metrics
- evaluation codes for both tasks: generation and multi-turn slide editing tasks
---

## Repository Structure
```
deckbench/
├── data/
│ └── paper_slide_urls.json # Paper and slide metadata including url links
│
├── analysis/
│ ├── analyze_generation.py # Slide-level metrics
│ └── analyze_multiturn.py # Layout & design heuristics
│
├── metrics/ # scripts and utils for calculating metrics
│
├── simulation_pipeline/ # user simulation and editing pipeline for multi-turn evaluation
│ └── custom/ # custom slide editor agent
│  ├── custom.yaml
│  ├── convert_html_to_pdf.python
│  └── editor_agent.py
│ ├── editor_agent_base.py
│ ├── multiturn_pipeline.py
│ └── multiturn_simulation.py
│
├── evaluation_config.yaml
├── generation_evaluation.py
├── multiturn_evaluation.py
├── utils.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Data Access and Licensing

### Important Note

**This repository does not redistribute conference papers or presentation slides.**

Due to licensing restrictions, we instead provide:
- metadata for each benchmark instance
- scripts to automatically retrieve publicly available PDFs and slides from official sources

Users are responsible for complying with the original licenses of the retrieved materials.

### Fetching the Data

```bash
python tools/fetch_data.py \
  --output_dir data/raw/
```

The script:

- downloads paper PDFs from official conference or author pages
- retrieves publicly hosted presentation slides when available
- validates checksums against benchmark metadata

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DeckBench.git
cd DeckBench
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

**Important**: Download the marker-pdf model for PDF processing:

```bash
python down_model.py
```

This will download the marker-pdf model to the `models/` directory (~2GB).

### 5. Configure API Keys

**Option 1: Use the provided template (Recommended)**

Rename the provided template file and add your API key:

```bash
# Rename the template file
mv ".env copy" .env

# Edit the .env file and add your OpenAI API key
nano .env  # or use your preferred editor
```

> **Note**: The `.env copy` file contains a template with all available configuration options. Simply rename it to `.env` and update the `OPENAI_API_KEY` value with your actual API key.

---

## Task 1: Slide Generation
### Task Definition

#### Input
- Full academic paper (PDF or structured text)

#### Output
- A complete slide deck (HTML or PPTX)

The generated deck does not need to match the reference slides exactly in length or ordering.

#### Running Slide Evaluation

```
python generation_evaluation.py \
  --slides_root /root/data/ref_slides \ #Directory containing reference slide deck PDFs
  --papers_root /root/data/papers \ #Directory containing reference paper PDFs
  --pipeline_root /root/data/gen_slides #Directory containing generated slide deck PDFs and output folder
```
For more information, please see the separate README with a full breakdown.

  ---
  

## Task 2: Multi-Turn Slide Editing
### Task Definition

#### Input (per turn)
- Current slide deck
- Natural-language editing instruction*
  
*Editing instructions are generated by a simulated user agent that compares intermediate decks against ground-truth final slides.

#### Output
- Updated slide deck

Editing is evaluated over multiple turns, reflecting realistic revision workflows.

### Stage 1: Running Multi-turn Slide Editing
User simulation generates multi-turn slide edits based on a selected **persona**.

```
python batch_user_simulation.py \
  --gt_slides_root /root/data/ref_slides \
  --generation_result_path /root/data/gen_slides \
  --experiment_folder test \
  --simulation_name simulation \
  --persona_name balanced_editor \
  --start_idx 0 \
  --end_idx -1 \
  --max_turns 5
```

The simulation supports the following personas:
- granular_analyst
- balanced_editor (**default**)
- executive

### Stage 2: PDF Conversion for Multi-turn Simulation
The current editor's simulated slide decks are generated as HTML and must be converted to PDF before evaluation. If your editor outputs PDFs this step is unnecessary.

```
python batch_html_to_pdf.py \
  --html_slide_root /root/data/Reveal/reveal.js/results_eval/daesik \
  --simulation_name simulation \
  --output_root /root/data/gen_slides \
  --start_idx 0 \
  --end_idx -1 \
  --multiturn
```

### Stage 3: Multi-turn Evaluation

The final stage evaluates the multi-turn simulated slide PDFs against ground-truth slides and papers.
```
python multiturn_evaluation.py \
  --gt_slides_root /root/data/ref_slides \
  --papers_root /root/data/papers \
  --slides_root /root/data/gen_slides \
  --output_folder /root/data/gen_slides \
  --start_idx 0 \
  --end_idx -1
```

---

## Citation
```
@inproceedings{deckbench2026,
  title     = {DECKBench: Benchmarking Multi-Agent Slide Generation and Editing from Academic Papers},
  author    = {authors},
  booktitle = {KDD 2026 Datasets and Benchmarks Track},
  year      = {2026}
}
```
---

## License

Code is released under the MIT License.

Dataset metadata and scripts are provided for research purposes only. Users must comply with the licenses of the original papers and slides.

---

## Contact

For questions or issues, please open a GitHub issue.




