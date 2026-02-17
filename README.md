
<h1 align="center">
DECKBench: A Benchmark for Multi-Agent Slide Generation and Editing
</h1>
<p align="center">
<a href="https://arxiv.org/abs/2602.13318">
<img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red" alt="arXiv Paper">
</a>
<a href="https://huggingface.co/datasets/mheisler/DeckBench">
<img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow?logo=huggingface" alt="HuggingFace Models">
</a>
<!-- [![Hugging aFace](https://img.shields.io/badge/huggingface-Dataset-FFD21E?logo=huggingface)](https://huggingface.co/datasets/mheisler/DeckBench) -->
</p>

This repository contains the official benchmark and evaluation code for **DECKBench**, a reproducible benchmark for **academic paper–to–slide generation and multi-turn slide editing**.

DECKBench evaluates the *full presentation workflow*, from converting long research papers into slide decks to iteratively refining those decks through natural-language editing instructions. The benchmark is designed for evaluating **LLM- and Agent-based systems** under realistic, multi-turn conditions.

📄 **Paper**: *DECKBench: Benchmarking Multi-Agent Slide Generation and Editing from Academic Papers*  
🧪 **Status**: KDD 2026 submission  
📦 **Release**: Post-submission / arXiv

---

## Overview

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
└── requirements.txt
```

---

## Data Access and Licensing

### Important Note

**This repository does not redistribute conference papers or presentation slides.**

Due to licensing restrictions, we instead provide:
- metadata json file for each paper and slides link.

Users are responsible for complying with the original licenses of the retrieved materials.
The paper and slides links are provided in `data/paper_slide_urls.json`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/morgan-heisler/DeckBench.git
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
#### Installation for HTML to PDF conversion
- **Reveal.js HTML Presentation Framework** : To covert HTML to PDF, a HTML presentaiton framework **Reveal** (https://github.com/hakimel/reveal.js) is required. The argument **reveal_path** should be provided to use Reveal framework. To match the same style of the provided initial slide decks, the css file and background image file are provided under **Reveal_extra**.
- **decktape** : Please follow the installation guideline at https://github.com/astefanutti/decktape.
- Optional **katex_path** : convert math formula with local Katex (https://github.com/KaTeX/KaTeX/releases) if math conversion fails.

### 4. Install Agent Frameworks and MCP tools
- The default agent framework used is OpenAIAgent. Please follow the installation guideline at https://github.com/openai/openai-agents-python.
- Alternative agent framework supported is AWorld. Please follow the installation guideline at https://github.com/inclusionAI/AWorld.
- File system MCP tool is required for the simulation pipeline. Please follow the installation guideline at https://github.com/MarcusJellinghaus/mcp_server_filesystem

### 5. Download Required Models
The following models are required to calculate embeddings for metric calculation. Download them from Hugging Face (~5.5GB).  
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- clip-vit-base-patch32: https://huggingface.co/openai/clip-vit-base-patch32
- gpt2: https://huggingface.co/openai-community/gpt2

### 6. Configure API Keys

Export your API key (for example, `OPENAI_API_KEY`) as environment variable with your actual API key.
The API keys are used for simulation pipeline and evaluaiton scripts. The default key is OpenAI GPT key, but you can configure to use difference service and key in each configuraiton file(YAML).
- simulation_pipeline/custom/config.yaml : configure `api_keys` to add service names(for example, GPT) and keys. The keys can be retrieved from environment variables. 
- evaluation_config.yaml : configure `api_keys` to add service names(for example, GPT) and keys. The keys can be retrieved from environment variables.
---

## Evaluation for Task 1: Slide Generation
### Task Definition

#### Input
- Full academic paper (PDF or structured text)

#### Output
- A complete slide deck (HTML, Latex or PPTX).
- The output deck should be converted to PDF file for evaluation.

The generated deck does not need to match the reference slides exactly in length or ordering.
This repository provides evaluation scripts, and not providing the generation scripts. Instead, the generated decks by the baseline method is provided via Hugging Face. [![Hugging Face](https://img.shields.io/badge/huggingface-Dataset-FFD21E?logo=huggingface)](https://huggingface.co/datasets/mheisler/DeckBench)


#### Running Generated Deck Evaluation

```
python generation_evaluation.py \
  --data_path.gt_slides_root /root/data/ref_slides \
  --data_path.papers_root /root/data/papers \
  --data_path.deck_list_path /root/data/gen_pdf_slides \
  --output_folder /root/data/gen_eval_output \
  --config evaluation_config.yaml \
  --save_analysis_output
```
#### Arguments
Argument	Description
- --data_path.gt_slides_root	: Directory containing reference slide deck PDFs
- --data_path.papers_root	: Directory containing reference paper PDFs
- --data_path.deck_list_path	: Directory containing generated slide deck PDFs
- --output_folder : Directory to save evaluation output files for all decks (json file per deck)
- --config : Configuration YAML for evaluation
- --save_analysis_output : if set, output final summary result file(generation_metrics.csv) under output_folder/analysis

For more information, please see the separate README with a full breakdown.

  ---
  

## Evaluation for Task 2: Multi-Turn Slide Editing
### Task Definition

#### Input (per turn)
- Current slide deck
- Natural-language editing instruction*
  
*To evaluate the multi-turn slide editign, editing instructions are generated by a simulated user agent that compares the deck at each turn against ground-truth slides.

#### Output
- Updated slide deck

Editing is evaluated over multiple turns, reflecting realistic revision workflows.

### Stage 1: Running Multi-turn User Simulation
User simulation generates editing instruction per each turn based on a selected **persona**.
The editor agent takes the simulated editing instruction to edit the previous slide deck at each turn.

```
python simulation_pipeline/multiturn_simulation.py \
  --data_path.gt_slides_root /root/data/ref_slides \
  --data_path.deck_list_path /root/data/gen_slides \
  --simulation.simulation_name simulation_1 \
  --simulation.max_turns 5 \
  --user_agent.persona_name balanced_editor \
  --config simulation_pipeline/custom/config.yaml
```
#### Arguments
Argument	Description
- --data_path.gt_slides_root	: Directory containing reference slide deck PDFs
- --data_path.deck_list_path	: Directory containing initial slide deck PDFs
- --simulation.simulation_name : Name of each simulation, a subfolder with simulation will be generated
- --simulation.max_turns : Maximum turn number
- --user_agent.persona_name balanced_editor : Persona name
- --config : Configuration YAML for simulation

The simulation supports the following personas:
- granular_analyst
- balanced_editor (**default**)
- executive

### Stage 2: PDF Conversion for Multi-turn Simulation
The baseline editor's simulated slide decks are generated as HTML and must be converted to PDF before evaluation. If your editor outputs PDFs, this step is unnecessary.

```
python simulation_pipeline/custom/convert_html_to_pdf.py \
  --deck_list_path /root/data/gen_slides \
  --output_path /root/data/sim_slides \
  --reveal_path /root/Reveal/reveal.js \
  --katex_path /root/Reveal/reveal.js/katex \
  --simulation_name simulation_1 \
  --multiturn
```
#### Arguments
Argument	Description
- --data_path.deck_list_path	: Directory containing simulated slide deck PDFs
- --output_path : Directory to save converted slide deck PDFs
- --reveal_path : Reveal package path
- --katex_path : (Optional) Katex local path, used for math formula conversion
- --simulation.simulation_name : Name of each simulation
- --multiturn : set to make it multiturn conversion

### Stage 3: Multi-turn Editing Evaluation

The final stage evaluates the multi-turn edited slide PDFs against ground-truth slides and papers.
```
python multiturn_evaluation.py \
  --data_path.papers_root /root/data/papers \
  --data_path.gt_slides_root /root/data/ref_slides \
  --data_path.deck_list_path /root/data/sim_slides \
  --output_folder /root/data/sim_eval_output \
  --config evaluation_config.yaml \
  --save_analysis_output
```
#### Arguments
Argument	Description
- --data_path.papers_root	: Directory containing reference paper PDFs
- --data_path.gt_slides_root	: Directory containing reference slide deck PDFs
- --data_path.deck_list_path	: Directory containing generated edited deck PDFs
- --output_folder : Directory to save evaluation output files for all decks (json file per deck)
- --config : Configuration YAML for evaluation
- --save_analysis_output : if set, output final summary result file(baseline_relative_rate_summary.csv) under output/analysis

For more information, please see the separate README with a full breakdown.

### Implementing/Evaluating Custom Editing Agents

This repository provides a baseline Editor Agent implemented in **simulation_pipeline/custom** folder. 
- A custom agentic editing system can be implemented by inheriting abstract class **EditorAgentBase** implemented in **simulation_pipeline/editor_agent_base.py**. The baseline Editor Agent **EditorAgent** is implemented in the script **simulation_pipeline/custom/editor_agent.py**, and you can refer to this example to implement your own editor agent for evaluation.
- By implementing a new custom editor agent and any necessary scripts(ex: PDF conversion code) in **simulation_pipeline/custom** folder, the required changes can be limited to this folder, minimizing impact to other parts of the repository.
- In **simulation_pipeline/custom/config.yaml**, the argument **editor_agent.editor_agent_class_path** should be updated to specify the path to the implemented subclass(ex: CustomEditorAgent) for dynamic class importing from the simulation script. 

---

## Citation
```
@misc{jang2026deckbenchbenchmarkingmultiagentframeworks,
      title={DECKBench: Benchmarking Multi-Agent Frameworks for Academic Slide Generation and Editing}, 
      author={Daesik Jang and Morgan Lindsay Heisler and Linzi Xing and Yifei Li and Edward Wang and Ying Xiong and Yong Zhang and Zhenan Fan},
      year={2026},
      eprint={2602.13318},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.13318}, 
}
```
---

## License

Code is released under the MIT License.

Dataset metadata and scripts are provided for research purposes only. Users must comply with the licenses of the original papers and slides.

---

## Contact

For questions or issues, please open a GitHub issue.




