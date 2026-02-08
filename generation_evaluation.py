#!/usr/bin/env python3
"""
generation_evaluation.py

Usage:
    python generation_evaluation.py --deck_list_path PATH --gt_slides_root PATH --papers_root PATH

Outputs:
    - Prints matching pairs and scores for slide-level matching
    - Saves results to slide_similarity_results.json

Dependencies:
    pip install beautifulsoup4 lxml sentence-transformers transformers torch torchvision Pillow scipy tqdm requests fastdtw PyMuPDF dtaidistance
    apt-get install tidy
    pip install --upgrade pymupdf

Models Needed:
    GPT-2, all-MiniLM-L6-v2, CLIP-ViT-base-patch32
"""

import os
import re
import shutil
import argparse
import asyncio
from pathlib import Path
import glob

import logging


# Third-party imports
import torch
from tqdm import tqdm
import enlighten

from omegacli import OmegaConf as omegacli_conf

from metrics.evaluator import SlideEvaluator
from utils import safe_copy
from analysis.analyze_generation import *

# Transformers logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()  # suppress warnings/info

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch Slide Evaluator")
    parser.add_argument("--data_path.papers_root", default="", help="Folder with paper PDFs")
    parser.add_argument("--data_path.gt_slides_root", default="", help="Folder with ground-truth deck PDFs")
    parser.add_argument("--data_path.deck_list_path", default="", help="Root folder containing generated deck PDFs")
    parser.add_argument("--output_folder", default="./generation_evaluation_results", help="Where to save results")

    parser.add_argument("--embedding_models.text_model", default="/root/data/Agent/metric_models/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_models.clip_model", default="/root/data/Agent/metric_models/clip-vit-base-patch32")
    parser.add_argument("--embedding_models.perplexity_model", default="/root/data/Agent/metric_models/gpt2")

    parser.add_argument('--evaluation.start_idx', type=int, default=0, help='start paper index, from 0')
    parser.add_argument('--evaluation.end_idx', type=int, default=-1, help='end paper index, from 0')

    parser.add_argument("--agent_type", default="OpenAIAgent", help="Agent type: OpenAIAgent, AWorld")

    parser.add_argument("--start_deck_name", default="", help="deck_name to start")
    parser.add_argument("--end_deck_name", default="", help="deck_name to end")

    parser.add_argument('--save_analysis_output', action='store_true', help='verbose messages')

    parser.add_argument("--config", required=True, help="config yaml file")

    # args = parser.parse_args()

    # Merge config with argments
    user_provided_args, default_args = omegacli_conf.from_argparse(parser)
    config_path = user_provided_args.config
    if config_path and config_path.endswith(('.yaml', '.yml')):
        yaml_config = omegacli_conf.load(config_path)
    else:
        yaml_config = omegacli_conf.create({})    
    config = omegacli_conf.merge(default_args, yaml_config, user_provided_args)



    # Get API Keys as dictionary
    default_api_keys = omegacli_conf.create({"None": "dummy"})
    api_keys = config.get("api_keys", default_api_keys)
    api_keys_dict = omegacli_conf.to_container(api_keys, resolve=True)
 
    # Set LLM Judge API Key
    llm_judge_api_key = 'dummy'
    if config.llm_judge.llm_judge_api_type in api_keys_dict.keys():     
        llm_judge_api_key = api_keys_dict[config.llm_judge.llm_judge_api_type]
    # else:
    #     print('Warning : API key type is invalid. Can not find api_key!')



    # Initialize evaluator
    evaluator = SlideEvaluator(
        text_model_path = config.embedding_models.text_model_path,
        clip_model_path = config.embedding_models.clip_model_path,
        perplexity_model_path = config.embedding_models.perplexity_model_path,

        llm_judge_model_name =  config.llm_judge.llm_judge_model_name, #"deepseek-v3.1", #"gpt-5" ,
        llm_judge_model_server = config.llm_judge.llm_judge_model_server, #"https://api.modelarts-maas.com/v1", # "https://api.openai.com/v1" , 
        llm_judge_api_key = llm_judge_api_key,
        agent_type = config.agent_type #'OpenAIAgent'
    )

    print('SlideEvaluator initialized!')


    deck_list_path = config.data_path.deck_list_path
    papers_root = config.data_path.papers_root
    gt_slides_root = config.data_path.gt_slides_root

    output_folder_path = config.output_folder

    prefix0 = config.deck_name_prefix0

    start_idx = config.evaluation.start_idx
    end_idx = config.evaluation.end_idx


    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)


    # Collect all generated slide files (PDF)
    deck_files = []
    for root, _, files in os.walk(deck_list_path):
        for f in files:
            # if re.match(r"slide_([\w-]+)\.(html|pdf)$", f):  # <-- match both
            # prefix0 = EditorAgent.get_deck_name_prefix0()
            if re.match( prefix0 + r"([\w-]+)\.pdf$", f):  # <-- match only pdf
                deck_files.append(os.path.join(root, f))
    deck_files.sort()
    print(f"Found {len(deck_files)} generated slide decks (PDF).")

    # deck_files2 = glob.glob( os.path.join(deck_list_path , '*.pdf') )
    # deck_files2.sort()
    # print(f"Found {len(deck_files2)} generated slide decks (HTML or PDF).")


    # if start_deck_name is provided, then update start_idx
    if config.start_deck_name != '':        
        for sidx, file_path in enumerate(deck_files): # iterate slides
            file_name = os.path.basename(file_path)
            the_path = Path(file_name)
            file_stem = the_path.stem
            if file_stem == config.startstart_deck_name_slide_name:
                start_idx = sidx
                print('= Start index is decided as ', start_idx, ' with paper name : ', config.start_deck_name)
                break
    # if end_deck_name is provided, then update end_idx
    if config.end_deck_name != '':        
        for sidx, file_path in enumerate(deck_files): # iterate slides
            file_name = os.path.basename(file_path)
            the_path = Path(file_name)
            file_stem = the_path.stem
            if file_stem == config.end_deck_name:
                end_idx = sidx
                print('= End index is decided as ', end_idx, ' with paper name : ', config.end_deck_name)
                break




    num_of_papers = len(deck_files)
    if start_idx >=0:
        if end_idx >= start_idx: 
            num_of_papers = end_idx - start_idx + 1
        else:
            num_of_papers = num_of_papers - start_idx
    else:
        if end_idx >= 0:
            num_of_papers = end_idx 



    papers_with_issues = []


    # Enlighten bar
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('Generation Evaluation',
                                    color='white_on_red',
                                    justify=enlighten.Justify.CENTER)
    pbar = manager.counter(total=num_of_papers, desc='Evaluation', unit='ticks', color='green')


    # Process files sequentially (reverse order optional)
    for sidx, slide_path in enumerate(deck_files):
        if start_idx >= 0 and  sidx < start_idx:
            continue

        print("\n==========================================")
        print(f"Processing: {slide_path}")

        status_bar.update('Evaluate deck : ' + slide_path, force=True) # or status_bar.refresh()

        # Extract paper ID
        # match = re.search(r"slide_([\w-]+)\.(html|pdf)$", slide_path)
        # prefix0 = EditorAgent.get_deck_name_prefix0()
        match = re.search(prefix0 + r"([\w-]+)\.(html|pdf)$", slide_path)
        if not match:
            print("❌ Could not extract ID, skipping.")
            continue
        paper_id = match.group(1)

        # Build paths
        gt_pdf = os.path.join(gt_slides_root, f"{paper_id}_1.pdf")
        paper_pdf = os.path.join(papers_root, f"{paper_id}.pdf")

        if not os.path.exists(gt_pdf):
            print(f"❌ GT slide PDF missing: {gt_pdf}")
            continue
        if not os.path.exists(paper_pdf):
            print(f"❌ Paper PDF missing: {paper_pdf}")
            continue

        # Run evaluation
        print("==== Running evaluation ====")
        evaluator.evaluate(gt_pdf, slide_path, paper_pdf, output_folder_path, verbose=False)

        pbar.update()

        if end_idx >= 0 and end_idx >= start_idx and sidx >= end_idx:
            break

    manager.stop()

    analysis_output = os.path.join(output_folder_path, 'analysis')
    metrics = analyze_folder(output_folder_path, analysis_output, config.save_analysis_output)

    print("\n🎉 Batch evaluation complete.")
    if papers_with_issues:
        print(f"Papers with issues: {papers_with_issues}")


if __name__ == "__main__":
    main()