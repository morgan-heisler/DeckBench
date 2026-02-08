#!/usr/bin/env python3
"""
multiturn_evaluation.py - Deck-to-Deck Comparison
"""
import os
import re
import argparse
import asyncio
import json
import asyncio

from pathlib import Path
import glob

import enlighten

from omegacli import OmegaConf as omegacli_conf

from metrics.evaluator import SlideEvaluator
from utils import safe_copy
from analysis.analyze_multiturn import *


def compare_decks(evaluator, prev_deck_file, current_deck_file, current_turn_id=0, instruction='', deck_comparison_dict={}, queue=None, verbose=False):

    # def extract_id(path):
    #     m = re.search(r"slide_(\d+)\.html$", path)
    #     if m is None:
    #         m = re.search(r"slide_(\d+)\.pdf$", path)
    #     return int(m.group(1)) if m else -1

    evaluator.original_slides = prev_deck_file
    evaluator.instruction = instruction

    prev_turn_id = current_turn_id -1

    # turn_id = extract_turn_id(prev_deck_file)

    # if turn_id == -1:
    #     print(f"❌ Cannot extract turn ID from {prev_deck_file}, skipping")
    #     return results

    # print(f"\n--- Comparing previous slide vs current slide: Paper {turn_id} ---")

    ## Run previous deck evaluation
    prev_metrics = None
    if prev_turn_id in deck_comparison_dict.keys():
        prev_metrics = deck_comparison_dict[prev_turn_id]
    else:
        prev_metrics = evaluator.evaluate_turn2(
            # evaluator=evaluator,
            slide_path_gen=prev_deck_file,
            prev_metrics=prev_metrics,
            queue=queue,
            verbose=verbose     
        )

    ## Now current deck evaluation
    metrics = evaluator.evaluate_turn2(
        # evaluator=evaluator,
        slide_path_gen=current_deck_file,
        prev_metrics=prev_metrics,
        queue=queue,
        verbose=verbose,
    )

    deck_comparison_dict[current_turn_id] = metrics
    print(f"✅ Completed comparison for Paper {current_turn_id}")

    return deck_comparison_dict

def main():

    parser = argparse.ArgumentParser(description="Multi-deck Slide Evaluator")

    parser.add_argument("--data_path.papers_root", default="", help="Folder with paper PDFs")
    parser.add_argument("--data_path.gt_slides_root", default="", help="Folder with slides PDFs")
    parser.add_argument("--data_path.deck_list_path", default="", help="Root folder for generated deck list")
    parser.add_argument("--output_folder", default="./multiturn_evaluation_results", help="Where to save results")

    parser.add_argument('--evaluation.start_idx', type=int, default=0, help='start paper index, from 0')
    parser.add_argument('--evaluation.end_idx', type=int, default=-1, help='end paper index, from 0')

    parser.add_argument("--embedding_models.text_model_path", default="/root/data/Agent/metric_models/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_models.clip_model_path", default="/root/data/Agent/metric_models/clip-vit-base-patch32")
    parser.add_argument("--embedding_models.perplexity_model_path", default="/root/data/Agent/metric_models/gpt2")

    parser.add_argument("--llm_judge.llm_judge_model_name", default="")
    parser.add_argument("--llm_judge.llm_judge_model_server", default="")
    parser.add_argument("--llm_judge.llm_judge_api_type", default="None")

    parser.add_argument("--agent_type", default="OpenAIAgent", help="Agent type: OpenAIAgent, AWorld")

    parser.add_argument("--start_deck_name", default="", help="deck_name to start")
    parser.add_argument("--end_deck_name", default="", help="deck_name to end")
    parser.add_argument('--verbose', action='store_true', help='verbose messages')

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


    verbose = config.verbose


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
    output_folder = config.output_folder

    prefix0 = config.deck_name_prefix0

    start_idx = config.evaluation.start_idx
    end_idx = config.evaluation.end_idx

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)


    # Read GT slides list
    gt_slides_root = config.data_path.gt_slides_root
    gt_deck_name_path_map = {}
    if os.path.exists(gt_slides_root):
        gt_deck_files = glob.glob( os.path.join(gt_slides_root ,'*.pdf') )
        for gt_deck_file in gt_deck_files:
            gt_deck_file_name = os.path.basename(gt_deck_file)
            # file_stem = gt_deck_file_name.split('.')[0]
            file_path = Path(gt_deck_file_name)
            file_stem = file_path.stem

            deck_name = file_stem.removesuffix("_1")
            gt_deck_name_path_map[deck_name] = gt_deck_file



    def find_gt_deck_path(deck_name):
        gt_deck_path = ''
        if deck_name in gt_deck_name_path_map.keys():
            gt_deck_path = gt_deck_name_path_map[deck_name]

        return gt_deck_path



    def extract_turn_id_key(pdf_file_path):
        pdf_file = os.path.basename(pdf_file_path)
        file_path = Path(pdf_file)
        gen_file_name = file_path.stem
        # if turn_name is not included(it means original slide), add turn name 'turn0'
        tail_name = gen_file_name.split('_')[-1]

        if 'turn' in tail_name:
            return tail_name
        
        return 'turn0'




    # Read slide folders
    p = Path(deck_list_path)
    deck_names = [entry.name for entry in p.iterdir() if entry.is_dir()]
    deck_names.sort()

    # if start_deck_name is provided, then update start_idx
    if config.start_deck_name != '':        
        for sidx, deck_name in enumerate(deck_names): # iterate slides
            if deck_name == config.start_deck_name:
                start_idx = sidx
                print('= Start index is decided as ', start_idx, ' with slide name : ', config.start_deck_name)
                break
    # if end_deck_name is provided, then update end_idx
    if config.end_deck_name != '':        
        for sidx, deck_name in enumerate(deck_names): # iterate slides
            if deck_name == config.end_deck_name:
                end_idx = sidx
                print('= End index is decided as ', end_idx, ' with slide name : ', config.end_deck_name)
                break


    num_of_papers = len(deck_names)
    if start_idx >=0: # and end_idx >=0:
        if end_idx >= start_idx: 
            num_of_papers = end_idx - start_idx + 1
        else:
            num_of_papers = num_of_papers - start_idx
    else:
        if end_idx >= 0:
            num_of_papers = end_idx 



    # Enlighten bar
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('Multi-turn Evaluation',
                                    color='white_on_red',
                                    justify=enlighten.Justify.CENTER)
    pbar = manager.counter(total=num_of_papers, desc='Evaluation', unit='ticks', color='green')


    for sidx, deck_name in enumerate(deck_names): # iterate slides

        if start_idx >= 0 and  sidx < start_idx:
            continue

        gt_deck_file_path = find_gt_deck_path(deck_name)
        paper_file_path = os.path.join(papers_root, deck_name+'.pdf')
        deck_folder_path = os.path.join(deck_list_path, deck_name)


        print('')
        print('')
        print('========== Evaluate deck : ', deck_name, ' with GT deck : ', gt_deck_file_path, ' , paper : ', paper_file_path)


        status_bar.update('Evaluate deck : ' + deck_name, force=True) # or status_bar.refresh()


        # Read simulation json
        # json_file_name = 'slide_'+deck_name+'.json'
        # json_file_name = EditorAgent.get_deck_name_prefix(deck_name) + '.json'
        json_file_name = prefix0+deck_name+'.json'
                
        # json_file_name = 'simulation.json'
        simulation_json_file = os.path.join(deck_folder_path, json_file_name)
        simulation_info_dict = {}
        if os.path.exists(simulation_json_file):
            with open(simulation_json_file, 'r') as file:
                simulation_info_dict = json.load(file)
        # if simulation json is not available
        if len(simulation_info_dict) <=0:
            print('Error : multiturn info json not found : ', json_file_name)
            continue




        # Calculate Embeddings
        evaluator.calculate_paper_embeddings(paper_file_path, verbose=verbose)
        evaluator.calculate_gt_slide_embeddings(gt_deck_file_path, verbose=verbose)



        pdf_files = glob.glob( os.path.join(deck_folder_path , '*.pdf') )
        # pdf_files.sort()
        # sorted_list = sorted(pdf_files, key=lambda x: x.split()[1])
        pdf_files = sorted(pdf_files, key=extract_turn_id_key)


        deck_comparison_dict ={}

        prev_deck_file = ''
        current_deck_file = ''
        for pidx, pdf_file_path in enumerate(pdf_files): # iterate turns

            # get turn id
            pdf_file = os.path.basename(pdf_file_path)
            file_path = Path(pdf_file)
            gen_file_name = file_path.stem
            # if turn_name is not included(it means original slide), add turn name 'turn0'
            tail_name = gen_file_name.split('_')[-1]

            if 'turn' not in tail_name:
                turn_id = 0
            else:
                turn_id_str = tail_name.removeprefix('turn')
                if turn_id_str.isdigit():
                    turn_id = int(turn_id_str)
                else:
                    turn_id = 0

            print('')
            print('----Turn ID : ', turn_id,  ' - Evaluate slide pdf : ', pdf_file_path)

            if turn_id >= 0:
                # Get instruction
                instruction = ''
                turn_name = tail_name.capitalize()
                if turn_name in simulation_info_dict.keys():
                    turn_info_dict = simulation_info_dict[turn_name]
                    if "user_editing_prompt" in turn_info_dict.keys():
                        instruction = turn_info_dict["user_editing_prompt"]
                    elif "user_prompt" in turn_info_dict.keys():
                        instruction = turn_info_dict["user_prompt"]

                current_turn_id = turn_id
                current_deck_file = pdf_file_path

                print('Compare decks at turn id : ', current_turn_id, ' , with prev file : ', prev_deck_file, ' , with current file : ', current_deck_file)
                print('    : user instruction : ', instruction)

                deck_comparison_dict = compare_decks(evaluator, prev_deck_file, current_deck_file, current_turn_id, instruction, deck_comparison_dict)

                # update previous slide file path
                prev_deck_file = current_deck_file

        json_file_name = deck_name + '_' + 'multiturn_results.json'
        # Save results
        # os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, json_file_name) # "deck_comparison_results.json")
        with open(out_path, "w") as f:
            json.dump(deck_comparison_dict, f, indent=2)

        print(f"\n🎉 Multi-deck comparison complete. Results saved to {out_path}")

        pbar.update()

        if end_idx >= 0 and end_idx >= start_idx and sidx >= end_idx:
            break

    manager.stop()

    analysis_output = os.path.join(output_folder, 'analysis')
    analyze_folder(output_folder, analysis_output, save_output=config.save_analysis_output)


if __name__ == "__main__":

    main()