import sys, os
import glob
import argparse

import json

import logging
from loguru import logger

import urllib.parse

from multiprocessing import Process, Queue, Manager
import multiprocessing as mp
import threading

import enlighten

from omegacli import OmegaConf as omegacli_conf

from multiturn_pipeline import *


def batch_simulation(config):

    # Get API Keys as dictionary
    default_api_keys = omegacli_conf.create({"None": "dummy"})
    api_keys = config.get("api_keys", default_api_keys)
    api_keys_dict = omegacli_conf.to_container(api_keys, resolve=True)


    user_model_name = config.user_agent.user_model_name
    user_model_server = config.user_agent.user_model_server
    user_api_key_type = config.user_agent.user_api_key_type

    editor_model_name = config.editor_agent.editor_model_name
    editor_model_server = config.editor_agent.editor_model_server
    editor_api_key_type = config.editor_agent.editor_api_key_type



    agent_type = config.agent_type
    deck_list_folder = config.data_path.deck_list_path
    gt_slides_root = config.data_path.gt_slides_root


    selected_persona_name = config.user_agent.persona_name

    simulation_folder_name = config.simulation.simulation_name
    max_turns = config.simulation.max_turns
    
    start_idx = config.simulation.start_idx
    end_idx = config.simulation.end_idx


    failure_json = config.simulation.failure_json
    failure_json_path = os.path.join(deck_list_folder, failure_json) 


    remove_duplicate = config.remove_duplicate
    verbose = config.verbose

    simulation_max_tries = config.simulation.get("max_tries", 2)


    print('')
    print('*** User Agent model name : ', user_model_name)
    print('*** Editor Agent model name : ', editor_model_name)
    print('*** User Presona name : ', selected_persona_name)
    print('')


    # Read GT slides list
    # generate a GT deck dictionary per deck name
    gt_deck_name_path_map = {}
    if os.path.exists(gt_slides_root):
        gt_slide_files = glob.glob( os.path.join(gt_slides_root ,'*.pdf') )
        for gt_slide_file in gt_slide_files:
            gt_slide_file_name = os.path.basename(gt_slide_file)
            # file_stem = gt_slide_file_name.split('.')[0]
            file_path = Path(gt_slide_file_name)
            file_stem = file_path.stem

            deck_name = file_stem.removesuffix("_1")
            gt_deck_name_path_map[deck_name] = gt_slide_file

    def find_gt_slide_path(deck_name):
        gt_slide_path = ''
        if deck_name in gt_deck_name_path_map.keys():
            gt_slide_path = gt_deck_name_path_map[deck_name]

        return gt_slide_path

    def treat_duplicate_simulation(deck_list_folder, deck_name, simulation_folder_name):
        simulation_output_folder = os.path.join(deck_list_folder, deck_name, simulation_folder_name) # add sub folder
        simulation_backup_folder = os.path.join(deck_list_folder, deck_name, simulation_folder_name+'_old') # add sub folder
        if os.path.exists(simulation_output_folder): # simulation folder already exist
            print('--> Warning: the simulation folder already exist')
            if remove_duplicate: # remove old simulation folder
                try:
                    shutil.rmtree(simulation_output_folder)
                    message = f"Folder '{simulation_output_folder}' and its contents removed successfully."
                    print('--> ', message)
                except OSError as e:
                    # message = f"Error removing folder '{exp_full_path}': {e}"
                    pass
            else: # not removing, copy to backup
                if os.path.exists(simulation_backup_folder):
                    try:
                        shutil.rmtree(simulation_backup_folder)
                    except OSError as e:
                        pass
                # Rename the simulation folder to backup
                try:
                    # Rename the folder using shutil.move()
                    shutil.move(simulation_output_folder, simulation_backup_folder)
                    print(f"--> Directory '{simulation_output_folder}' renamed to '{simulation_backup_folder}' successfully.")
                except FileNotFoundError:
                    # print(f"Error: The source directory '{old_folder_name}' was not found.")
                    pass
                except Exception as e:
                    # print(f"An error occurred: {e}")
                    pass

        if os.path.exists(simulation_backup_folder): # simulation folder already exist
            print('--> Warning: the simulation backup folder already exist')
            if remove_duplicate: # remove old simulation folder
                try:
                    shutil.rmtree(simulation_backup_folder)
                    message = f"Folder '{simulation_backup_folder}' and its contents removed successfully."
                    print('--> ', message)
                except OSError as e:
                    # message = f"Error removing folder '{exp_full_path}': {e}"
                    pass

    # decide logger type
    is_loguru = False
    if agent_type == "AWorld":
        is_loguru = True

    # Collect slides
    deck_names = []
    # Read only previously failed simulations
    if config.failed_papers: 
        if os.path.exists(failure_json_path): 
            failure_dict = {}
            with open(failure_json_path, 'r') as file:
                failure_dict = json.load(file)

            if len(failure_dict) > 0:
                for fid in failure_dict.keys():
                    paper_stem = failure_dict[fid]
                    deck_names.append(paper_stem)
    else:
        # Read deck folders
        p = Path(deck_list_folder)
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

    # calculate number of decks to process
    num_of_papers = len(deck_names)
    if start_idx >=0: # and end_idx >=0:
        if end_idx >= start_idx: 
            num_of_papers = end_idx - start_idx + 1
        else:
            num_of_papers = num_of_papers - start_idx
    else:
        if end_idx >= 0:
            num_of_papers = end_idx 

    # to save failed paper deck name
    failed_paper_dict = {}

    # Enlighten bar
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('User Simulation',
                                    color='white_on_red',
                                    justify=enlighten.Justify.CENTER)
    BAR_FMT = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:d}/{total:d} ' + \
              u'[{elapsed}<{eta}, {interval:.0f} sec/{unit}{unit_pad}]'
    pbar = manager.counter(total=num_of_papers, desc='Slide Deck', unit='deck', color='green',
                        bar_format=BAR_FMT
                        )

    # Simulation look with decks
    for sidx, deck_name in enumerate(deck_names): # iterate decks

        if start_idx >= 0 and  sidx < start_idx:
            continue

        gt_slide_file_path = find_gt_slide_path(deck_name)

        print('')
        print('')
        print('================== Simulating deck : ', sidx, ' - ', deck_name, ' with GT deck : ', gt_slide_file_path)

        status_bar.update('Simulating deck : ' + deck_name, force=True) 

        deck_folder = os.path.join(deck_list_folder, deck_name) # add sub folder
        simulation_folder = os.path.join(deck_folder, simulation_folder_name)

        # Retry simulation
        is_success = False
        for tidx in range(simulation_max_tries):

            # treat old simulation folder, by default backup to old folder
            treat_duplicate_simulation(deck_list_folder, deck_name, simulation_folder_name)

            # Run simulaiton pipeline for one slide neck
            result = user_simulation_pipeline(deck_name, deck_folder, simulation_folder, agent_type, 
                                gt_slide_file_path, selected_persona_name, max_turns,
                                user_model_name=user_model_name, user_model_server=user_model_server, user_api_key_type= user_api_key_type,
                                editor_model_name=editor_model_name, editor_model_server=editor_model_server, editor_api_key_type= editor_api_key_type,
                                api_keys_dict=api_keys_dict,     
                                user_simulation_system_prompt='',
                                config=config,
                                enlighten_manager=manager,
                                verbose=verbose
                                )

            if 'success' in result.lower():
                # print('Done')
                is_success = True
                break
            else:
                # print('FAIL')
                pass


            if tidx+1 < simulation_max_tries:
                print('')
                print('================== Retry simulation  trial : ', tidx+2, '.')



        if not is_success:
            print('!---- Failed simulation : ', deck_name)
            failed_paper_dict.update({sidx: deck_name})            

            # save failure json
            if not config.no_save_failure:
                if len(failed_paper_dict) > 0:
                    if not os.path.exists(deck_list_folder): # create if not exist
                        os.makedirs(deck_list_folder)
                    with open(failure_json_path, "w") as f:
                        json.dump(failed_paper_dict, f, sort_keys=True, indent=4)
        else:
            print('!---- Succeeded simulation : ', deck_name)

        pbar.update()

        if end_idx >= 0 and end_idx >= start_idx and sidx >= end_idx:
            break


    manager.stop()

    # Remove previous failed json file if 
    if len(failed_paper_dict) <=0 and not config.no_save_failure:
        print('!---- No failed simulation')
        if os.path.exists(failure_json_path): # if previous failure json, then remove it
            # if old one exist, then backup            
            old_failure_file_name = failure_json.split('.')[0] + '_old.json'
            old_failure_json_path = os.path.join(deck_list_folder, old_failure_file_name)
            shutil.copy2(failure_json_path, old_failure_json_path)
            # Remove
            try:
                os.remove(failure_json_path)
                print(f"Old failure json File '{failure_json_path}' has been removed successfully.")
            except FileNotFoundError:
                print(f"Error: The file '{failure_json_path}' was not found.")
            except PermissionError:
                print(f"Error: Permission denied to delete the file '{failure_json_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        print('')
        print('')
        print('!---- Number of failed simulations : ', len(failed_paper_dict))
        for paper_id in failed_paper_dict.keys():
            print('  failed paper : ', paper_id, ' , ', failed_paper_dict[paper_id])





def main():

    parser = argparse.ArgumentParser(description="Batch User Simulation")

    parser.add_argument("--data_path.gt_slides_root", default="", help="Folder with slides PDFs")
    parser.add_argument("--data_path.deck_list_path", default="", help="initial deck list folder path")

    parser.add_argument("--simulation.simulation_name", default="simulation", help="Simulation name")
    parser.add_argument('--simulation.max_turns', type=int, default=2, help='maximum turn numbers')

    parser.add_argument("--user_agent.persona_name", default="balanced_editor", help="granular_analyst, balanced_editor, executive, creative_facilitator")

    parser.add_argument("--agent_type", default="OpenAIAgent", help="Agent type: OpenAIAgent, AWorld")

    parser.add_argument('--simulation.start_idx', type=int, default=0, help='start paper index, from 0')
    parser.add_argument('--simulation.end_idx', type=int, default=-1, help='end paper index, from 0. -1 means until the last one')

    parser.add_argument("--simulation.failure_json", default="simulation_failure.json", help="file name to save failure json")

    parser.add_argument("--config", required=True, help="config yaml file")

    # additional non-configurable argument options
    parser.add_argument("--start_deck_name", default="", help="deck_name to start")
    parser.add_argument("--end_deck_name", default="", help="deck_name to end")

    parser.add_argument("--reveal_path", default="", help="reveal package path")

    parser.add_argument('--failed_papers', action='store_true', help='Simulate for failed papers')
    parser.add_argument('--remove_duplicate', action='store_true', help='remove old simulation folder')
    parser.add_argument('--no_save_failure', action='store_true', help='not saving failure results') #default is False
    parser.add_argument('--verbose', action='store_true', help='verbose messages')
    # args = parser.parse_args()


    # Merge config with argments
    user_provided_args, default_args = omegacli_conf.from_argparse(parser)
    config_path = user_provided_args.config
    if config_path and config_path.endswith(('.yaml', '.yml')):
        yaml_config = omegacli_conf.load(config_path)
    else:
        yaml_config = omegacli_conf.create({})    
    config = omegacli_conf.merge(default_args, yaml_config, user_provided_args)

    # print(omegacli_conf.to_yaml(config))

    batch_simulation(config)


if __name__ == "__main__":
    main()
