import os
import shutil
from pathlib import Path
import json

import logging
from loguru import logger
import sys
import copy
import filecmp
import glob
import re

import asyncio
from multiprocessing import Process, Queue

# Get the current directory of child_script.py
current_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from metrics.agents import *
from metrics import prompts as prompt
from utils import *

OPENAIAGENT_TIMEOUT = 700

def initialize_user_agent(system_prompt, mcp_tool_config={}, user_model_name="", user_model_server="", user_api_key="", agent_type="OpenAIAgent"):

    global OPENAIAGENT_TIMEOUT
    
    user_agent = None

    # Prepare User Agent
    name = "User Agent"
    description="I'm a agent simulating user to generate editing prompt."

    if agent_type == 'OpenAIAgent':
        user_agent = init_agent_openai(name=name, system_prompt= system_prompt,  
            mcp_config=mcp_tool_config, model_name=user_model_name, model_server=user_model_server, api_key=user_api_key, timeout = OPENAIAGENT_TIMEOUT, tool_timeout = 500)
    elif agent_type == 'AWorld':
        user_agent = init_agent_aworld(name=name, system_prompt= system_prompt, agent_prompt=None, description=description, 
                mcp_config=mcp_tool_config, model_name= user_model_name, model_server= user_model_server, api_key=user_api_key) 
    else:
        return 'Error : Invalid agent type!'

    return user_agent

def inference_user_simulation_turn(user_agent, user_prompt, agent_type='OpenAIAgent', user_simulation_system_prompt=''):

    agent_answer = ''

    # Run Inference
    if agent_type == 'OpenAIAgent':
        agent_answer = asyncio.run( inference_openai(user_agent, user_prompt) )
    elif agent_type == 'AWorld':
        agent_answer = inference_aworld(user_agent, user_prompt)

    return agent_answer

def extract_pdf_slide_md(pdf_path, output_folder=None, verbose=False):
    """
    Converts a PDF (of presentation slides) into md file.

    Returns:
        str: contains extracted markdown
    """
    import fitz  

    doc = fitz.open(pdf_path)

    # Extract page titles
    page_titles = []
    doc_title = doc.metadata.get("title", "No Title Found")
    toc = doc.get_toc()
    for level, title, page_num, *rest in toc:
        page_titles.append(title)

    # if output_folder:
    #     os.makedirs(output_folder, exist_ok=True)
    # else:
    #     output_folder = os.path.join(os.path.dirname(pdf_path), "images")
    #     os.makedirs(output_folder, exist_ok=True)

    if verbose:
        print('Page titles : ', page_titles)

    output_content = ''
    for i, page in enumerate(doc):

        slide_id = f"slide-{i+1}"

        # ---------- Page size ----------
        page_width = page.rect.width
        page_height = page.rect.height

        # ---------- Text ----------
        text = page.get_text("text") or ""

        if verbose:
            print(i, ' , Slide : ', slide_id)
            print('text : ', text)

        if i >= len(page_titles):
            # get first line when there is no title
            page_title = text.split('\n', 1)[0]
        else:
            page_title =page_titles[i].lstrip()

        if verbose:
            print('Page title : ', page_title)
            print('')
            print('')


        # Remove leading "Slide ?:"
        if ':' in page_title:
            if 'slide' in page_title.split(':')[0].lower():
                remaining_words_list = page_title.split(':')[1:]
                page_title = ' '.join(remaining_words_list)
                page_title = page_title.lstrip()


        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            ext = 'jpg' #base_image["ext"]
            image_name = f"![](images/{slide_id}_img{img_index}.{ext})"
            # image_bytes = base_image["image"]

            text = text + image_name+'\n'

            # img_path = os.path.join(output_folder, image_name)
            # with open(img_path, "wb") as f:
            #     f.write(image_bytes)
            # images.append(img_path)

        if i == 0:
            output_content += page_title
            output_content += '\n\n'
        else:
            output_content += "# "+ page_title
            output_content += '\n\n'
            output_content += text
            output_content += '\n\n'

    doc.close()
    return output_content

def extract_and_load_markdown_file_content(paper_extracted_data_file_path='', gt_slide_file_path='', gt_slide_extracted_file_path=''):

    # Read extracted md file
    paper_extract_md_content = ''
    if paper_extracted_data_file_path != '' and os.path.exists(paper_extracted_data_file_path):
        with open(paper_extracted_data_file_path, 'r') as file:
            paper_extract_md_content = file.read()

    if gt_slide_file_path != '' and os.path.exists(gt_slide_file_path):
        # Convert the slide pdf file to md file
        slide_content = extract_pdf_slide_md(gt_slide_file_path)
        # Save md to file
        if gt_slide_extracted_file_path != '':
            with open(gt_slide_extracted_file_path, 'w') as file:
                file.write(slide_content)

    gt_slide_extracted_content = ''
    if gt_slide_extracted_file_path != '' and os.path.exists(gt_slide_extracted_file_path):
        with open(gt_slide_extracted_file_path, 'r') as file:
            gt_slide_extracted_content = file.read()

    return paper_extract_md_content, gt_slide_extracted_content

def extract_json_field_from_string(agent_answer, json_key):

    # Extract only json output string
    match = re.search(r"({[\s\S]*})|(\[[\s\S]*\])", agent_answer)
    if match:
        # Get the matching JSON string (either group 1 or group 2)
        json_string = match.group(1) if match.group(1) else match.group(2)            
        try:
            # print('')
            # print('')
            # print('json string : ')
            # print(json_string)
            json_object = json.loads(json_string)

            # if isinstance(json_object, dict) and 'edited_html' in json_object.keys():
            #     json_string = json_object['edited_html']
            if isinstance(json_object, dict) and json_key in json_object.keys():
                json_string = json_object[json_key]
            agent_answer = json_string
        except json.JSONDecodeError as e:
            print(f"Error decoding extracted JSON: {e}")
            pass

    return agent_answer


def user_simulation_pipeline(deck_name, deck_folder, simulation_folder, agent_type,
                        gt_slide_file_path, selected_persona_name, max_turns,
                        user_model_name=None, user_model_server=None, user_api_key_type='None',
                        editor_model_name=None, editor_model_server=None, editor_api_key_type='None',
                        api_keys_dict={},     
                        user_simulation_system_prompt='',
                        config = None,
                        enlighten_manager=None,
                        verbose=False
                        ):  

    # Dynamic load Editor Agnet class
    editor_agent_class_path = 'custom.editor_agent.EditorAgent'
    if config is not None:
        editor_agent_class_path = config.editor_agent.editor_agent_class_path
    EditorAgent = import_class_from_string(editor_agent_class_path)

    # Initialize Simulation folder
    reveal_path = config.get("reveal_path", "")
    EditorAgent.initialize_simulation_folder(deck_name, deck_folder, simulation_folder, reveal_path) 

    # Use filesystem tool to read data from files
    use_filesystem_mcp_tool_for_user = True

    # Maximum trials of editor agent
    editor_max_tries = config.editor_agent.get("max_tries", 2)

    if not os.path.exists(simulation_folder):
        return 'Error : simulation folder not exist'



    # Copy original paper extracted markdown file to simulation folder
    paper_extracted_file_name = 'extracted.md'

    # Get API keys
    user_api_key = 'dummy'
    if user_api_key_type in api_keys_dict.keys():
        user_api_key = api_keys_dict[user_api_key_type]
    editor_api_key = 'dummy'
    if editor_api_key_type in api_keys_dict.keys():
        editor_api_key = api_keys_dict[editor_api_key_type]


    # Prepare extracted md data file and content
    paper_extracted_data_file_path = os.path.join(simulation_folder, paper_extracted_file_name)

    # Read paper extracted md file and filter out invalid sections
    if os.path.exists(paper_extracted_data_file_path):
        extract_prompt = ''
        with open(paper_extracted_data_file_path, 'r') as file:
            extract_prompt = file.read()
        if extract_prompt == '':
            return '', 0
        # filter out references and rest of content
        if len(extract_prompt) > 70000:
            extract_prompt = filter_extracted_markdown(extract_prompt, filter_keywords = ['references', 'reference', 'acknowledgments', 'acknowledgment', 'appendix', 'appendices'])
            # if still big, then filter more
            if len(extract_prompt) > 70000:
                extract_prompt = filter_extracted_markdown(extract_prompt, filter_keywords = ['ablation'])

        with open(paper_extracted_data_file_path, "w") as file:
            file.write(extract_prompt)


    # Get GT deck extracted file path
    gt_slide_extracted_file_name = Path(gt_slide_file_path).stem + '.md'
    gt_slide_extracted_file_path = os.path.join(simulation_folder, gt_slide_extracted_file_name)
    # Convert GT slide pdf to md
    if gt_slide_file_path != '' and os.path.exists(gt_slide_file_path):
        # Convert the slide pdf file to md file
        slide_content = extract_pdf_slide_md(gt_slide_file_path)
        # Save md to file
        if gt_slide_extracted_file_path != '':
            with open(gt_slide_extracted_file_path, 'w') as file:
                file.write(slide_content)


    # Read paper extracted md and GT slide md content
    paper_extract_md_content, gt_slide_extracted_content = extract_and_load_markdown_file_content(paper_extracted_data_file_path,
        gt_slide_extracted_file_path=gt_slide_extracted_file_path)



    ################################################################################
    #### Initialize User Agent
    #### Prepare system prompts for user agent
    if use_filesystem_mcp_tool_for_user: # not using filesystem
        if user_simulation_system_prompt == '':   
            user_simulation_system_prompt = prompt.user_simulation_system_prompt
    else: # use string prompt
        if user_simulation_system_prompt == '':        
            user_simulation_system_prompt = prompt.user_simulation_system_prompt_without_filesystem    

    # Prepare mcp tool for user agent
    if use_filesystem_mcp_tool_for_user: # not using filesystem
        # user_mcp_tool_config = configure_mineru_mcp_tool()
        user_mcp_tool_config = filesystem_mcp_config        
    else: # use string prompt
        user_mcp_tool_config = {}

    ## Prepare User Agent System prompt
    if use_filesystem_mcp_tool_for_user: # use filesystem tool to use files
        paper_prompt = f"\n[paper_info_path] : {paper_extracted_data_file_path}\n"
        gold_deck_prompt = f"\n[gold_deck_path] : {gt_slide_extracted_file_path}\n"
        user_simulation_system_prompt += paper_prompt
        user_simulation_system_prompt += gold_deck_prompt
    else:
        paper_prompt = f"\n[paper_content] : {paper_extract_md_content}\n"
        gold_deck_prompt = f"\n[gold_deck_content] : {gt_slide_extracted_content}\n"
        user_simulation_system_prompt += paper_prompt
        user_simulation_system_prompt += gold_deck_prompt

    #### Initialize  User Agent
    user_agent = initialize_user_agent(system_prompt= user_simulation_system_prompt, mcp_tool_config=user_mcp_tool_config,
                    user_model_name=user_model_name, user_model_server=user_model_server, user_api_key=user_api_key, agent_type=agent_type )
    if user_agent is None:
        return 'Error : Failed to create a User Agent.'

    # Select persona
    selected_persona = prompt.personas[selected_persona_name]



    ################################################################################
    #### Initialize Editor Agent
    editoragent = EditorAgent(deck_name, simulation_folder, selected_persona_name)
    editoragent.initialize_editor_agent_and_system_prompt(paper_extract_md_content, model_name=editor_model_name, model_server=editor_model_server, api_key=editor_api_key, agent_type=agent_type)


    # create an enlighten progress bar for multiple turns of simulaiton
    BAR_FMT = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:d}/{total:d} ' + \
              u'[{elapsed}<{eta}, {interval:.0f} sec/{unit}{unit_pad}]'
    turns_pbar = enlighten_manager.counter(total=max_turns, desc='    Simulation turns', unit='turn', color='yellow', bar_format=BAR_FMT, leave=False)


    ################################################################################
    ## FOR EACH TURN
    turn_idx = 0
    while turn_idx < max_turns:

        print('')
        print(f'------------------ Turn {turn_idx+1} : User Agent')

        ##========================================================================##
        ## USER SIMULATION AGENT : Inference
        # Read previous deck file
        previous_deck_md_file_path = editoragent.previous_deck_md_file_path
        previous_deck_content = ''
        if previous_deck_md_file_path != '' and os.path.exists(previous_deck_md_file_path):
            with open(previous_deck_md_file_path, 'r') as file:
                previous_deck_content = file.read()
        user_prompt = f"""{{
            "previous_deck": "{previous_deck_content}", 
            "user_persona": {selected_persona}
            }}
            """
        #### Inference with User Simulation Agent
        agent_answer = inference_user_simulation_turn(user_agent, user_prompt, agent_type=agent_type) #, user_simulation_system_prompt=user_simulation_system_prompt)

        # check if User Agent generate result properly
        if len(agent_answer) < 10:
            print((f"Failed with User Agent at Turn{turn_idx+1}"))
            break

        # Extract only json output string
        agent_answer = extract_json_field_from_string(agent_answer, json_key='editing_request')

        # Save editing request prompt
        user_editing_prompt = agent_answer
        edit_txt_file = os.path.join(simulation_folder, f'edit_{turn_idx+1}.txt')
        with open(edit_txt_file, 'w') as f:
            f.write(user_editing_prompt)


        ##========================================================================##
        ## EDITING AGENT : Inference
        print('')
        print(f'------------------ Turn {turn_idx+1} : Editor Agent')

        # Compose user query prompt with editing request
        editoragent.compose_user_prompt_turn(user_editing_prompt)

        #### Inference with Editor Agent
        is_success_turn = False
        for tidx in range(editor_max_tries): # multiple try editing

            previous_deck_file_path, previous_deck_md_file_path, is_success_turn = \
                editoragent.inference_editing_turn( turn_idx, agent_type=agent_type, verbose=verbose)

            if is_success_turn:
                break

            if tidx+1 < editor_max_tries:
                print('')
                print('------------------ Retry editor agent  trial : ', tidx+2, '.')


        # if simulation was not successful then stop multiturn
        if not is_success_turn:
            print(f"Error at Turn{turn_idx+1}. Stop here.")
            break


        turn_idx += 1


        turns_pbar.update()

    turns_pbar.close()

    # Check if simulation successful
    minimum_max_turns = 0 # 0 means any number of simulated max_turns are allowed. doesn't check max_turns
    is_success_multiturn = EditorAgent.is_simulation_successful(deck_name, simulation_folder, minimum_max_turns=minimum_max_turns, verbose=verbose)

    # Save simulation info dict
    if is_success_multiturn:
        return 'Success'

    else:
        return 'Error: simulation failed!'


