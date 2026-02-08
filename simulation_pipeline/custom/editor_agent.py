import os
import json

import sys
import filecmp
import glob
import shutil
from pathlib import Path

import asyncio

from bs4 import BeautifulSoup, Doctype
from html_to_markdown import convert

# Get the current directory of child_script.py
current_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project/)
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_parent_dir)

from metrics.agents import *
from utils import *
from simulation_pipeline.editor_agent_base import EditorAgentBase

OPENAIAGENT_TIMEOUT = 700
TOOL_TIMEOUT = 300

class EditorAgent(EditorAgentBase):

    def __init__(self, deck_name, simulation_folder, selected_persona_name):
        super().__init__(deck_name, simulation_folder, selected_persona_name)


    @staticmethod
    def copy_folder(source_folder_path, target_folder_path, overwrite=True, verbose=False):
        if not os.path.exists(target_folder_path) or overwrite:
            if os.path.exists(source_folder_path):
                try:
                    shutil.copytree(source_folder_path, target_folder_path, dirs_exist_ok=overwrite)
                    if verbose:
                        print(f"Successfully copied '{source_folder_path}' to '{target_folder_path}'")
                except FileExistsError:
                    if verbose:
                        print(f"Error: Destination directory '{target_folder_path}' already exists.")
                        print("Consider deleting the destination directory first or using dirs_exist_ok=True (Python 3.8+).")
                    else:
                        pass
                except Exception as e:
                    if verbose:
                        print(f"An error occurred during folder copy: {e}")
                    else:
                        pass
            else:
                if verbose:
                    print(f"Error: source folder not exist : {source_folder_path}")
                else:
                    pass
        else:
            if verbose:
                print(f"Error: target folder already exist : {source_folder_path}")
            else:
                pass

    @staticmethod
    def count_sections_images_in_html(filename, verbose=False):
        from bs4 import BeautifulSoup
            
        # Ensure the file exists before trying to open it
        if not os.path.exists(filename):
            if verbose:
                print(f"Error: The file '{filename}' was not found. In count_sections_images_in_html()")
            return 0, 0, 0

        folder_path = os.path.dirname(filename)
        invalid_img_paths = []
        try:
            # Open and read the HTML file
            with open(filename, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Parse the HTML content using Beautiful Soup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            if soup is None:
                if verbose:
                    print('Error: the html is not valid in count_sections_images_in_html()')
                return 0, 0, 0
            # Find all occurrences of the <section> tag
            sections = soup.find_all('section')

            div_slides = soup.find('div', class_='slides')

            if div_slides is None:
                if verbose:
                    print('Error: the html includes no slides in count_sections_images_in_html()')
                return 0, 0, 0

            div_images = div_slides.find_all('img')
            image_count = len(div_images)

            # find img path not exist
            for img in div_images:
                src = img.get('src')
                img_path = os.path.join(folder_path, src)

                if not os.path.exists(img_path):
                    invalid_img_paths.append(img_path)

            invalid_image_count = len(invalid_img_paths)

            # The number of tags is the length of the list returned by find_all
            return len(sections), image_count, invalid_image_count

        except Exception as e:
            if verbose:
                print(f"An error occurred: {filename} , {e}")
            return 0, 0, 0

    @staticmethod
    def is_html_file(filepath):
        from bs4 import BeautifulSoup, Doctype

        """
        Checks if a file is likely an HTML document using Beautiful Soup.
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return False

        try:
            # 1. Read the file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 2. Parse with Beautiful Soup using the 'html.parser'
            soup = BeautifulSoup(content, 'html.parser')

            # Find all occurrences of the <section> tag
            sections = soup.find_all('section')
            num_sections = len(sections)

            # 3. Check for top-level HTML tags
            # The 'html' and 'body' tags are fundamental to an HTML document.
            # If Beautiful Soup finds them, it has successfully parsed it as HTML.
            # Since some generate htmls don't include 'html' and 'body', so check 'head' and number of sections.
            # if soup.html is not None or soup.body is not None:
            #     return True
            if soup.head is not None or num_sections > 0:
                return True
            else:
                # Check if there is a DOCTYPE declaration (e.g., <!DOCTYPE html>)
                # which is also a strong sign of an HTML document
                if any(isinstance(item, Doctype) for item in soup.contents):
                    return True
                return False

        except Exception as e:
            print(f"BeautifulSoup : An error occurred during parsing: {filepath}, {e}")
            return False

    @staticmethod
    def is_generation_successful(html_file_path, min_file_size=5000, min_sections=4, min_images=0, allow_invalid_image_link=True, validate_html_content=True, verbose=False):
        is_success = True
        info_dict = {
                    'file_path': html_file_path,
                    'file_size': 0,
                    'sections': 0,
                    'images': 0,
                    'invalid_images': 0
                }
                
        if not os.path.exists(html_file_path):
            is_success = False
            if verbose:
                print('Fail : file not exist : ', html_file_path)
            return is_success, info_dict
        else: # check output file size

            if not EditorAgent.is_html_file(html_file_path): 
                is_success = False
                if verbose:
                    print('Fail : not an html file : ', html_file_path)
                return is_success, info_dict

            file_size_bytes = os.path.getsize(html_file_path)
            if file_size_bytes < min_file_size:
                is_success = False
                if verbose:
                    print('Fail : file size : ', file_size_bytes, ' is less than ', min_file_size, ' : ', html_file_path)

            info_dict['file_size'] = file_size_bytes

            if validate_html_content:
                num_sections, num_images, num_invalid_images = EditorAgent.count_sections_images_in_html(html_file_path, verbose=True)

                # Get number of sections, images, invalid images
                if is_success:
                    # Check number of sections, images, invalid images
                    num_valid_images = num_images - num_invalid_images # count only valid images
                    if num_sections < min_sections or num_valid_images < min_images:
                        is_success = False
                        if verbose:
                            if num_sections < min_sections:
                                print('Fail : section number : ', num_sections, ' is less than ', min_sections, ' : ', html_file_path)
                            if num_valid_images < min_images:
                                print('Fail : valid image number : ', num_valid_images, ' is less than ', min_images, ' : ', html_file_path)

                    elif not allow_invalid_image_link:
                        if num_invalid_images > 0:
                            is_success = False
                            if verbose:
                                print(f'Fail : {num_invalid_images} images are invalid :  {html_file_path}')

                info_dict = {
                        'file_path': html_file_path,
                        'file_size': file_size_bytes,
                        'sections': num_sections,
                        'images': num_images,
                        'invalid_images': num_invalid_images
                }

        return is_success, info_dict

    @staticmethod
    def save_content_as_file(code_output, html_file_path):
        # process if the output is json dictionary
        output_html_content = code_output
        try:
            output_dict = json.loads(code_output)
            if 'arguments' in output_dict.keys():
                arguments = output_dict['arguments']
                if 'content' in arguments.keys():
                    output_html_content = arguments['content']            
        except json.JSONDecodeError:
            output_html_content = code_output

        html_head_string = "<!DOCTYPE html>"
        index = output_html_content.find(html_head_string)
        if index != -1:  # Ensure the substring is found
            output_html_content = output_html_content[index:]

        html_tail_string = "</html>"
        index = output_html_content.find(html_tail_string)
        if index != -1:  # Ensure the substring is found
            output_html_content = output_html_content[:index + len(html_tail_string)+1]

        with open(html_file_path, 'w') as f:
            f.write(output_html_content)

    @staticmethod
    def get_md_file_path(deck_file_path):
        # make Markdown file path
        folder_path = os.path.dirname(deck_file_path)
        file_name = os.path.basename(deck_file_path)
        md_file_name = Path(file_name).stem + '.md'
        md_file_path = os.path.join(folder_path, md_file_name)

        return md_file_path

    @staticmethod
    def html_to_md_file(html_file_path, verbose=False):

        if not os.path.exists(html_file_path):
            if verbose:
                print(f"Error: File not found at {html_file_path}")
            return ''
        try:
            # make Markdown file path
            md_file_path = EditorAgent.get_md_file_path(html_file_path)

            # Open and read the HTML file
            with open(html_file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Parse the HTML content using Beautiful Soup
            soup = BeautifulSoup(html_content, 'html.parser')        
            if soup is None:
                if verbose:
                    print('Error: the html is not valid')
                return ''

            # Parse the HTML with Beautiful Soup
            soup = BeautifulSoup(html_content, 'html.parser')
            # Convert the soup object (or just the string) to Markdown
            markdown_content = convert(str(soup)) 

            if verbose:
                print(markdown_content)

            if len(markdown_content) > 0:
                with open(md_file_path, "w") as file:
                    file.write(markdown_content)
            else:
                if verbose:
                    print(f"Failed to convert to markdown : {html_file_path}")
                return ''

            return md_file_path

        except Exception as e:
            if verbose:
                print(f"An error occurred: {html_file_path} , {e}")
            return ''

    @staticmethod
    def _initialize_editor_agent(system_prompt, tool_config={},  
                    model_name=None, model_server=None, api_key='dummy', agent_type='OpenAIAgent'):  
        agent = None

        name = "Editor Agent"
        description="I'm a agent editing deck from paper markdown and old deck."

        if agent_type == 'OpenAIAgent':
            agent = init_agent_openai(name=name, system_prompt=system_prompt,  
                mcp_config=tool_config, model_name=model_name, model_server=model_server, api_key=api_key, timeout = OPENAIAGENT_TIMEOUT, tool_timeout = TOOL_TIMEOUT)
        elif agent_type == 'AWorld':
            agent = init_agent_aworld(name=name, system_prompt=system_prompt, agent_prompt=None, description=description, 
                    mcp_config=tool_config, model_name=model_name, model_server=model_server, api_key=api_key) 
        else:
            return 'Error : Invalid agent type!'

        return agent

    @staticmethod
    def _inference_editing_turn(agent, user_prompt, agent_type='OpenAIAgent'):
        agent_answer = ''
        # Run Inference
        if agent_type == 'OpenAIAgent':
            agent_answer = asyncio.run( inference_openai(agent, user_prompt) )
        elif agent_type == 'AWorld':
            agent_answer = inference_aworld(agent, user_prompt)

        return agent_answer

    #=========================================================================================================
    @staticmethod
    def get_deck_name_prefix0():
        return f'slide_'

    #=========================================================================================================
    @staticmethod
    def is_simulation_successful(deck_name, simulation_folder, minimum_max_turns=0,verbose=True):
                        
        is_success = True

        min_file_size=5000
        check_turn_and_html_numbers = True # check if the turn numbers in json and actual html files are the same
        check_html_file = True # check the validity fo html files
        validate_with_previous=False # check if new turn result is different from previous one

        # Check if simulation folder exist
        if not os.path.exists(simulation_folder):
            is_success = False
            if verbose:
                print(f'Fail : simulation folder not exist : {simulation_folder} : {deck_name}')
            return is_success 

        # Read simulation json
        json_file_name = 'simulation.json'
        simulation_info_json_path = os.path.join(simulation_folder, json_file_name)
        simulation_info_dict = {}
        if os.path.exists(simulation_info_json_path):
            with open(simulation_info_json_path, 'r') as file:
                simulation_info_dict = json.load(file)
        # if simulation json is not available
        if len(simulation_info_dict) <=0:
            if verbose:
                print(f'Error : simulation json not found : {json_file_name} : {deck_name}')
            is_success = False
            return is_success 

        # Collect all turn html files
        html_files_org = glob.glob( os.path.join(simulation_folder, '*.html') )
        html_files_org.sort()


        html_files = []
        # filter out noisy htmls which is not slide
        for html_file_path in html_files_org:         
            html_file_name = os.path.basename(html_file_path)
            deck_prefix = EditorAgent.get_deck_name_prefix(deck_name)
            if deck_prefix in html_file_name:
                html_files.append(html_file_path)


        # Check number of html files > 1 : at least one turn
        if len(html_files) <= 1:
            is_success = False
            if verbose:
                print(f'Fail : no simulated html file : {deck_name} ' )

        valid_turn_count = 0
        turn_names = list(simulation_info_dict.keys())
        turn_names.sort()

        edited_file_names = []
        # for turn_name in simulation_info_dict.keys():
        for tidx, turn_name in enumerate(turn_names):
            turn_info_dict = simulation_info_dict[turn_name]
            if "edited_slide" in turn_info_dict.keys() or "edited_deck" in turn_info_dict.keys():
                if "edited_slide" in turn_info_dict.keys():
                    edited_deck_file_name = turn_info_dict["edited_slide"]
                elif "edited_deck" in turn_info_dict.keys():
                    edited_deck_file_name = turn_info_dict["edited_deck"]

                edited_file_names.append(edited_deck_file_name)

                edited_deck_path = os.path.join(simulation_folder, edited_deck_file_name)
                if os.path.exists(edited_deck_path):

                    # Check file size
                    file_size_bytes = os.path.getsize(edited_deck_path)
                    if file_size_bytes < min_file_size: # if any edited slide is small, then fail
                        is_success = False
                        if verbose:
                            print(f'Fail : file size : {file_size_bytes} is less than {min_file_size} at turn {turn_name} : {deck_name}')

                    # check difference from previous turn file
                    if validate_with_previous and tidx > 0:
                        prev_turn_name = turn_names[tidx-1]
                        prev_turn_info_dict = simulation_info_dict[prev_turn_name]

                        if "edited_slide" in prev_turn_info_dict.keys() or "edited_deck" in prev_turn_info_dict.keys():
                            if "edited_slide" in prev_turn_info_dict.keys():
                                prev_edited_deck_file_name = prev_turn_info_dict["edited_slide"]
                            elif  "edited_deck" in prev_turn_info_dict.keys():
                                prev_edited_deck_file_name = prev_turn_info_dict["edited_deck"]


                            prev_edited_deck_path = os.path.join(simulation_folder, prev_edited_deck_file_name)
                            if os.path.exists(prev_edited_deck_path):
                                
                                # Compare with previous deck file
                                # Perform a deep comparison (checks content)
                                # shallow=False ensures content is checked even if metadata is the same
                                are_files_same = filecmp.cmp(prev_edited_deck_path, edited_deck_path, shallow=False)
                                if are_files_same:
                                    if verbose:
                                        print(f'Fail : No file change at turn {turn_name} : {deck_name}')
                                    is_success = False


                else: # if edited file not exist, then fail
                    if check_turn_and_html_numbers:
                        is_success = False
                        if verbose:
                            print(f'Fail : edited file not exist :  {edited_deck_path} : {deck_name}')

            valid_turn_count += 1

        # Check if valid turn count is same as minimum_max_turns : only when minimum_max_turns is given
        if valid_turn_count < minimum_max_turns:
            is_success = False
            if verbose:
                print(f'Fail : simulated turn number {valid_turn_count} is smaller than max_turns {max_tuminimum_max_turnsrns} : {deck_name} ' )

        # Check if edited turn file number and html file number match
        if check_turn_and_html_numbers:
            if len(edited_file_names) != len(html_files)-1: # except the original generate html file
                is_success = False
                if verbose:
                    print(f'Fail : Number of html file mismatch with edited turn files : {deck_name} ' )

        if check_html_file:
            for html_file_path in html_files: # iterate turns

                # skip the initial deck file since it was previously generated and not applicable for simulation validity
                html_file_name = os.path.basename(html_file_path)
                tail_name = html_file_name.split('_')[-1]
                if 'turn' not in tail_name or 'turn0' not in tail_name:
                    continue

                file_size_bytes = os.path.getsize(html_file_path)
                if file_size_bytes < min_file_size:
                    if verbose:
                        print(f'Error : slide html is too small: {html_file_path}')
                    is_success = False
                    # break

                if not EditorAgent.is_html_file(html_file_path):
                    if verbose:
                        print(f'Error : slide html is invalid: {html_file_path}')
                    is_success = False

        return is_success

    #=========================================================================================================
    @staticmethod
    def initialize_simulation_folder(deck_name, deck_folder, simulation_folder, reveal_path=""): 
        if not os.path.exists(simulation_folder):
            os.makedirs(simulation_folder)

        if reveal_path != "":
            # variable_name = "REVEAL_PATH"
            # reveal_folder = os.getenv(variable_name, "") 
            reveal_folder = reveal_path

            # Copy Reveal system subfolder to experiment folder
            reveal_sub_folders = ['dist', 'plugin', 'template']
            for reveal_sub_folder in reveal_sub_folders:
                src_reveal_folder = os.path.join(reveal_folder, reveal_sub_folder)
                dest_reveal_folder = os.path.join(simulation_folder, reveal_sub_folder)
                EditorAgent.copy_folder(src_reveal_folder, dest_reveal_folder)

        # check original html
        original_deck_file_name = EditorAgent.get_deck_name_prefix(deck_name)+'.html'

        original_html_file_path = os.path.join(deck_folder, original_deck_file_name)
        if not os.path.exists(original_html_file_path):
            print('Error : previous HTML file not exist!', original_html_file_path)
        else:
            # Copy Original html file to simulation folder
            shutil.copy(original_html_file_path, simulation_folder)

        # Copy original extracted markdown file to simulation folder
        extracted_file_name = 'extracted.md'
        extracted_data_file_path = os.path.join(deck_folder, extracted_file_name)
        if not os.path.exists(extracted_data_file_path):
            print('Error : extracted md file not exist! ', extracted_data_file_path)
        else:
            shutil.copy(extracted_data_file_path, simulation_folder)

        # Copy images folder from original folder to target output folder
        image_folder_sub = f'images'
        original_images_folder = os.path.join(deck_folder, image_folder_sub)
        if os.path.exists(original_images_folder):
            target_images_folder = os.path.join(simulation_folder, image_folder_sub) 
            EditorAgent.copy_folder(original_images_folder, target_images_folder)


    #=========================================================================================================
    def initialize_editor_agent_and_system_prompt(self, paper_extract_md_content, 
        model_name=None, model_server=None, api_key='dummy', 
        agent_type='OpenAIAgent'):
        
        # Initialize: editor system prompt
        self.edit_html_system_prompt = """
        You are an AI assistant that edits the previous slide HTML content based on the editing requirements with multi-turn editing requests.
        Obey the following rules strictly. 
        1. Based on the editing requirements, edit the slides and combine them into the original HTML code. Provide the full HTML code.
        2. Use the most recent turn slide HTML content for editing instead of editing older turn slides. 
        3. Use the provided paper content when additional information or resource required to edit the slide HTML.
        4. Include math expressions in MathML format where appropriate.
        5. When not requested, don't remove elements such as figures, tables or math expressions.
        6. When replacing or adding new figure images, keep the original image link from the paper content and don't hallucinate new image.
        7. The output must be in HTMLformat.
        """
        edit_html_system_prompt = self.edit_html_system_prompt
        if edit_html_system_prompt =='':
            print('Error : Error with edit_html_system_prompt.')
            return #None
        paper_prompt = f"\n[paper_content] : {paper_extract_md_content}\n"
        edit_html_system_prompt += paper_prompt

        # Initialize: Editor Agent
        editor_agent = EditorAgent._initialize_editor_agent(system_prompt= edit_html_system_prompt, tool_config= {},
                        model_name=model_name, model_server=model_server, api_key=api_key, agent_type=agent_type )
        if editor_agent is None:
            print('Error : Failed to create a Editor Agent.')
            return #None
        self.editor_agents = [editor_agent]


        # Initial: deck file , will be updated each turn
        # self.previous_deck_filename = f'slide_{self.deck_name}.html'
        self.previous_deck_filename = EditorAgent.get_deck_name_prefix(self.deck_name)+'.html'

        self.previous_deck_file_path = os.path.join(self.simulation_folder, self.previous_deck_filename)
        # generate deck MD file if not exists
        previous_deck_md_file_path = EditorAgent.get_md_file_path(self.previous_deck_file_path)
        if not os.path.exists(previous_deck_md_file_path):
            previous_deck_md_file_path = EditorAgent.html_to_md_file(self.previous_deck_file_path)
            if previous_deck_md_file_path == '' or not os.path.exists(previous_deck_md_file_path):
                print(f"Error: failed to generate initial deck markdown file.")
        self.previous_deck_md_file_path = previous_deck_md_file_path




    #=========================================================================================================
    def compose_user_prompt_turn(self, user_editing_prompt):

        previous_deck_file_path = self.previous_deck_file_path

        #### Prepare user prompt
        previous_html_content = ''
        with open(previous_deck_file_path, 'r') as file:
            previous_html_content = file.read()
        if previous_html_content == '':
            print(f"Error with previous html file at Turn{turn_idx+1}")
            return

        user_prompt = f"Edit 'Previous slide HTML' following the 'Editing requirement' based on the 'paper_content', and return the edited content of the html as string.\n"
        user_prompt += f"[Previous slide HTML] : {previous_html_content}\n"    
        user_prompt += f"[Editing requirement] : {user_editing_prompt}\n"

        self.user_prompt = user_prompt
        self.user_editing_prompt = user_editing_prompt


    #=========================================================================================================
    def inference_editing_turn(self, turn_idx, agent_type='OpenAIAgent',verbose=False):

        deck_name = self.deck_name
        editor_agents = self.editor_agents
        simulation_folder = self.simulation_folder

        # get this turn 
        user_editing_prompt =self.user_editing_prompt # simulated editing request prompt
        user_prompt = self.user_prompt # composed user prompt for the agent

        if len(editor_agents) > 0:
            editor_agent = editor_agents[0]
        else:
            print(f"Error: no editor agent provided at Turn{turn_idx+1}")
            return '','', False

        if editor_agent is None:
            print(f"Error: editor agent is None at Turn{turn_idx+1}")
            return '','', False

        if user_prompt == '':
            print(f"Error: Editor Agent user prompt is not ready at Turn{turn_idx+1}")
            return '','', False            


        edited_deck_md_file_path = ''
        edited_deck_filename = EditorAgent.get_deck_name_prefix(deck_name) + f'_turn{turn_idx+1}.html'
        edited_deck_file_path = os.path.join(simulation_folder, edited_deck_filename)

        #### Inference
        agent_answer = EditorAgent._inference_editing_turn(editor_agent, user_prompt, agent_type=agent_type)

        #### Post process
        # check if User Agent generate result properly
        if agent_answer.startswith('Error') or agent_answer == '' or len(agent_answer) < 10:
            print(f"Error: invalid agent answer at Turn{turn_idx+1}")
            return '','', False
        else:
            new_html_result = agent_answer
            EditorAgent.save_content_as_file(new_html_result, edited_deck_file_path)

        # Validate edited result
        is_success, _ = EditorAgent.is_generation_successful(edited_deck_file_path, 
                                    verbose=verbose)

         # Convert previous html to previous deck md for next turn
        if is_success:           
            edited_deck_md_file_path = EditorAgent.html_to_md_file(edited_deck_file_path)
            if edited_deck_md_file_path == '' or not os.path.exists(edited_deck_md_file_path):
                print(f"Error: simulation failed! edited deck html file failed to be converted to markdown file. at Turn{turn_idx+1}. Stop here.")
                is_success = False
            
        if not is_success:    
            print(f"Failed with Editor Agent at Turn{turn_idx+1} at trial {tidx+1}")
            return '','', False
        else: # Save simulation result json
            EditorAgent.save_simulation_result_info(turn_idx, simulation_folder, 
                    deck_name, user_editing_prompt, 
                    self.selected_persona_name, #selected_persona,  
                    self.previous_deck_filename,  edited_deck_filename)


        self.previous_deck_filename = edited_deck_filename
        self.previous_deck_file_path = edited_deck_file_path
        self.previous_deck_md_file_path = edited_deck_md_file_path

        return edited_deck_file_path, edited_deck_md_file_path, is_success