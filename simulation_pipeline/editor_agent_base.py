import os
import json

from abc import ABC, abstractmethod

class EditorAgentBase(ABC):

    def __init__(self, deck_name, simulation_folder, selected_persona_name):

        self.editor_agents = []

        self.deck_name = deck_name
        self.simulation_folder = simulation_folder
        self.selected_persona_name = selected_persona_name

        # will be updated each turn
        self.user_prompt = ''
        self.user_editing_prompt = ''

    #=========================================================================================================
    @abstractmethod
    def initialize_editor_agent_and_system_prompt(self, extract_md_content, 
        model_name=None, model_server=None, api_key='dummy', 
        agent_type='OpenAIAgent'):            
        pass

    #=========================================================================================================
    @abstractmethod
    def compose_user_prompt_turn(self, user_editing_prompt):
        pass

    #=========================================================================================================
    @abstractmethod
    def inference_editing_turn(self, turn_idx, agent_type='OpenAIAgent'):
        pass


    @staticmethod
    def initialize_simulation_folder(deck_name, deck_folder, simulation_folder):
        if not os.path.exists(simulation_folder):
            os.makedirs(simulation_folder)

    @staticmethod
    def is_simulation_successful(deck_name, simulation_folder, minimum_max_turns=0, verbose=True):
        return True


    @staticmethod
    def get_deck_name_prefix0():
        return f'slide_'
    @classmethod
    def get_deck_name_prefix(cls, deck_name):
        return  cls.get_deck_name_prefix0() + f'{deck_name}'


    @staticmethod    
    def save_simulation_result_info(turn_idx, simulation_folder, deck_name, user_editing_prompt, 
            selected_persona_name, 
            prev_deck_filename,  edited_deck_filename):

        # Read previous turn simulation info json
        json_file_name = 'simulation.json'
        simulation_info_json_path = os.path.join(simulation_folder, json_file_name)
        simulation_info_dict = {}
        if os.path.exists(simulation_info_json_path):
            with open(simulation_info_json_path, 'r') as file:
                simulation_info_dict = json.load(file)

        json_file_name = 'slide_'+deck_name+'.json'
        simulation_simple_info_json_path = os.path.join(simulation_folder, json_file_name)
        simulation_simple_info_dict = {}
        if os.path.exists(simulation_simple_info_json_path):
            with open(simulation_simple_info_json_path, 'r') as file:
                simulation_simple_info_dict = json.load(file)

        # append simulation info json
        turn_name = f"Turn{turn_idx+1}"
        simulation_info_dict[turn_name] =  {
                "deck_name": deck_name,
                "user_persona_name": selected_persona_name,
                "user_editing_prompt": user_editing_prompt,
                "previous_deck": prev_deck_filename, 
                "edited_deck":  edited_deck_filename
            }
        # Save simulation info dict
        if len(simulation_info_dict) > 0:
            with open(simulation_info_json_path, "w") as f:
                json.dump(simulation_info_dict, f, sort_keys=True, indent=4)
            print('Saved simulation json file : ', simulation_info_json_path)


        # simple simulation info dict
        simulation_simple_info_dict[turn_name] =  {
                "user_persona_name": selected_persona_name,
                "user_editing_prompt": user_editing_prompt,
            }
        # # Save simple simulation info dict
        # if len(simulation_simple_info_dict) > 0:
        #     with open(simulation_simple_info_json_path, "w") as f:
        #         json.dump(simulation_simple_info_dict, f, sort_keys=True, indent=4)
        #     print('Saved simple simulation json file : ', simulation_simple_info_json_path)

        return simulation_info_dict, simulation_simple_info_dict


