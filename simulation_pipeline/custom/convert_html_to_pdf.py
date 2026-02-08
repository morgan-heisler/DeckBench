import os
import re
import argparse
import asyncio
import json
import shutil
import sys

from pathlib import Path
import glob
import subprocess

import enlighten

from editor_agent import EditorAgent


def copy_folder(source_folder_path, target_folder_path, overwrite=True, verbose=False):

    is_copied = False
    if not os.path.exists(target_folder_path) or overwrite:
        if os.path.exists(source_folder_path):
            try:
                shutil.copytree(source_folder_path, target_folder_path, dirs_exist_ok=overwrite)
                is_copied = True
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

    return is_copied

def overwrite_folder(source_reveal_subfolder_path, target_reveal_subfolder_path, verbose=False):
    
    is_copied_reveal = False

    if os.path.exists(source_reveal_subfolder_path):
        # if target reveal subfolder exists, then remove first
        if os.path.exists(target_reveal_subfolder_path):
            try:
                shutil.rmtree(target_reveal_subfolder_path)
                print('    : Reveal folder removed: ', target_reveal_subfolder_path)
            except OSError as e:
                pass
        # Copy Reveal subfolder
        is_copied_reveal = copy_folder(source_reveal_subfolder_path, target_reveal_subfolder_path, verbose=verbose)
    else:
        print('Error: Reveal subfolder to copy not exist: ', source_reveal_subfolder_path)

    return is_copied_reveal


async def convert_html_to_pdf(input_html_path, output_pdf_file, slide_pages = ""):
    """Convert html file to the ouput pdf file.
    
    Args:
        input_html_path: The path to the html file to convert.
        output_pdf_file : output PDF file path
        slide_pages : slide number to convert
    Returns:
        the folder path to the generated pdf file
    """

    def count_sections_in_html(filename):
        from bs4 import BeautifulSoup
        
        # Ensure the file exists before trying to open it
        if not os.path.exists(filename):
            print(f"Error: The file '{filename}' was not found.")
            return 0

        try:
            # Open and read the HTML file
            with open(filename, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # Parse the HTML content using Beautiful Soup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all occurrences of the <section> tag
            sections = soup.find_all('section')
            
            # The number of tags is the length of the list returned by find_all
            return len(sections)

        except Exception as e:
            print(f"An error occurred: {e}")
            return 0

    http_proxy = os.environ.get('http_proxy')
    if not os.path.exists(input_html_path):
        return 'Error : Input HTML file not exist.'

    # Decide slide range
    num_sections = count_sections_in_html(input_html_path)
    if num_sections > 0:
        slide_range1 = "--slides"
        slide_range2 = f"1-{num_sections}"
        print("    : Calculated slide range : ", slide_range2)
    else:
        slide_range1 = ""
        slide_range2 = ""

    command = [
        "decktape",

        "--chrome-path=/usr/bin/google-chrome",
        "--chrome-arg=--no-sandbox",
        "--chrome-arg=--disable-dev-shm-usage",
        "--chrome-arg=--no-zygote",
        "--chrome-arg=--disable-gpu",
        "--chrome-arg=--ignore-certificate-errors",
        "--chrome-arg=--allow-running-insecure-content",
        "--chrome-arg=--allow-file-access-from-files",
        "--chrome-arg=--disable-web-security",

        f"--chrome-arg=--proxy-server={http_proxy}",

        "reveal",

        slide_range1,
        slide_range2,

        input_html_path ,
        output_pdf_file    
    ]

    if slide_pages != "":
        command.insert(1, "--slides=" + slide_pages)

    try:
        result = subprocess.run(command, capture_output=True, text=True,  check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command to convert html to pdf failed with error: {e}")
        print(f"-- Trying with Chrome instead...")

        try:
            await html_slides_to_pdf(input_html_path, output_pdf_file)
        except:
            print('Error: Failed to conver with Chrome.')
            return ''

    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after {e.timeout} seconds.")

        try:
            await html_slides_to_pdf(input_html_path, output_pdf_file)
        except:
            print('Error: Failed to conver with Chrome.')
            return ''
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found.")
        return ''

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ''

    if os.path.exists(output_pdf_file):
        return output_pdf_file 
    else:
        return ''

async def add_katex_script(filename, out_file_path, katex_path): 
    from bs4 import BeautifulSoup

    if katex_path == "":
        return
    
    # Ensure the file exists before trying to open it
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return

    try:
        # Open and read the HTML file
        with open(filename, 'r', encoding='utf-8') as file:
            html_content = file.read()            
        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(html_content, 'html.parser')   

        # Find the <head> tag
        head_tag = soup.head

        # Create the new link tag
        new_link = soup.new_tag('link')
        new_link['rel'] = 'stylesheet'
        new_link['href'] = os.path.join(katex_path, "katex.min.css") 
        new_link['type'] = 'text/css'


        # Create the new script tag
        external_script_tag = soup.new_tag("script") 
        external_script_tag["src"] = os.path.join(katex_path, "katex.min.js") 
        external_script_tag["defer"] = True 

        external_script_tag2 = soup.new_tag("script") 
        external_script_tag2["src"] = os.path.join(katex_path, "contrib/auto-render.min.js") 
        external_script_tag2["defer"] = True 
        external_script_tag2["onload"] = "renderMathInElement(document.body);"

        # For inline JavaScript code
        inline_script_tag = soup.new_tag("script") 
        inline_script_tag.string = """document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                // Configure auto-render options here, e.g., delimiters
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\(", right: "\\)", display: false},
                    {left: "\\[", right: "\\]", display: true}
                ],
                throwOnError : false
            });
        });
        """

        # Use append() to add it to the end of the head tag
        head_tag.append(new_link)
        head_tag.append(external_script_tag)
        head_tag.append(external_script_tag2)
        head_tag.append(inline_script_tag)

        # Save the modified HTML content to a new file
        try:
            with open(out_file_path, "w", encoding='utf-8') as file:
                file.write(str(soup.prettify())) # Use prettify() for formatted output
        except IOError as e:
            print(f"Error saving file: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

def convert_deck(html_file_path, gen_pdf_path, reveal_path="", katex_path=""):

    folder_path = os.path.dirname(html_file_path)

    # copy reveal package folders
    remove_reveal_folders = []
    if reveal_path != "":
        reveal_subfolders = ['dist', 'plugin', 'template'] #katex
        for reveal_subfolder in reveal_subfolders:
            source_reveal_subfolder_path  = os.path.join(reveal_path, reveal_subfolder)
            # Copy Reveal subfolders for generated deck
            target_reveal_subfolder_path = os.path.join(folder_path, reveal_subfolder)
            if not os.path.exists(target_reveal_subfolder_path): # if the target folder didn't exist previously, then add to remove list after conversion
                remove_reveal_folders.append(target_reveal_subfolder_path)
            is_copied_reveal = overwrite_folder(source_reveal_subfolder_path, target_reveal_subfolder_path)

    # if Ketex local is provided, then modify the HTML code
    modified_html_file_name = 'modified.html'
    if katex_path != "":
        # modified file path        
        modified_html_path = os.path.join(folder_path, modified_html_file_name)
        # Add Katex script to html head
        asyncio.run( add_katex_script(html_file_path, modified_html_path, katex_path ) )
    else:
        modified_html_path = html_file_path

    # Convert modified HTML to PDF
    generated_pdf_path = asyncio.run( convert_html_to_pdf(modified_html_path, gen_pdf_path) )

    # Remove the temp modified html path
    if katex_path != '':
        file_name = os.path.basename(modified_html_path)
        if file_name == modified_html_file_name:
            os.remove(modified_html_path)

    # Remove temporary reveal folders that didn't exist before
    for target_reveal_subfolder_path in remove_reveal_folders:
        if os.path.exists(target_reveal_subfolder_path):
            try:
                shutil.rmtree(target_reveal_subfolder_path)
            except OSError as e:
                pass

    return generated_pdf_path

def main():

    parser = argparse.ArgumentParser(description="Multi-deck Slide Evaluator")
    parser.add_argument("--deck_list_path", required=True, help="Root folder for generated slides")
    parser.add_argument("--output_path", required=True, help="Root folder for converted pdfs")

    parser.add_argument("--reveal_path", default="", help="Reveal package path")
    parser.add_argument("--katex_path", default="", help="Katex folder path")

    parser.add_argument('--start_idx', type=int, default=0, help='start paper index, from 0')
    parser.add_argument('--end_idx', type=int, default=-1, help='end paper index, from 0')

    parser.add_argument('--multiturn', action='store_true', help='For multiturn evaluation mode')
    parser.add_argument("--simulation_name", default="simulation", help="Simulation name")

    # optional extra arguments
    parser.add_argument("--start_deck_name", default="", help="deck_name to start")
    parser.add_argument("--end_deck_name", default="", help="deck_name to end")
    parser.add_argument('--only_failed', action='store_true', help='convert only failed conversion') #default is False

    args = parser.parse_args()


    deck_list_path = args.deck_list_path
    output_folder = args.output_path

    reveal_path = args.reveal_path
    katex_path = args.katex_path

    start_idx = args.start_idx
    end_idx = args.end_idx

    sim_folder = args.simulation_name 

    only_failed = args.only_failed

    failure_json_path = os.path.join(output_folder, 'conversion_failure.json')

    if not os.path.exists(deck_list_path):
        print('Error : deck_list_path not exist!')
        exit()

    if not os.path.exists(katex_path):
        print('Error : katex_path not exist!')
        exit()


    is_multiturn = args.multiturn
    print('Multiturn : ', is_multiturn)

    if not is_multiturn: # ignore sim_folder if not multiturn
        sim_folder = ''
    else: # multiturn
        if sim_folder == '':
            print('Error : Simulation folder not provided!')
            exit()
        print('Simulation name : ', sim_folder)

    deck_names = []
    # Generate failed conversion
    if args.only_failed: 
        if os.path.exists(failure_json_path): 
            failure_dict = {}
            with open(failure_json_path, 'r') as file:
                failure_dict = json.load(file)

            if len(failure_dict) > 0:                
                for fid in failure_dict.keys():
                    paper_stem = failure_dict[fid]
                    deck_names.append(paper_stem)
    else:
        p = Path(deck_list_path)
        deck_names = [entry.name for entry in p.iterdir() if entry.is_dir()]    
    deck_names.sort()

    


    # if start_deck_name is provided, then update start_idx
    if args.start_deck_name != '':        
        for sidx, deck_name in enumerate(deck_names): # iterate slides
            if deck_name == args.start_deck_name:
                start_idx = sidx
                print('= Start index is decided as ', start_idx, ' with slide name : ', args.start_deck_name)
                break
    # if end_deck_name is provided, then update end_idx
    if args.end_deck_name != '':        
        for sidx, deck_name in enumerate(deck_names): # iterate slides
            if deck_name == args.end_deck_name:
                end_idx = sidx
                print('= End index is decided as ', end_idx, ' with slide name : ', args.end_deck_name)
                break

    num_of_papers = len(deck_names)
    if start_idx >=0: 
        if end_idx >= start_idx: 
            num_of_papers = end_idx - start_idx + 1
        else:
            num_of_papers = num_of_papers - start_idx
    else:
        if end_idx >= 0:
            num_of_papers = end_idx 


    # Enlighten bar
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('Convert HTML to PDF',
                                    color='white_on_red',
                                    justify=enlighten.Justify.CENTER)
    pbar = manager.counter(total=num_of_papers, desc='Conversion', unit='ticks', color='green')


    failed_paper_dict = {}
    failure_message = {}
    for sidx, deck_name in enumerate(deck_names): # iterate slides

        if start_idx >= 0 and  sidx < start_idx:
            continue

        bConversionFailed = False

        status_bar.update('Convert slide : ' + deck_name, force=True) 

        print('Convert slide : ', deck_name)
        html_deck_path = os.path.join(deck_list_path, deck_name)


        if not os.path.exists(html_deck_path):
            bConversionFailed = True
            error_message = f'Error : slide folder not exist : {html_deck_path}'
            failed_paper_dict.update({sidx: deck_name}) 
            failure_message.update({deck_name: error_message})
            print(error_message)
            pbar.update()
            continue

        # create output PDF folder
        # pdf_folder_path should be different for generation and multi-turn. As for multi-turn, slide subfolders
        if is_multiturn:
            pdf_folder_path = os.path.join(output_folder, deck_name)
        else:
            pdf_folder_path = output_folder
        os.makedirs(pdf_folder_path, exist_ok=True)

        # Copy simulation.json
        ######### need to save prompt only, read and generate new json
        if is_multiturn:
            multiturn_info_dict = {}
            simulation_json_file = os.path.join(html_deck_path , sim_folder, 'simulation.json')
            if os.path.exists(simulation_json_file):
                # shutil.copy2(simulation_json_file, pdf_folder_path)
                simulation_info_dict = {}
                if os.path.exists(simulation_json_file):
                    with open(simulation_json_file, 'r') as file:
                        simulation_info_dict = json.load(file)
                for turn_name in simulation_info_dict.keys():
                    multiturn_info_dict[turn_name] ={}
                    turn_info_dict = simulation_info_dict[turn_name]
                    if "user_prompt" in turn_info_dict.keys():
                        instruction = turn_info_dict["user_prompt"]

                        prompt_dict = {'user_prompt':instruction}
                        multiturn_info_dict[turn_name].update(prompt_dict)
                    elif "user_editing_prompt" in turn_info_dict.keys():
                        instruction = turn_info_dict["user_editing_prompt"]

                        prompt_dict = {'user_editing_prompt':instruction}
                        multiturn_info_dict[turn_name].update(prompt_dict)

                    if "user_persona_name" in turn_info_dict.keys():
                        persona_name = turn_info_dict["user_persona_name"]

                        persona_name_dict = {'user_persona_name':persona_name}
                        multiturn_info_dict[turn_name].update(persona_name_dict)

                # save new json
                # json_file_name = 'slide_'+deck_name+'.json'
                json_file_name = EditorAgent.get_deck_name_prefix(deck_name)+'.json'

                out_path = os.path.join(pdf_folder_path, json_file_name) 
                with open(out_path, "w") as f:
                    json.dump(multiturn_info_dict, f, indent=4)

            else:
                bConversionFailed = True
                error_message = f'Error : can not find simulation.json with slide : {deck_name} at index : {sidx}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                pbar.update()
                continue


        # if simulation folder not exist then error
        if is_multiturn and sim_folder != '':
            sim_folder_path = os.path.join(html_deck_path , sim_folder)
            if not os.path.exists(sim_folder_path):
                bConversionFailed = True
                error_message = f'Error : simulation folder not exist : , {sim_folder}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                pbar.update()
                continue

        html_files = glob.glob( os.path.join(html_deck_path , sim_folder, '*.html') )
        html_files.sort()
        for html_file_path in html_files: # iterate turns

            print('  - Convert html : ', html_file_path)

            if not os.path.exists(html_file_path):
                bConversionFailed = True
                error_message = f'Error : can not find the slide html : {html_file_path} at index : {sidx}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                continue

            file_size_bytes = os.path.getsize(html_file_path)
            if file_size_bytes <= 10:
                bConversionFailed = True
                error_message = f'Error : slide html is empty: {html_file_path} at index : {sidx}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                continue

            if not EditorAgent.is_html_file(html_file_path):
                bConversionFailed = True
                error_message = f'Error : slide html is invalid: {html_file_path} at index : {sidx}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                continue




            html_file = os.path.basename(html_file_path)
            file_path = Path(html_file)
            gen_file_name = file_path.stem

            if is_multiturn:
                # if turn_name is not included(it means original slide), add turn name 'turn0'
                tail_name = gen_file_name.split('_')[-1]
                if 'turn' not in tail_name:
                    gen_file_name = gen_file_name + '_turn0'
            gen_pdf_path = os.path.join(pdf_folder_path, gen_file_name+'.pdf')

            # Convert HTML to PDF
            generated_pdf_path = convert_deck(html_file_path, gen_pdf_path, reveal_path, katex_path)

            if generated_pdf_path != '':
                print('  - Generated pdf file : ', generated_pdf_path)
            else:
                bConversionFailed = True
                error_message = f'  - Failed to generate pdf file for slide : , {deck_name}'
                failed_paper_dict.update({sidx: deck_name}) 
                failure_message.update({deck_name: error_message})
                print(error_message)
                continue

        if bConversionFailed:
            error_message = f'Error : failed conversion with slide : , {deck_name} , at index : , {sidx}'
            print(error_message)

        print('')

        pbar.update()

        if end_idx >= 0 and end_idx >= start_idx and sidx >= end_idx:
            break

    manager.stop()


    if len(failed_paper_dict) > 0:
        print('')
        print('------------- Conversion Failure Summary -------------')
        for sidx in failed_paper_dict.keys():
            deck_name = failed_paper_dict[sidx]
            print(f'- Failed :{sidx} : {deck_name}')
            if deck_name in failure_message.keys():
                error_message = failure_message[deck_name]
                print(f'   : {error_message}')
        print('')
 
    # Save conversion failure json
    if os.path.exists(output_folder):            
        if len(failed_paper_dict) > 0:
            with open(failure_json_path, "w") as f:
                json.dump(failed_paper_dict, f, sort_keys=True, indent=4)
        else: # Remove old 
            print('!---- No failed conversion')
            if os.path.exists(failure_json_path): # if previous failure json, then remove it
                # if old one exist, then backup
                old_failure_json_path = os.path.join(output_folder, 'conversion_failure_old.json')
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

if __name__ == "__main__":

    main()
