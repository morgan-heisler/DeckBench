import json
import os
import re
import numpy as np
from multiprocessing import Process, Queue
from collections import defaultdict

import sys
from pathlib import Path

# Add the benchmark root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # go up two levels
from utils import EmbeddingEngine

# Logging
from loguru import logger as loguru_logger


from .metric_utils import calculate_paper_embeddings, calculate_slide_embeddings, calculate_initial_embeddings, combine_slide_embeddings_concat

from .metric_utils import convert_slides_to_pdf

from .slide_metrics import SlideMetrics
from .deck_metrics import DeckMetrics


# Prepare API key dict
variable_name = "OPENAI_API_KEY"
ChatGPT_key = os.getenv(variable_name, "dummy")
variable_name = "MAAS_API_KEY"
MaaS_key = os.getenv(variable_name, "dummy")
api_keys_dict = {
    "None": 'dummy',
    "GPT": ChatGPT_key,
    "MaaS": MaaS_key
}

class SlideEvaluator(SlideMetrics, DeckMetrics):

    def __init__(self, 
                text_model_path="/root/data/models/all-MiniLM-L6-v2", clip_model_path="/root/data/models/clip-vit-base-patch32", perplexity_model_path = '/root/data/models/gpt2', 
                llm_judge_model_name='', llm_judge_model_server='', llm_judge_api_type='None', agent_type = 'OpenAIAgent'):

        # super().__init__()

        SlideMetrics.__init__(self) 
        DeckMetrics.__init__(self) 

        loguru_logger.remove() # Disable logging of AWorld agents (loguru)

        self.engine = None
        self.ppl_model = self.ppl_tokenizer = None
        self.coherence_agent = self.content_agent = self.tsbench_agent = None

        # Necessary for TSBench
        self.instruction = ''
        self.original_slides = None

        self.agent_type = agent_type # 'OpenAIAgent' #'AWorld'

        api_key = 'dummy'
        if llm_judge_api_type in api_keys_dict.keys():     
            api_key = api_keys_dict[llm_judge_api_type]
        else:
            print('Warning : API key type is invalid. Can not find api_key!')
        if api_key == 'dummy':
            print('Warining : API key is dummy. LLM Judge may not work with GPT or MaaS!')

        # Initialize embedder
        self.engine = EmbeddingEngine(text_model_name=text_model_path, clip_model_name=clip_model_path)
        # Initialize gpt2 model for perplexity
        self.load_gpt2_model(perplexity_model_path)

        # Initialize GTP5 agents for llm judge
        self.initialize_llm_judge(llm_judge_model_name, llm_judge_model_server, api_key=api_key, agent_type=self.agent_type)

        self.paper_embs = None
        self.slides_gt = self.txt_emb_gt = self.vis_emb_gt = None
        self.slides_gen = self.txt_emb_gen = self.vis_emb_gen = None

        self.gen_combined = self.gt_combined = self.valid_slide_indices_gen = self.valid_slide_indices_gt = None

        self.paper_pdf_path = ''
        self.slide_pdf_path_gt = ''

    def initialize_llm_judge(self, llm_judge_model_name='gpt-5', llm_judge_model_server=None, api_key='dummy', agent_type='OpenAIAgent'):
        # Repo
        from .agents import initialize_llm_judge_agents

        if llm_judge_model_name == "" or llm_judge_model_name is None or llm_judge_model_server == "" or llm_judge_model_server is None:
            return

        self.coherence_agent, self.content_agent, self.tsbench_agent = initialize_llm_judge_agents(model_name=llm_judge_model_name, model_server=llm_judge_model_server, api_key= api_key, agent_type=agent_type)

    def initialize_embeddings(self, slide_pdf_path_gt, slide_pdf_path_gen, paper_path, paper_chunk_size=1000, verbose=False):
        self.slides_gen, self.slides_gt, self.txt_emb_gen, self.txt_emb_gt, self.vis_emb_gen, self.vis_emb_gt, self.paper_embs, self.gen_combined, self.gt_combined, self.valid_slide_indices_gen, self.valid_slide_indices_gt = \
            calculate_initial_embeddings(self.engine, slide_pdf_path_gt, slide_pdf_path_gen, paper_path, paper_chunk_size=paper_chunk_size, verbose=verbose)

    # -------------------------
    # -- Collect Slide-level
    @staticmethod
    def collect_slide_metrics(slide_metrics_reference_free, row_ind, col_ind, combined, text_sim, visual_sim, combined_sim, pos_sim, dtw_inv):
        slide_metrics_matches = []
        for r, c in zip(row_ind, col_ind):
            slide_metrics_matches.append({
                "gt_index": int(r),
                "gen_index": int(c),
                "dtw_alignment": ",".join(str(x) for x in dtw_inv.get(r, [])),
                "text_similarity": float(text_sim[r,c]),
                "visual_similarity": float(visual_sim[r,c]),
                "positional_similarity": float(pos_sim[r,c]),
                "combined_similarity": float(combined[r,c]),
                "combined_visual_text_similarity": float(combined_sim[r,c]),
                "perplexity": slide_metrics_reference_free[c]["perplexity"],
                "fidelity": slide_metrics_reference_free[c]["fidelity"]
            })

        # Sort matches by gt_index
        slide_metrics_matches = sorted(matches, key=lambda x: x["gt_index"])

        return slide_metrics_matches

    # -- Collect All
    @staticmethod
    def collect_all_metrics(slide_html_path_gt, slide_html_path_gen, paper_path, deck_dtw, deck_transition, deck_perplexity, deck_fidelity, alignment, slide_metrics_matches, text_sim, output_prefix):

        n, m = text_sim.shape

        # Save summary
        summary = {
            "gt_file": slide_html_path_gt,
            "gen_file": slide_html_path_gen,
            "paper_file": paper_path,
            "n_gt": n,
            "n_gen": m,
            "deck_dtw": float(deck_dtw),
            "deck_transition": float(deck_transition),
            "deck_perplexity": deck_perplexity,
            "deck_fidelity": deck_fidelity,
            # "LLM-as-a-judge Coherence": coherence_agent_answer_dict,
            # "LLM-as-a-judge Content": content_agent_answer_dict,
            # "LLM-as-a-judge TSBench": tsbench_agent_answer_dict,
            # "weights": {"text": w_text, "visual": w_vis, "positional": w_pos},
            "matches": slide_metrics_matches, 
            "deck_dtw_alignment": alignment,
        }
        json_path = output_prefix + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON results to {json_path}")

        # print top matches
        print("\nTop matches (by gt index):")
        for mm in matches:
            print(f"GT slide {mm['gt_index']} <-> GEN slide {mm['gen_index']} | combined={mm['combined_similarity']:.4f} (text={mm['text_similarity']:.4f}, vis={mm['visual_similarity']:.4f}, pos={mm['positional_similarity']:.4f})")
        return summary        


    def evaluate(self, slide_pdf_path_gt, slide_path_gen, paper_pdf_path, paper_chunk_size=1000, queue=None,  verbose=False):

        # Initialize embeddings
        if verbose:
            print('Calculate embeddings ...')
        if queue is not None:
            queue.put('Calculate embeddings ...')

        self.initialize_embeddings(slide_pdf_path_gt, slide_path_gen, paper_pdf_path, paper_chunk_size=paper_chunk_size, verbose=False)


        if verbose:
            print('Slide-level Reference-free...')
        if queue is not None:
            queue.put('Calculate slide-level reference-free metrics ...')

        # Slide-level Reference-free
        slide_metrics_reference_free = self.calculate_slide_metrics_reference_free()

        if verbose:
            print('Slide-level Reference-based...')
        if queue is not None:
            queue.put('Calculate slide-level reference-based metrics ...')

        # Slide-level Reference-based
        row_ind, col_ind, combined, text_sim, visual_sim, combined_sim, pos_sim = self.calculate_slide_metrics_reference_based()

        if verbose:
            print('Deck-level Reference-based...')
        if queue is not None:
            queue.put('Calculate deck-level reference-based metrics ...')


        # Deck-level Reference-free
        deck_perplexity, deck_faithfulness, deck_fidelity = self.calculate_deck_metrics_reference_free()


        # Deck-level Reference-based
        deck_dtw, deck_transition, dtw_inv, alignment = self.calculate_deck_metrics_reference_based()

        # Deck-level Reference-free
        coherence_agent_answer_dict, content_agent_answer_dict, tsbench_agent_answer_dict = self.calculate_deck_metrics_reference_free_llm_judge(verbose=True)

        # Extract paper ID
        match = re.search(r"slide_([\w-]+)\.(html|pdf)$", slide_path_gen)
        if not match:
            print("❌ Could not extract ID, skipping.")
        paper_id = match.group(1)

        if verbose:
            print('Saving Metrics...')

        matches = []
        for r, c in zip(row_ind, col_ind):
            matches.append({
                "gt_index": int(r),
                "gen_index": int(c),
                "dtw_alignment": ",".join(str(x) for x in dtw_inv.get(r, [])),
                "text_similarity": float(text_sim[r,c]),
                "figure_similarity": float(visual_sim[r,c]),
                "perplexity": slide_metrics_reference_free[c]["perplexity"],
                "faithfulness": slide_metrics_reference_free[c]["faithfulness"],
                "layout": slide_metrics_reference_free[c]["layout"]
            })
        # Save summary
        summary = {
            "gt_file": slide_pdf_path_gt,
            "gen_file": slide_path_gen,
            "paper_file": paper_pdf_path,
            "deck_dtw": float(deck_dtw), 
            "deck_transition": float(deck_transition), 
            "deck_perplexity": float(deck_perplexity), 
            "deck_faithfulness": float(deck_faithfulness), 
            "deck_fidelity": deck_fidelity,
            "matches": matches,
        }
        json_path = os.path.join(os.path.dirname(slide_path_gen),f"slide_{paper_id}_similarity_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON results to {json_path}")



    def evaluate_turn2(self, slide_path_gen, prev_metrics=None, queue=None,  verbose=False):
        """
        Evaluate a single generated slide deck against GT slides and paper PDF.

        Args:
            slide_path_gen: path to generated slide PDF
            prev_metrics: optional dict of previous reference-based metrics to compute delta
            paper_chunk_size: chunk size for text embeddings
            queue: optional multiprocessing queue for progress messages
            verbose: print progress messages
        Returns:
            summary dict with slide-level and deck-level metrics, including tsbench outputs
        """

        # if slide path is not valid, then return empty
        summary = {}
        if slide_path_gen is None or (slide_path_gen == '') or (not os.path.exists(slide_path_gen)) :
            return summary

        if verbose:
            print("Calculate generated embeddings ...")
        if queue is not None:
            queue.put("Calculate generated embeddings ...")

        self.slides_gen, self.txt_emb_gen, self.vis_emb_gen, self.valid_slide_indices_gen = calculate_slide_embeddings(self.engine, slide_path_gen, minimum_text_length=50)
        # For generated slides
        self.gen_combined = combine_slide_embeddings_concat(self.txt_emb_gen, self.vis_emb_gen, w_text=0.6, w_visual=0.4)

        # --- Slide-level reference-free metrics ---
        if verbose:
            print("Slide-level Reference-free ...")
        if queue is not None:
            queue.put("Calculate slide-level reference-free metrics ...")
        slide_metrics_reference_free = self.calculate_slide_metrics_reference_free()


        # --- Slide-level reference-based metrics ---
        if verbose:
            print("Slide-level Reference-based ...")
        if queue is not None:
            queue.put("Calculate slide-level reference-based metrics ...")
        row_ind, col_ind, combined, text_sim, visual_sim, combined_sim, pos_sim = \
            self.calculate_slide_metrics_reference_based()

        # --- Deck-level reference-free ---
        if verbose:
            print("Deck-level Reference-free ...")
        if queue is not None:
            queue.put("Deck-level Reference-free ...")
        deck_perplexity, deck_faithfulness, deck_fidelity = \
            self.calculate_deck_metrics_reference_free()

        # --- Deck-level reference-based ---
        if verbose:
            print("Deck-level Reference-based ...")
        if queue is not None:
            queue.put("Deck-level Reference-based ...")
        deck_dtw, deck_transition, dtw_inv, alignment = \
            self.calculate_deck_metrics_reference_based()

        # --- Deck-level reference-free LLM judge (TSBench, coherence, content) ---
        if verbose:
            print("Deck-level LLM-judge metrics ...")
        if queue is not None:
            queue.put("Deck-level LLM-judge metrics ...")
        coherence_agent_answer_dict, content_agent_answer_dict, tsbench_agent_answer_dict = \
            self.calculate_deck_metrics_reference_free_llm_judge(verbose=True)

        # --- Compute rate of change for reference-based metrics if prev_metrics provided ---
        ref_based_rate_change = {}
        if prev_metrics:
            prev_dtw = prev_metrics.get("deck_dtw", None)
            prev_transition = prev_metrics.get("deck_transition", None)
            if prev_dtw is not None and prev_transition is not None:
                ref_based_rate_change = {
                    "dtw_rate_change": float(deck_dtw - prev_dtw),
                    "transition_rate_change": float(deck_transition - prev_transition)
                }

        # --- Compile per-slide matches ---
        matches = []
        for r, c in zip(row_ind, col_ind):
            matches.append({
                "gt_index": int(r),
                "gen_index": int(c),
                "dtw_alignment": ",".join(str(x) for x in dtw_inv.get(r, [])),
                "text_similarity": float(text_sim[r, c]),
                "figure_similarity": float(visual_sim[r, c]),
                "perplexity": float(slide_metrics_reference_free[c]["perplexity"]),
                "faithfulness": float(slide_metrics_reference_free[c]["faithfulness"]),
                "layout": slide_metrics_reference_free[c]["layout"]
            })

        # --- Summary dict ---
        summary = {
            "gt_file": self.slide_pdf_path_gt,
            "gen_file": slide_path_gen,
            "paper_file": self.paper_pdf_path,
            "deck_dtw": float(deck_dtw),
            "deck_transition": float(deck_transition),
            "deck_perplexity": float(deck_perplexity),
            "deck_faithfulness": float(deck_faithfulness),
            "deck_fidelity": float(deck_fidelity),
            "slide_matches": matches,
            "coherence_llm": coherence_agent_answer_dict,
            "content_llm": content_agent_answer_dict,
            "tsbench_llm": tsbench_agent_answer_dict,
            "ref_based_rate_change": ref_based_rate_change
        }

        return summary



    def calculate_paper_embeddings(self, paper_path, paper_chunk_size=1000, minimum_text_length = 50, verbose=False):
        self.paper_pdf_path = paper_path

        self.paper_embs = calculate_paper_embeddings(self.engine, paper_path, paper_chunk_size, minimum_text_length, verbose=verbose)

    def calculate_gt_slide_embeddings(self, slide_pdf_path_gt, minimum_text_length = 50, verbose=False):
        self.slide_pdf_path_gt = slide_pdf_path_gt

        self.slides_gt, self.txt_emb_gt, self.vis_emb_gt, self.valid_slide_indices_gt = calculate_slide_embeddings(self.engine, slide_pdf_path_gt, minimum_text_length, verbose=verbose)
        # For ground truth slides
        self.gt_combined = combine_slide_embeddings_concat(self.txt_emb_gt, self.vis_emb_gt, w_text=0.6, w_visual=0.4)
