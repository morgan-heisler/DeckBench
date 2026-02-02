import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

import asyncio

from .prompts import coherence_system_prompt, content_system_prompt, tsbench_system_prompt, coherence_prompt, content_prompt, tsbench_prompt
from .agents import inference_aworld, inference_openai

from .slide_metrics import SlideMetrics


class DeckMetrics:

    def __init__(self):
        self.coherence_agent = self.content_agent = self.tsbench_agent = None

        self.paper_embs = None
        self.slides_gt = self.txt_emb_gt = self.vis_emb_gt = None
        self.slides_gen = self.txt_emb_gen = self.vis_emb_gen = None

    # -------------------------
    # Deck-level Reference-based functions
    # -------------------------
    @staticmethod
    def compute_deck_dtw_similarity(emb_gt, emb_gen, metric="cosine"):
        """
        Compute DTW similarity between two sequences of embeddings.
        Returns:
        - similarity (normalized)
        - alignment path (list of tuples)
        """
        from dtaidistance import dtw_ndim

        # constrain the warping path to ±5 slides around the diagonal
        distance = dtw_ndim.distance(emb_gt, emb_gen, window=5)
        max_len = max(len(emb_gt), len(emb_gen))
        similarity = np.exp(-distance / max_len)
        best_path = dtw_ndim.warping_path(emb_gt, emb_gen, window=5)

        return similarity, best_path

    @staticmethod
    def compute_transition_consistency(emb_gt, emb_gen):
        """
        Models transitions as edges between consecutive slides:
        T_gt = [cos(e₁, e₂), cos(e₂, e₃), …]
        T_gen = [cos(g₁, g₂), cos(g₂, g₃), …]
        Then compares their distributions or mean absolute difference.
        This captures narrative smoothness — how similar the “flow” of ideas is.
        """
        trans_gt = [np.dot(emb_gt[i], emb_gt[i+1]) /
                    (np.linalg.norm(emb_gt[i]) * np.linalg.norm(emb_gt[i+1]))
                    for i in range(len(emb_gt)-1)]
        trans_gen = [np.dot(emb_gen[i], emb_gen[i+1]) /
                    (np.linalg.norm(emb_gen[i]) * np.linalg.norm(emb_gen[i+1]))
                    for i in range(len(emb_gen)-1)]
        return 1 - abs(np.mean(trans_gt) - np.mean(trans_gen))


    # -------------------------
    # Reference-free functions : faithfulness , fidelity
    # -------------------------
    @staticmethod
    def calculate_deck_faithfulness(slide_txt_embs, paper_embs):

        # Calculate faithfulness matrix
        all_faithfulness = []
        for sidx, slide_txt_emb in enumerate(slide_txt_embs):
            slide_emb = slide_txt_emb.reshape(1, -1)
            slide_faithfulness = []
            for paper_chunk_emb in paper_embs:
                paper_emb = paper_chunk_emb.reshape(1, -1)
                similarity = float(SlideMetrics.compute_similarity(slide_emb, paper_emb))
                slide_faithfulness.append(similarity)
            all_faithfulness.append(slide_faithfulness)
        all_faithfulness_arr = np.array(all_faithfulness)

        # Hungarian Matching
        all_faithfulness_arr = 1.0 - all_faithfulness_arr
        # Find the optimal assignment
        row_ind, col_ind = hungarian(all_faithfulness_arr)

        match_pairs = list( zip(row_ind.tolist(), col_ind.tolist()) )

        match_indices = ( row_ind, col_ind )
        mathed_faithfulness = np.array(all_faithfulness)[match_indices]
        faithfulness_avg = np.mean(mathed_faithfulness)

        return faithfulness_avg,  mathed_faithfulness, match_pairs

    @staticmethod
    def calculate_deck_fidelity(paper_embs, slide_txt_embs):
        all_coverages = []
        for paper_chunk_emb in paper_embs:
            paper_emb = paper_chunk_emb.reshape(1, -1)

            slide_coverages = []
            for sidx, slide_txt_emb in enumerate(slide_txt_embs):
                slide_emb = slide_txt_emb.reshape(1, -1)

                similarity = float(SlideMetrics.compute_similarity(slide_emb, paper_emb))

                slide_coverages.append(similarity)
            all_coverages.append(slide_coverages)
        all_coverages = np.array(all_coverages)

        max_per_chunk = np.max(all_coverages, axis=1)
        max_idx_per_chunk = np.argmax(all_coverages, axis=1)

        coverage_avg = np.mean(max_per_chunk)

        return coverage_avg, max_per_chunk, max_idx_per_chunk

    # -------------------------
    # -- Deck-level Reference-free metrics --
    # -------------------------
    def calculate_deck_metrics_reference_free(self, verbose=False):

        deck_perplexity = None
        deck_faithfulness = None
        deck_fidelity = None

        # Deck-level perplexity
        if self.slides_gen is not None:
            deck_text = " ".join(
                [s.get("text", "") if isinstance(s, dict) else str(s) for s in self.slides_gen]
            )
            try:
                deck_perplexity = float(self.compute_perplexity(deck_text))
            except Exception as e:
                deck_perplexity = None
                print(f"[warn] Deck-level perplexity failed: {e}")


        # Deck-level faithfulness
        try:
            if self.paper_embs is not None and self.txt_emb_gen is not None:
                deck_faithfulness,  mathed_faithfulness, match_pairs = \
                    DeckMetrics.calculate_deck_faithfulness(self.txt_emb_gen, self.paper_embs)
        except Exception as e:
            deck_faithfulness = None
            print(f"[warn] Deck-level faithfulness failed: {e}")


        # Deck-level fidelity
        try:
            if self.paper_embs is not None and self.txt_emb_gen is not None:

                deck_fidelity, max_per_chunk, max_idx_per_chunk = \
                    DeckMetrics.calculate_deck_fidelity(self.paper_embs, self.txt_emb_gen)
        except Exception as e:
            deck_fidelity = None
            print(f"[warn] Deck-level fidelity failed: {e}")

        return deck_perplexity, deck_faithfulness, deck_fidelity


    # -------------------------
    # -- Deck-level reference-based metrics --
    # -------------------------
    def calculate_deck_metrics_reference_based(self, verbose=False):
        from collections import defaultdict


        deck_dtw = deck_transition = dtw_inv = alignment = None

        if self.txt_emb_gt is None or self.txt_emb_gen is None:
            return deck_dtw, deck_transition, dtw_inv, alignment

        # deck dtw similarity
        if verbose:
            print("Computing deck dtw similarity...")

        deck_dtw, alignment = DeckMetrics.compute_deck_dtw_similarity(self.txt_emb_gt, self.txt_emb_gen)

        # Convert DTW alignment to a lookup by generated slide index
        dtw_inv = defaultdict(list)
        for gt_idx, gen_idx in alignment:
            dtw_inv[gt_idx].append(gen_idx)




        # deck transition consistency
        if verbose:
            print("Computing deck transition consistency...")
        deck_transition = DeckMetrics.compute_transition_consistency(self.txt_emb_gt, self.txt_emb_gen)

        return deck_dtw, deck_transition, dtw_inv, alignment


    # -------------------------
    # LLM-as-a-judge (Deck-level Reference-free-llm-judge) metrics --
    # -------------------------
    def calculate_deck_metrics_reference_free_llm_judge(self, verbose=False):
        import json

        coherence_agent_answer_dict = {}
        content_agent_answer_dict = {}
        tsbench_agent_answer_dict = {}

        slides_gen = self.slides_gen

        if slides_gen is None:
            return coherence_agent_answer_dict, content_agent_answer_dict, tsbench_agent_answer_dict

        coherence_messages = [{
            "role": "system",
            "content": [{"type": "text", "text": coherence_system_prompt}]
        }]

        content_messages = [{
            "role": "system",
            "content": [{"type": "text", "text": content_system_prompt}]
        }]

        tsbench_messages = [{
            "role": "system",
            "content": [{"type": "text", "text": tsbench_system_prompt}]
        }]

        # Run inference
        # coherence_messages, 
        if self.coherence_agent is not None:
            if self.agent_type == 'OpenAIAgent':
                coherence_agent_answer = asyncio.run( inference_openai(self.coherence_agent, coherence_prompt.format(slides_gen=slides_gen)) )
            elif self.agent_type == 'AWorld':
                coherence_agent_answer = inference_aworld(self.coherence_agent, coherence_prompt.format(slides_gen=slides_gen)) # coherence_messages, coherence_group)
            
            if verbose:
                print('Coherence Agent answer : ', coherence_agent_answer)
            if coherence_agent_answer != '':
                coherence_agent_answer_dict = json.loads(coherence_agent_answer) # AGENT RETURNS A STRING, SO WE NEED TO CONVERT TO DICT

        # content_messages, 
        if self.content_agent is not None:
            if self.agent_type == 'OpenAIAgent':
                content_agent_answer = asyncio.run( inference_openai(self.content_agent, content_prompt.format(slides_gen=slides_gen)) )
            elif self.agent_type == 'AWorld':
                content_agent_answer = inference_aworld(self.content_agent, content_prompt.format(slides_gen=slides_gen)) # content_messages, content_group)
            
            if verbose:
                print('Content Agent answer : ', content_agent_answer)
            if content_agent_answer != '':
                content_agent_answer_dict = json.loads(content_agent_answer) # AGENT RETURNS A STRING, SO WE NEED TO CONVERT TO DICT

        # tsbench_messages, 
        if self.tsbench_agent is not None and self.instruction is not None and self.instruction != '' and self.original_slides is not None:

            if self.agent_type == 'OpenAIAgent':
                tsbench_agent_answer = asyncio.run( inference_openai(self.tsbench_agent, tsbench_prompt.format(instruction=self.instruction, original_slides=self.original_slides, edited_slides=slides_gen)) )
            elif self.agent_type == 'AWorld':
                tsbench_agent_answer = inference_aworld(self.tsbench_agent, tsbench_prompt.format(instruction=self.instruction, original_slides=self.original_slides, edited_slides=slides_gen)) # tsbench_messages, tsbench_group)

            if verbose:
                print('TSBench Agent answer : ', tsbench_agent_answer)
            if tsbench_agent_answer != '':
                tsbench_agent_answer_dict = json.loads(tsbench_agent_answer) # AGENT RETURNS A STRING, SO WE NEED TO CONVERT TO DICT

        return coherence_agent_answer_dict, content_agent_answer_dict, tsbench_agent_answer_dict









