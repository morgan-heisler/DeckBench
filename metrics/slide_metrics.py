from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from .metric_utils import cosine_sim_matrix, max_cosine_sim_per_slide, positional_similarity_matrix, is_within_canvas, bboxes_overlap



class SlideMetrics:

    def __init__(self):
        self.engine = None

        self.ppl_model = self.ppl_tokenizer = None

        self.paper_embs = None
        self.slides_gt = self.txt_emb_gt = self.vis_emb_gt = None
        self.slides_gen = self.txt_emb_gen = self.vis_emb_gen = None

    # -------------------------
    # Slide-level Reference-free functions
    # -------------------------
    # load once globally
    def load_gpt2_model(self, model_path='/root/data/models/gpt2'):
        self.ppl_model = GPT2LMHeadModel.from_pretrained(model_path) 
        self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained(model_path) 

    def compute_perplexity(self, text: str) -> float:
        encodings = self.ppl_tokenizer(text, return_tensors="pt", truncation=True)
        max_length = self.ppl_model.config.n_positions
        stride = 512
        lls = []

        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = i + stride
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            lls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    @staticmethod
    def calculate_slide_layout_metrics(elements, page_width, page_height):
        """
        elements: list of dicts with keys:
            - id
            - type: title | bullet | image | figure | text
            - bbox: (x, y, w, h)
            - font_size (optional)
        """

        # ---- Element-in-canvas ----
        in_canvas = {}
        for el in elements:
            in_canvas[el["id"]] = is_within_canvas(
                el["bbox"], page_width, page_height
            )

        # ---- Element overlap (True = no overlap) ----
        overlap_free = {el["id"]: True for el in elements}
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if bboxes_overlap(
                    elements[i]["bbox"], elements[j]["bbox"]
                ):
                    overlap_free[elements[i]["id"]] = False
                    overlap_free[elements[j]["id"]] = False

        # ---- Font size check ----
        titles = [
            el for el in elements
            if el["type"] == "title" and "font_size" in el
        ]
        bullets = [
            el for el in elements
            if el["type"] == "bullet" and "font_size" in el
        ]

        if titles and bullets:
            avg_title_size = sum(el["font_size"] for el in titles) / len(titles)
            avg_bullet_size = sum(el["font_size"] for el in bullets) / len(bullets)
            font_size_check = avg_title_size > avg_bullet_size
        else:
            font_size_check = None  # cannot evaluate

        return {
            "element_in_canvas": in_canvas,
            "element_overlap_free": overlap_free,
            "font_size_check": font_size_check,
        }

    @staticmethod
    def compute_figure_relevance(text_emb, image_emb):
        from torch.nn.functional import cosine_similarity
        # both should be normalized embeddings (already normalized if from CLIP)
        return cosine_similarity(text_emb, image_emb).mean().item()

    @staticmethod
    def compute_similarity(slide_text_emb, paper_embs):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(slide_text_emb, paper_embs)
        # you could take max (best matching section) or average
        return sims.max()


    # -------------------------
    # Slide-level Reference-based functions
    # -------------------------
    @staticmethod
    def order_aware_hungarian(text_sim, visual_sim, pos_sim, 
                            w_text=0.5, w_vis=0.3, w_pos=0.2, 
                            order_lambda=0.05):
        """
        Order-aware Hungarian matching for slide similarity.
        """

        n, m = text_sim.shape
        # Combined similarity (same as your current logic)
        combined = w_text * text_sim + w_vis * visual_sim + w_pos * pos_sim

        # Hungarian solves min cost; we want to maximize similarity -> negate it
        cost = -combined

        # Add order-aware penalty term |i - j| * λ
        i_idx = np.arange(n).reshape(-1, 1)
        j_idx = np.arange(m).reshape(1, -1)
        order_penalty = np.abs(i_idx - j_idx) * order_lambda
        cost = cost + order_penalty

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        return row_ind, col_ind, combined

    # -------------------------
    # --- Slide-level reference-free metrics ---
    # -------------------------
    @staticmethod
    def calculate_slide_faithfulness(slide_emb, paper_embs):
        slide_faithfulnesses = []
        for paper_chunk_emb in paper_embs:
            paper_emb = paper_chunk_emb.reshape(1, -1)
            similarity = float(SlideMetrics.compute_similarity(slide_emb, paper_emb))
            slide_faithfulnesses.append(similarity)
        slide_faithfulnesses_arr = np.array(slide_faithfulnesses)
        max_faithfulness = np.max(slide_faithfulnesses_arr)

        return max_faithfulness

    def calculate_slide_metrics_reference_free(self, verbose=False):

        if verbose:
            print("Computing slide-level metrics (perplexity, figure relevance, fidelity)...")

        device = self.engine.device
        clip_tokenizer = self.engine.clip_tokenizer  # assuming your EmbeddingEngine loads this
        clip_model = self.engine.clip_model

        slide_metrics_reference_free = []
        for i, s in enumerate(self.slides_gen):
            metrics = {"gen_index": i, "slide_id": s.get("id", f"slide_{i}")}
            text = s.get("text", "")
            imgs = s.get("images", [])
            elements = s.get("elements", [])
            page_width = s.get("page_width", "")
            page_height = s.get("page_height", "")

            # 1. Perplexity
            if len(text)>50:
                try:
                    metrics["perplexity"] = float(self.compute_perplexity(text))
                except Exception as e:
                    metrics["perplexity"] = None
                    print(f"[warn] Perplexity failed on slide {i}: {e}")
            else:
                metrics["perplexity"] = None

            # 2. Fidelity (if paper_embs provided)
            try:
                if self.paper_embs is not None and len(text.strip()) > 0:
                    slide_emb = self.txt_emb_gen[i].reshape(1, -1)
                    slide_faithfulness = SlideMetrics.calculate_slide_faithfulness(slide_emb, self.paper_embs)
                    metrics['faithfulness'] = slide_faithfulness
                    
                else:
                    metrics["faithfulness"] = None
            except Exception as e:
                metrics["faithfulness"] = None
                print(f"[warn] Faithfulness failed on slide {i}: {e}")

            ## Layout
            normalized_elements = []

            for i, el in enumerate(elements):
                if not el.get("pos") or not el.get("size"):
                    continue

                x, y = el["pos"]
                w, h = el["size"]

                normalized_elements.append({
                    "id": f"el-{i}",
                    "type": "text",  # or infer later
                    "bbox": (x, y, w, h),
                })

            metrics["layout"] = SlideMetrics.calculate_slide_layout_metrics(
                normalized_elements,
                page_width,
                page_height,
            ) 

            slide_metrics_reference_free.append(metrics)

        return slide_metrics_reference_free
        # --- END Slide-level reference-free METRICS SECTION ---

    # -------------------------
    # --- Slide-level reference-based metrics ---
    # -------------------------
    def calculate_slide_metrics_reference_based(self, verbose=False):

        # cosine similarity matrices
        if verbose:
            print("Computing cosine similarity matrices...")
        text_sim = cosine_sim_matrix(self.txt_emb_gt, self.txt_emb_gen)  # n x m
        # visual_sim = SlideEvaluator.cosine_sim_matrix(self.vis_emb_gt, self.vis_emb_gen)
        visual_sim = max_cosine_sim_per_slide(self.vis_emb_gt, self.vis_emb_gen)
        combined_sim = cosine_sim_matrix(self.gt_combined, self.gen_combined)

        # positional similarity
        if verbose:
            print("Computing positional similarities...")
        pos_sim = positional_similarity_matrix(self.slides_gt, self.slides_gen)


        # -- Slide-level reference-based

        # # Matching using Hungarian algorithm:
        # # We'll define a combined similarity to be used for matching. If you want separate matching per modality, change here.
        # # For now use simple average of the three similarities (treat missing visuals as 0)
        # # Normalize shapes: ensure text_sim, visual_sim, pos_sim have same shape
        n, m = text_sim.shape
        w_text, w_vis, w_pos = 0.3, 0.3, 0.6

        # # Hungarian solves min cost; we want max similarity -> cost = -combined
        row_ind, col_ind, combined = SlideMetrics.order_aware_hungarian(text_sim, visual_sim, combined_sim, 
                                w_text=w_text, w_vis=w_vis, w_pos=w_pos, 
                                order_lambda=0.05)


        return row_ind, col_ind, combined, text_sim, visual_sim, combined_sim, pos_sim


