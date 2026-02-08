import torch
import os
import sys
import shutil
import numpy as np
from PIL import Image
import importlib
from typing import Any

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception as e:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from e

try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
except Exception as e:
    raise ImportError("Please install transformers and torch: pip install transformers torch torchvision") from e

# =========================
# NLP / Vision Models
# =========================
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except ImportError as e:
    raise ImportError(
        "Please install sentence-transformers: pip install sentence-transformers"
    ) from e

def safe_copy(src: str, dst: str):
    if os.path.exists(dst):
        print(f"⚠️  {dst} already exists, skipping copy.")
        return
    try:
        shutil.copytree(src, dst)
        print(f"Copied {src} → {dst}")
    except Exception as e:
        print(f"Copy error: {e}")


def import_from_string(module_path: str) -> Any:
    if not module_path or not isinstance(module_path, str):
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected a non-empty string like 'module.path.ObjectName'."
        )

    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected format 'module.path.ObjectName', got {len(parts)} part(s)."
        )

    module_name, object_name = parts

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_name}' from path '{module_path}': {e}"
        ) from e

    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{object_name}'."
        ) from e

    return obj


def import_class_from_string(module_class_path: str)->Any:
    module_name = module_class_path.rsplit('.', 1)[0]
    class_name = module_class_path.rsplit('.', 1)[-1]

    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        
        # Get the class from the module using getattr
        Class_dynamic = getattr(module, class_name)
                
    except ImportError:
        print(f"Module {module_name} not found.")
    except AttributeError:
        print(f"Class {class_name} not found in {module_name}.")

    return Class_dynamic



def filter_extracted_markdown(paper_md_text, filter_keywords = ['references', 'reference', 'acknowledgements', 'acknowledgement', 'appendix', 'appendices']):
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    # valid_section_titles = ['abstract', 'introduction', 'background', 'implementation', 'experiment',  'conclusion']
    # make valid title more strict
    valid_section_titles = ['abstract', 'introduction', 'background', 'conclusion']
    min_invalid_index = 3

    final_section_title = 'conclusion'
    
    # print('paper_md_text : ', paper_md_text)

    # split sections and filter out
    paper_text = paper_md_text
    if paper_md_text != '':
        # Define the headers to split on and their corresponding names
        headers_to_split_on = [
            ("#", "Header1")
        ]
        # Initialize the splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Find invalid section
        collected_valid_section_titles = []
        valid_indices = []
        invalid_indices = []
        final_section_indices = []
        docs = markdown_splitter.split_text(paper_md_text)
        for didx, doc in enumerate(docs):
            meta_dict = doc.metadata
            header1 = header2 = header3 = ''
            if 'Header1' in meta_dict.keys():
                header1 = meta_dict['Header1'].lower()

            # print(didx, ' : Header1 : ', header1)

            # Find invalid sections
            is_invalid = False
            for keyword in filter_keywords:
                if keyword in header1: #rr keyword in header2 or keyword in header3:
                    is_invalid = True
                    break
            if is_invalid:
                invalid_indices.append(didx)

            # Find valid sections
            is_valid = False
            found_keyword = ''
            for keyword in valid_section_titles:
                if keyword in header1 and keyword not in collected_valid_section_titles: #rr keyword in header2 or keyword in header3:
                    is_valid = True
                    found_keyword = keyword
                    break
            if is_valid:
                valid_indices.append(didx)
                if found_keyword != '':
                    collected_valid_section_titles.append(found_keyword)
            # else:
            #     print('Invalid header : ', header1)

            # Find final section titles
            if final_section_title in header1:
                final_section_indices.append(didx)

        # Find the first invalid index it is after all valid sections
        last_valid_index = -1
        if len(valid_indices) > 0:
            last_valid_index = valid_indices[-1]
        first_invalid_index = -1
        for invalid_index in invalid_indices:
            if invalid_index > last_valid_index:
                first_invalid_index = invalid_index
                break

        # print('')
        # print('valid_indices : ', valid_indices)
        # print('last_valid_index : ', last_valid_index)
        # print('invalid_indices : ', invalid_indices)
        # print('first_invalid_index : ', first_invalid_index)

        # if found first invalid index is too small, it means it was not found properly.
        # Then use the final section(such as Conclusion) and set the next section as first invalid index
        if first_invalid_index < min_invalid_index and len(final_section_indices) > 0:
            final_section_index = final_section_indices[-1]
            first_invalid_index = final_section_index + 1


        # Collect all valid section contents
        if first_invalid_index >= min_invalid_index:
            paper_text = ''
            docs = markdown_splitter.split_text(paper_md_text)
            for didx, doc in enumerate(docs):
                if didx >= first_invalid_index: # don't include the first invalid section and after
                    break

                meta_dict = doc.metadata
                header1 = ''
                if 'Header1' in meta_dict.keys():
                    header1 = meta_dict['Header1'].lower()

                paper_text += '# ' + header1
                paper_text += '\n'
                paper_text += doc.page_content
                paper_text += '\n\n'



        # print('')
        # print('filtered paper_text : ', paper_text)
        # print('')

        # # Split the Markdown text
        # docs = markdown_splitter.split_text(paper_md_text)
        # for didx, doc in enumerate(docs):
        #     meta_dict = doc.metadata
        #     header1 = header2 = header3 = ''
        #     if 'Header1' in meta_dict.keys():
        #         header1 = meta_dict['Header1'].lower()

        #     is_valid = True
        #     for keyword in filter_keywords:
        #         if keyword in header1: #rr keyword in header2 or keyword in header3:
        #         # if keyword == header1: #rr keyword in header2 or keyword in header3:
        #             # print(f"Metadata: {doc.metadata}")
        #             # print(f"Content: {doc.page_content}")
        #             # print("-" * 20)    
        #             is_valid = False
        #             break

        #     if is_valid:
        #         paper_text += '# ' + header1
        #         paper_text += '\n'
        #         paper_text += doc.page_content
        #         paper_text += '\n\n'
        #     else:
        #         break

    return paper_text



# -------------------------
# Embedding utils
# -------------------------
class EmbeddingEngine:
    def __init__(self, device=None, text_model_name="/root/data/models/all-MiniLM-L6-v2", clip_model_name="/root/data/models/clip-vit-base-patch32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # text
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        # CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

    def embed_texts(self, texts, batch_size=32):
        # returns numpy array (n, d)
        embs = self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
        return embs

    def embed_texts_clip(self, texts, batch_size=32):
        """
        Embed text using CLIP text encoder (same space as images).
        Returns numpy array (n, 512), normalized.
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.clip_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                all_embs.append(text_features.cpu().numpy())
        return np.vstack(all_embs)

    def embed_images(self, pil_images, batch_size=16):
        """
        pil_images: list of PIL.Image objects (or None)
        returns numpy array (n, d) normalized
        """
        if len(pil_images) == 0:
            return np.zeros((0, self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else 512))
        all_embs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            # replace None with a blank RGBA image
            batch_proc = [img if isinstance(img, Image.Image) else Image.new("RGB", (224,224), (255,255,255)) for img in batch]
            inputs = self.clip_processor(images=batch_proc, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_outputs = self.clip_model.get_image_features(**inputs)
                image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
                image_np = image_outputs.cpu().numpy()
            all_embs.append(image_np)
        return np.vstack(all_embs)

        