# from scipy.optimize import linear_sum_assignment
import os
from tqdm import tqdm
from pathlib import Path
import subprocess
import numpy as np
import re
from bs4 import BeautifulSoup
from PIL import Image

# =========================
# PDF / Document Processing
# =========================
import fitz  # PyMuPDF
import pymupdf.layout
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Disable link extraction globally
def _safe_get_links(self):
    return []

fitz.Page.get_links = _safe_get_links


TEMP_IMAGES_PATH_GT = '/tmp/gt_images'
TEMP_IMAGES_PATH_GEN = '/tmp/gen_images' 
TEMP_IMAGES_PATH_DECK = '/tmp/deck_images' 


def combine_slide_embeddings_concat(text_emb, visual_emb, w_text=0.6, w_visual=0.4):
    # Normalize each modality
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    visual_emb = visual_emb / (np.linalg.norm(visual_emb, axis=1, keepdims=True) + 1e-8)

    # Scale by weights
    text_emb_scaled = w_text * text_emb
    visual_emb_scaled = w_visual * visual_emb

    # Concatenate along last axis
    combined = np.concatenate([text_emb_scaled, visual_emb_scaled], axis=1)

    # Normalize combined vector
    combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
    return combined

def compute_slide_level_visual_embeddings(slides, embedder, text_field='text', verbose=False):
    """
    Embed slide visuals with fallback to text if no images.
    """
    vectors = []
    for s in tqdm(slides, disable= not verbose, desc="Embedding slide visuals with fallback"):

        imgs = []
        for src in s.get('images', []):
            img = load_image_from_file(src) 
            if img is not None:
                imgs.append(img)

        emb = None
        if imgs:
            im_embs = embedder.embed_images(imgs)
            if im_embs.size > 0:
                avg = im_embs.mean(axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                emb = avg

        # fallback to text
        if emb is None:
            text = s.get(text_field, "")
            if text.strip():
                text_emb = embedder.embed_texts_clip([text])[0]
                norm = np.linalg.norm(text_emb)
                if norm > 0:
                    text_emb = text_emb / norm
                emb = text_emb

        vectors.append(emb)

    # Determine embedding dimension safely
    if hasattr(embedder.clip_model, 'text_projection'):
        dim = embedder.clip_model.text_projection.out_features
    elif vectors and vectors[0] is not None:
        dim = vectors[0].shape[0]
    else:
        dim = 512

    # Replace None with zeros
    out = np.zeros((len(vectors), dim), dtype=float)
    for i, v in enumerate(vectors):
        if v is not None:
            out[i,:] = v

    return out


def get_valid_slide_texts(slide_dicts, minimum_text_length=50):
    # Select valid slides
    slide_texts = []
    valid_slide_indices = []
    for sidx, s in enumerate(slide_dicts):
        if 'text' in s.keys():
            slide_text  = s['text']
            if len(slide_text) >= minimum_text_length:
                slide_texts.append(slide_text)
                valid_slide_indices.append(sidx)

    return slide_texts, valid_slide_indices

# -------------------------
# -- External function call

# async def convert_slides_to_pdf( slide_path_gen):
#     # Prepare generated slide in pdf format
#     base_name, extension = os.path.splitext(slide_path_gen)
#     if extension == '.html':
#         # PDF path for generated html
#         gen_html_folder = os.path.dirname(slide_path_gen)    
#         html_file = os.path.basename(slide_path_gen)
#         file_path = Path(html_file)
#         gen_file_name = file_path.stem
#         gen_pdf_path = os.path.join(gen_html_folder, gen_file_name+'.pdf')

#         # Convert HTML to PDF
#         generated_pdf_path = await convert_html_to_pdf(slide_path_gen, gen_pdf_path)

#     elif extension == '.pdf':
#         gen_pdf_path = slide_path_gen

#     return gen_pdf_path


def split_paper_sections(paper_file_path, filter_keywords = ['reference', 'acknowledgment'], remove_table=True):
    pdf_doc = pymupdf.open(paper_file_path)

    # Redact tables (remove tables)
    if remove_table:
        for page in pdf_doc:
            tables = page.find_tables()
            for table in tables:
                try:
                    page.add_redact_annot(table.bbox)         
                except ValueError:
                    # print('Table attribute error')
                    pass

            # Apply the redactions to permanently remove the content
            page.apply_redactions()

    # pdf_doc.set_metadata({}) # can not remove picture text 
    paper_md_text = pymupdf4llm.to_markdown(pdf_doc, header=False, footer=False)

    # Remove picture text part
    sub_start = '**==> picture'
    sub_end = 'omitted <==**'
    paper_md_text = re.sub(r'{}.*?{}'.format(re.escape(sub_start),re.escape(sub_end)),'',paper_md_text)
    sub_start = '**----- Start of picture text -----**'
    sub_end = '**----- End of picture text -----**<br>'
    paper_md_text = re.sub(r'{}.*?{}'.format(re.escape(sub_start),re.escape(sub_end)),'',paper_md_text, flags=re.DOTALL)


    # split sections and filter out
    paper_sections = []
    paper_text = ''
    if paper_md_text != '':
        # Define the headers to split on and their corresponding names
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
        # Initialize the splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # Split the Markdown text
        docs = markdown_splitter.split_text(paper_md_text)
        for didx, doc in enumerate(docs):
            meta_dict = doc.metadata
            header1 = header2 = header3 = ''
            if 'Header1' in meta_dict.keys():
                header1 = meta_dict['Header1'].lower()
            if 'Header2' in meta_dict.keys():
                header2 = meta_dict['Header2'].lower()
            if 'Header3' in meta_dict.keys():
                header3 = meta_dict['Header3'].lower()

            is_valid = True
            for keyword in filter_keywords:
                if keyword in header2 or keyword in header3:
                    is_valid = False
                    break
            if is_valid:
                if didx==0:
                    header2 = header1+'\n\n'+header2 # add paper title
                paper_text += header2 
                paper_text += '\n'
                paper_text += doc.page_content
                paper_text += '\n\n'

                section_dict = {}
                section_dict['header'] = header2
                section_dict['content'] = doc.page_content
                paper_sections.append(section_dict)

    return paper_sections, paper_text

def filter_chunck_paper(paper_file_path, filter_keywords = ['reference', 'acknowledgment'], paper_chunk_size=1000, minimum_text_length=50, remove_table=True):

    paper_sections, _ = split_paper_sections(paper_file_path, filter_keywords = filter_keywords, remove_table=remove_table)

    paper_chuncks = []
    for section_dict in paper_sections:
        header = section_dict['header']
        content = section_dict['content']

        section_text = header
        section_text += '\n'
        section_text += content

        section_chunks = []
        for i in range(0, len(section_text), paper_chunk_size):
            chunk_text = section_text[i:i + paper_chunk_size] 
            if len(chunk_text) >= minimum_text_length: # filter out small chunk
                section_chunks.append(chunk_text)

        if len(section_chunks) > 0:
            paper_chuncks.extend(section_chunks)

    return paper_chuncks

def calculate_initial_embeddings(engine, slide_pdf_path_gt, slide_pdf_path_gen, paper_path, paper_chunk_size=1000, verbose=False): #, output_prefix="slide_similarity_results"):

    minimum_text_length = 50

    # Extract all slides
    slides_gt = extract_pdf_slides(slide_pdf_path_gt, TEMP_IMAGES_PATH_GT)
    slides_gen = extract_pdf_slides(slide_pdf_path_gen, TEMP_IMAGES_PATH_GEN)

    if verbose:
        print('slides_gt ------------------------------------')
        for sidx, slide_gt in enumerate(slides_gt):
            print(f'slide {sidx} : ', slide_gt['text'], '\n')
        print('slides_gen ------------------------------------')
        for sidx, slide_gen in enumerate(slides_gen):
            print(f'slide {sidx} : ', slide_gen['text'], '\n')

        print(f"Found {len(slides_gt)} ground-truth slides and {len(slides_gen)} generated slides.")



    # text embeddings
    texts_gt, valid_slide_indices_gt = get_valid_slide_texts(slides_gt, minimum_text_length=minimum_text_length)
    texts_gen, valid_slide_indices_gen = get_valid_slide_texts(slides_gen, minimum_text_length=minimum_text_length)

    if verbose:
        print("Computing text embeddings...")
    txt_emb_gt = engine.embed_texts(texts_gt) if len(texts_gt) > 0 else np.zeros((0, engine.text_model.get_sentence_embedding_dimension()))
    txt_emb_gen = engine.embed_texts(texts_gen) if len(texts_gen) > 0 else np.zeros((0, engine.text_model.get_sentence_embedding_dimension()))


    # Select valid slides
    slides_gt = [slides_gt[i] for i in valid_slide_indices_gt]
    slides_gen = [slides_gen[i] for i in valid_slide_indices_gen]



    # Paper embeddings
    if verbose:
        print("→ Loading paper markdown...")
    # From paper, extract and filter content
    filter_keywords = ['reference', 'acknowledgment']
    paper_chunks = filter_chunck_paper(paper_path, filter_keywords = filter_keywords, paper_chunk_size=paper_chunk_size, minimum_text_length=minimum_text_length, remove_table=True)

    # Chunk long paper text to avoid embedding length issues
    if verbose:
        print("→ Computing paper embeddings...")
    paper_embs = engine.embed_texts(paper_chunks)

    # Visual embeddings
    if verbose:
        print("Computing visual embeddings (CLIP) -- this may take some time if slides include many images...")
    vis_emb_gt = compute_slide_level_visual_embeddings(slides_gt, engine)
    vis_emb_gen = compute_slide_level_visual_embeddings(slides_gen, engine)

    # For ground truth slides
    gt_combined = combine_slide_embeddings_concat(txt_emb_gt, vis_emb_gt, w_text=0.6, w_visual=0.4)
    # For generated slides
    gen_combined = combine_slide_embeddings_concat(txt_emb_gen, vis_emb_gen, w_text=0.6, w_visual=0.4)


    return slides_gen, slides_gt, txt_emb_gen, txt_emb_gt, vis_emb_gen, vis_emb_gt, paper_embs, gen_combined, gt_combined, valid_slide_indices_gen, valid_slide_indices_gt

def calculate_paper_embeddings(engine, paper_path, paper_chunk_size=1000, minimum_text_length = 50, verbose=False):

    # Paper embeddings
    if verbose:
        print("→ Loading paper markdown...")
    # From paper, extract and filter content
    filter_keywords = ['reference', 'acknowledgment']
    paper_chunks = filter_chunck_paper(paper_path, filter_keywords = filter_keywords, paper_chunk_size=paper_chunk_size, minimum_text_length=minimum_text_length, remove_table=True)

    # Chunk long paper text to avoid embedding length issues
    if verbose:
        print("→ Computing paper embeddings...")
    paper_embs = engine.embed_texts(paper_chunks)    

    return paper_embs


def calculate_slide_embeddings(engine, slide_pdf_path, minimum_text_length = 50, verbose=False):

    if verbose:
        print("→ Extracting slides...")

    slides = extract_pdf_slides(slide_pdf_path, TEMP_IMAGES_PATH_DECK)

    texts, valid_slide_indices = get_valid_slide_texts(slides, minimum_text_length=minimum_text_length)

    if verbose:
        print("→ Computing slides embeddings...")

    txt_emb = engine.embed_texts(texts) if len(texts) > 0 else np.zeros((0, engine.text_model.get_sentence_embedding_dimension()))

    slides = [slides[i] for i in valid_slide_indices]
    vis_emb = compute_slide_level_visual_embeddings(slides, engine)

    return slides, txt_emb, vis_emb, valid_slide_indices


# -------------------------
# Utilities: Convert html to images
# -------------------------
def convert_html_to_images(input_html_path, output_image_folder, slide_pages = "", screenshot_size = "800x600", image_ext = "png", remove_old_files=True):
    """Convert html file to the ouput image file with png format.
    
    Args:
        input_html_path: The path to the html file to convert.
        output_image_folder : The output folder path
        slide_pages : slide number to convert
        screenshot_size: the size of output png images.
        image_ext : image file type
        remove_old_files : option to remove old image files
    Returns:
        the folder path to the generated image files
    """

    http_proxy = os.environ.get('http_proxy')

    output_file = "slides_temp.pdf" #temporary

    if not os.path.exists(input_html_path):
        return 'Error : Input HTML file not exist.'

    if remove_old_files:
        if os.path.exists(output_image_folder):
            # Remove the directory recursively
            try:
                shutil.rmtree(output_image_folder)
            except OSError as e:
                pass

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

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

        "--screenshots", # convert to images
        "--screenshots-size=" + screenshot_size,
        "--screenshots-format=" + image_ext,
        "--screenshots-directory=" + output_image_folder,        

        "reveal",

        input_html_path ,
        output_file    
    ]

    if slide_pages != "":
        command.insert(1, "--slides=" + slide_pages)


    try:
        # This command is designed to fail
        result = subprocess.run(command, capture_output=True, text=True,  check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command to convert html to pdf failed with error: {e}")

    # Remove pdf file
    if os.path.exists(output_file):
        # Try to delete the file.
        try:
            os.remove(output_file)
        except OSError as e:
            # If it fails, inform the user.
            print("Error: %s - %s." % (e.filename, e.strerror))
            return f"Error: {e.filename} - {e.strerror}"

    return output_image_folder #str(len(image_files))


# -------------------------
# Utilities: Convert html to PDF
# -------------------------
async def html_slides_to_pdf(input_html_path, output_pdf_file):
    import os
    import base64
    import asyncio
    import fitz  # PyMuPDF
    from playwright.async_api import async_playwright
    from urllib.parse import urlparse

    images_path = '/tmp/slides_images'
    # os.makedirs("/tmp/slides_images", exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Proxy setup (if needed)
    http_proxy = os.environ.get("http_proxy")
    proxy = None
    if http_proxy:
        parsed = urlparse(http_proxy)
        proxy = {
            "server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        }
        if parsed.username:
            proxy["username"] = parsed.username
        if parsed.password:
            proxy["password"] = parsed.password

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, proxy=proxy)
        page = await browser.new_page()
        await page.goto(f"file://{input_html_path}", wait_until="load")

        # Get slidesBase64 from the page
        slides_base64 = await page.evaluate("slidesBase64 || []")
        if not slides_base64:
            raise RuntimeError("slidesBase64 is not defined or empty in HTML")

        slide_count = len(slides_base64)
        print(f"[DEBUG] detected slide count: {slide_count}")

        iframe = await page.query_selector("#frame")
        next_btn = await page.query_selector("#next")

        pdf_doc = fitz.open()  # new PDF

        for i in range(slide_count):
            print(f"[DEBUG] rendering slide {i+1}/{slide_count}")

            # Get slide HTML and decode text
            slide_html = base64.b64decode(slides_base64[i]).decode("utf-8")
            # Extract inner text from body
            import re
            text_match = re.search(r"<body.*?>(.*?)</body>", slide_html, re.DOTALL | re.IGNORECASE)
            slide_text = text_match.group(1) if text_match else ""
            # Remove extra HTML tags
            slide_text = re.sub(r"<[^>]+>", "", slide_text).strip()

            # Screenshot the iframe
            # img_path = f"/tmp/slides_images/slide_{i}.png"
            img_path = images_path + f"/slide_{i}.png"
            await iframe.screenshot(path=img_path)
            
            # Create PDF page with image
            img = fitz.open(img_path)  # open image as PDF
            rect = img[0].rect
            pdf_page = pdf_doc.new_page(width=rect.width, height=rect.height)
            pdf_page.insert_image(rect, filename=img_path)

            # Insert invisible text layer
            pdf_page.insert_text((0, 0), slide_text, fontsize=12, overlay=True, render_mode=3)  # render_mode=3 = invisible

            # Go to next slide
            if i < slide_count - 1:
                await next_btn.click()
                await asyncio.sleep(0.3)  # wait for slide to render

        await browser.close()
        pdf_doc.save(output_pdf_file)
        pdf_doc.close()
        print(f"[INFO] Searchable PDF generated: {output_pdf_file}")

    if os.path.exists(images_path):
        try:
            shutil.rmtree(images_path)
        except OSError as e:
            pass

# async def convert_html_to_pdf(input_html_path, output_pdf_file, slide_pages = ""):
#     """Convert html file to the ouput pdf file.
    
#     Args:
#         input_html_path: The path to the html file to convert.
#         output_pdf_file : output PDF file path
#         slide_pages : slide number to convert
#     Returns:
#         the folder path to the generated pdf file
#     """

#     def count_sections_in_html(filename):
#         from bs4 import BeautifulSoup
        
#         # Ensure the file exists before trying to open it
#         if not os.path.exists(filename):
#             print(f"Error: The file '{filename}' was not found.")
#             return 0

#         try:
#             # Open and read the HTML file
#             with open(filename, 'r', encoding='utf-8') as file:
#                 html_content = file.read()
                
#             # Parse the HTML content using Beautiful Soup
#             soup = BeautifulSoup(html_content, 'html.parser')
            
#             # Find all occurrences of the <section> tag
#             sections = soup.find_all('section')
            
#             # The number of tags is the length of the list returned by find_all
#             return len(sections)

#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return 0

#     http_proxy = os.environ.get('http_proxy')
#     if not os.path.exists(input_html_path):
#         # raise RuntimeError(f"Error : Input HTML file not exist.")
#         return 'Error : Input HTML file not exist.'

#     # Decide slide range
#     num_sections = count_sections_in_html(input_html_path)
#     if num_sections > 0:
#         slide_range1 = "--slides"
#         slide_range2 = f"1-{num_sections}"
#         print("    : Calculated slide range : ", slide_range2)
#     else:
#         slide_range1 = ""
#         slide_range2 = ""

#     command = [
#         "decktape",

#         "--chrome-path=/usr/bin/google-chrome",
#         "--chrome-arg=--no-sandbox",
#         "--chrome-arg=--disable-dev-shm-usage",
#         "--chrome-arg=--no-zygote",
#         "--chrome-arg=--disable-gpu",
#         "--chrome-arg=--ignore-certificate-errors",
#         "--chrome-arg=--allow-running-insecure-content",
#         "--chrome-arg=--allow-file-access-from-files",
#         "--chrome-arg=--disable-web-security",

#         f"--chrome-arg=--proxy-server={http_proxy}",

#         "reveal",

#         slide_range1,
#         slide_range2,

#         input_html_path ,
#         output_pdf_file    
#     ]

#     if slide_pages != "":
#         command.insert(1, "--slides=" + slide_pages)

#     try:
#         result = subprocess.run(command, capture_output=True, text=True,  check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error: Command to convert html to pdf failed with error: {e}")
#         print(f"-- Trying with Chrome instead...")

#         try:
#             await html_slides_to_pdf(input_html_path, output_pdf_file)
#         except:
#             print('Error: Failed to conver with Chrome.')
#             return ''

#     except subprocess.TimeoutExpired as e:
#         print(f"Command timed out after {e.timeout} seconds.")

#         try:
#             await html_slides_to_pdf(input_html_path, output_pdf_file)
#         except:
#             print('Error: Failed to conver with Chrome.')
#             return ''
#     except FileNotFoundError:
#         print(f"Error: Command '{command[0]}' not found.")
#         return ''

#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return ''

#     if os.path.exists(output_pdf_file):
#         return output_pdf_file 
#     else:
#         return ''


def extract_pdf_slides(pdf_path, output_folder=None):
    """
    Converts a PDF (of presentation slides) into structured slide data.

    Returns:
        list[dict]: Each dict contains:
            {
                'id': str,
                'text': str,
                'images': [str],
                'elements': [{'pos': (x, y), 'size': (w, h)}],
                'page_width': float,
                'page_height': float
            }
    """
    doc = fitz.open(pdf_path)
    slides = []

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = os.path.join(os.path.dirname(pdf_path), "images")
        os.makedirs(output_folder, exist_ok=True)

    for i, page in enumerate(doc):
        slide_id = f"slide-{i}"

        # ---------- Page size ----------
        page_width = page.rect.width
        page_height = page.rect.height

        # ---------- Text ----------
        text = page.get_text("text") or ""

        # ---------- Images ----------
        images = []
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_name = f"{slide_id}_img{img_index}.{ext}"

            img_path = os.path.join(output_folder, image_name)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images.append(img_path)

        # ---------- Elements (layout blocks) ----------
        elements = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text_block, *_ = block
            elements.append({
                "pos": (x0, y0),
                "size": (x1 - x0, y1 - y0)
            })

        slides.append({
            "id": slide_id,
            "text": text.strip(),
            "images": images,
            "elements": elements,
            "page_width": page_width,
            "page_height": page_height,
        })

    doc.close()
    return slides



# -------------------------
# Utilities: Image loader
# -------------------------
def load_image_from_src_old(src, base=None, resize=(224,224), timeout=10):
    """
    src: URL or local path or data URI. base used to resolve relative paths.
    Returns PIL.Image or None
    """
    if not src:
        return None
    # handle data URI
    if src.startswith("data:"):
        header, b64 = src.split(",", 1)
        import base64
        data = base64.b64decode(b64)
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            if resize:
                img = img.resize(resize, Image.LANCZOS)
            return img
        except Exception:
            return None
    # resolve relative
    if base and not urlparse(src).scheme:
        src_resolved = urljoin(base, src)
    else:
        src_resolved = src
    # local file
    if src_resolved.startswith("file://"):
        local_path = src_resolved[len("file://"):]
        if os.path.exists(local_path):
            try:
                img = Image.open(local_path).convert("RGB")
                if resize:
                    img = img.resize(resize, Image.LANCZOS)
                return img
            except Exception:
                return None
        else:
            return None
    # http(s)
    try:
        r = requests.get(src_resolved, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        return img
    except Exception:
        return None


# -------------------------
# Utilities: Local Image loader
# -------------------------
def load_image_from_file(local_path, resize=(224,224), timeout=10): # only for local image file
    """
    src: URL or local path or data URI. base used to resolve relative paths.
    Returns PIL.Image or None
    """
    if not local_path:
        return None

    # local file
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            if resize:
                img = img.resize(resize, Image.LANCZOS)
            return img
        except Exception:
            return None
    else:
        return None








# -------------------------
# Similarity computations
# -------------------------
def cosine_sim_matrix(A, B):
    # A: nxd, B: mxd; returns n x m cosine similarity
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    # vectors assumed normalized; but guard
    # compute with dot
    return (A @ B.T)


def max_cosine_sim_per_slide(gt_embs, gen_embs):
    """
    Computes max cosine similarity between GT images and GEN images per slide.
    
    gt_embs: list of arrays [(n_gt_i, D), ...]
    gen_embs: list of arrays [(n_gen_j, D), ...]
    
    Returns:
        sims: 2D array of shape (n_gt_slides, n_gen_slides)
            each value = max cosine similarity between all images on that slide pair
            (0 if either slide has 0 images)
    """
    n_gt = len(gt_embs)
    n_gen = len(gen_embs)
    sims = np.zeros((n_gt, n_gen), dtype=float)
    
    for i, gt_arr in enumerate(gt_embs):
        for j, gen_arr in enumerate(gen_embs):
            if gt_arr.shape[0] == 0 or gen_arr.shape[0] == 0:
                sims[i, j] = 0.0
            else:
                # cosine similarity matrix between all images in the slide pair
                sim_matrix = cosine_sim_matrix(gt_arr, gen_arr)  # shape (n_gt_imgs, n_gen_imgs)
                sims[i, j] = sim_matrix.max()  # take the maximum
    return sims


def is_within_canvas(bbox, page_width, page_height):
    x, y, w, h = bbox
    return (
        x >= 0 and
        y >= 0 and
        x + w <= page_width and
        y + h <= page_height
    )

def bboxes_overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    return not (
        x1 + w1 <= x2 or
        x2 + w2 <= x1 or
        y1 + h1 <= y2 or
        y2 + h2 <= y1
    )

def positional_similarity_matrix(slidesA, slidesB):
    """
    For each slide, compute normalized Manhattan distance between element centers:
    Approach:
    - For each slide, compute list of element centers (x + w/2, y + h/2) for elements that have pos+size.
    - For slide pair, compute bipartite matching between element centers using manhattan distances and average normalized distance.
    - Normalize by a heuristic 'slide size' (max coordinate seen in both slides); convert distance -> similarity = 1 - clipped_norm_dist
    If no positions, return 0.5 as neutral score.
    """
    n = len(slidesA)
    m = len(slidesB)
    out = np.zeros((n,m), dtype=float)
    for i, a in enumerate(slidesA):
        centers_a = []
        for el in a.get('elements', []):
            if el['pos'] and el['size']:
                x, y = el['pos']
                w, h = el['size']
                centers_a.append((x + w/2.0, y + h/2.0))
        for j, b in enumerate(slidesB):
            centers_b = []
            for el in b.get('elements', []):
                if el['pos'] and el['size']:
                    x, y = el['pos']
                    w, h = el['size']
                    centers_b.append((x + w/2.0, y + h/2.0))
            if len(centers_a) == 0 and len(centers_b) == 0:
                out[i,j] = 0.5
                continue
            if len(centers_a) == 0 or len(centers_b) == 0:
                out[i,j] = 0.25
                continue
            ca = np.array(centers_a)
            cb = np.array(centers_b)
            # compute Manhattan distance matrix
            da = np.abs(ca[:,None,:] - cb[None,:,:]).sum(axis=2)  # p x q
            # Solve assignment (min total distance) between smaller set to larger
            from scipy.optimize import linear_sum_assignment as hungarian
            # pad to square with large values to allow match
            p, q = da.shape
            if p <= q:
                row_ind, col_ind = hungarian(da)
                chosen = da[row_ind, col_ind].sum()
                mean_dist = chosen / p
            else:
                row_ind, col_ind = hungarian(da)
                chosen = da[row_ind, col_ind].sum()
                mean_dist = chosen / q
            # normalize by heuristic: max coordinate span
            coords = np.vstack([ca, cb])
            max_span = max(coords[:,0].max() - coords[:,0].min(), coords[:,1].max() - coords[:,1].min(), 1.0)
            norm = mean_dist / max_span
            # clamp and convert to similarity
            sim = 1.0 - min(1.0, norm)
            out[i,j] = float(sim)
    return out




