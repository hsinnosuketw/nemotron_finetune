# pdf_helper.py
# ---------------------------------------------------------------------
# Utilities for CPIC guideline PDFs  –  PyMuPDF backend (no poppler)
#
# pip install pymupdf pillow requests
# ---------------------------------------------------------------------
from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path
from typing import List, Tuple

import fitz                    # PyMuPDF
import requests
from PIL import Image

# ------------------------------------------------------------------ #
# 1. Download helper (unchanged)                                     #
# ------------------------------------------------------------------ #
def download_pdf(url: str) -> io.BytesIO:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return io.BytesIO(resp.content)

# ------------------------------------------------------------------ #
# 2. Page extraction (images + text)                                 #
# ------------------------------------------------------------------ #
def _open_doc(pdf_source: io.BytesIO | str | Path) -> fitz.Document:
    """
    Accept BytesIO **or** path and return a fitz.Document.
    """
    if isinstance(pdf_source, io.BytesIO):
        # need bytes – cannot pass the BytesIO object directly
        return fitz.open(stream=pdf_source.getvalue(), filetype="pdf")
    else:
        return fitz.open(str(pdf_source))   # pathlib.Path or str

def get_pdf_images(
    pdf_buf: io.BytesIO | str | Path,
) -> Tuple[List[Image.Image], List[str]]:
    """
    Render every page to a Pillow image *and* extract its text.

    Returns
    -------
    images : list[PIL.Image.Image]
    texts  : list[str]
    """
    doc = _open_doc(pdf_buf)
    images, texts = [], []

    for page in doc:
        pix = page.get_pixmap(dpi=200)          # adjust DPI as needed
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        texts.append(page.get_text("text"))
    return images, texts

# ------------------------------------------------------------------ #
# 3. Directory-level loader                                          #
# ------------------------------------------------------------------ #
def get_cpic_pdf_images_texts(path: str | os.PathLike) -> List[dict]:
    """
    Walk `path` for *.pdf files and return
    [{'path','name','images','texts'}, …]
    """
    out = []
    for pdf_file in sorted(Path(path).glob("*.pdf")):
        imgs, txts = get_pdf_images(pdf_file)
        out.append(
            dict(
                path=str(pdf_file),
                name=pdf_file.name,
                images=imgs,
                texts=txts,
            )
        )
    return out

# ------------------------------------------------------------------ #
# 4. Image helpers (unchanged API)                                   #
# ------------------------------------------------------------------ #
def resize_image(image: Image.Image, max_height: int = 800) -> Image.Image:
    w, h = image.size
    if h <= max_height:
        return image
    ratio = max_height / h
    return image.resize((int(w * ratio), max_height))

def get_base64_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_to_base64(image: Image.Image, size: int = 640) -> str:
    img = resize_image(image, size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------------------------------------------ #
# 5. Misc helpers used by pipeline                                   #
# ------------------------------------------------------------------ #
def sha_id(name: str, page: int) -> str:
    """Deterministic SHA-256 page id."""
    return hashlib.sha256(f"{name}_{page}".encode()).hexdigest()

def open_pdf_page(pdf_path: str | Path, page_number: int) -> Image.Image:
    """
    Convenience wrapper used by `save_hits` in retrieve_cpic.py.
    Returns a Pillow image of the page.
    """
    doc = fitz.open(str(pdf_path))
    if page_number < 0 or page_number >= doc.page_count:
        raise IndexError(f"{pdf_path} has no page {page_number}")
    pix = doc.load_page(page_number).get_pixmap(dpi=200)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
