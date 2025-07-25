#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Send one page of a PDF to an NVIDIA/OpenAI multimodal model
using pdf2image + PyMuPDF backend (no poppler needed).
"""

import fitz  # pip install pymupdf
from pathlib import Path
import base64, os, io
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_path     # uses PyMuPDF when use_fitz=True

# ------------------------------------------------------------------ #
# 1. Extract *one page* with PyMuPDF backend
# ------------------------------------------------------------------ #



def pdf_page_to_base64(pdf_path: str | Path, page_index: int = 0) -> str:
    """
    Render a single PDF page with PyMuPDF and return base-64 PNG bytes.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= doc.page_count:
        raise ValueError(f"Invalid page index {page_index} for {pdf_path}")

    page = doc.load_page(page_index)
    pix  = page.get_pixmap(dpi=200)      # adjust DPI if needed

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------------------------------------------ #
# 2. Build the correct multimodal message
# ------------------------------------------------------------------ #
def make_mm_message(text: str, b64_png: str) -> list[dict]:
    """
    Returns the message array expected by OpenAI/NVIDIA multimodal chat.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_png}",
                        "detail": "auto",
                    },
                },
            ],
        }
    ]

# ------------------------------------------------------------------ #
# 3. Send to NVIDIA / OpenAI multimodal endpoint
# ------------------------------------------------------------------ #
def send_pdf_page(
    pdf_file    : str | Path,
    page_idx    : int,
    user_prompt : str,
    model       : str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    api_key     : str | None = os.getenv("NVIDIA_API_TOKEN"),
    base_url    : str = "https://integrate.api.nvidia.com/v1",
    temperature : float = 0.7,
    top_p       : float = 0.95,
    max_tokens  : int = 1024,
):
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_TOKEN (or pass api_key=…)")

    client = OpenAI(base_url=base_url, api_key=api_key)

    img_b64   = pdf_page_to_base64(pdf_file, page_idx)
    messages  = make_mm_message(user_prompt, img_b64)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )

    print("---- Response ----")
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


    
