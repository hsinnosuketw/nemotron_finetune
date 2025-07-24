#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF helper utilities for CPIC pipeline.
Includes functions to download, parse, render, resize, and encode PDF pages.
"""
import os
import hashlib
import requests
from io import BytesIO
from typing import List, Tuple, Dict

from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image
import base64


def download_pdf(url: str) -> BytesIO:
    """
    Download a PDF from a URL and return a BytesIO buffer.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to download PDF: Status code {response.status_code}"
        )
    return BytesIO(response.content)


def get_pdf_images(pdf_url: str) -> Tuple[List[Image.Image], List[str]]:
    """
    Given a PDF URL, download it, extract text per page,
    and render each page to a PIL Image.

    Returns:
        images: list of PIL.Image for each page
        page_texts: list of extracted text (str) for each page
    """
    # Download to buffer and save to temp path
    pdf_buf = download_pdf(pdf_url)
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_buf.getvalue())

    # Extract texts
    reader = PdfReader(temp_path)
    page_texts: List[str] = []
    for pg in reader.pages:
        text = pg.extract_text() or ""
        page_texts.append(text)

    # Render to images
    images = convert_from_path(temp_path)

    if len(images) != len(page_texts):
        raise RuntimeError(
            f"Page count mismatch: {len(images)} images vs {len(page_texts)} texts"
        )

    # Clean up temp file
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return images, page_texts


def get_cpic_pdf_images_texts(
    directory: str
) -> List[Dict[str, List]]:
    """
    Read all PDFs in a directory and return list of dicts:
      {
        'path': full filepath,
        'name': filename,
        'images': [PIL.Image ...],
        'texts': [str ...]
      }
    """
    results: List[Dict[str, List]] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(directory, fname)

        reader = PdfReader(full_path)
        texts: List[str] = [pg.extract_text() or "" for pg in reader.pages]
        images = convert_from_path(full_path)

        if len(images) != len(texts):
            raise RuntimeError(
                f"Page count mismatch in {fname}: "
                f"{len(images)} images vs {len(texts)} texts"
            )

        results.append({
            "path": full_path,
            "name": fname,
            "images": images,
            "texts": texts,
        })

    return results


def resize_image(
    image: Image.Image,
    max_height: int = 800
) -> Image.Image:
    """
    Resize the PIL Image to have at most max_height, preserving aspect ratio.
    """
    width, height = image.size
    if height <= max_height:
        return image
    scale = max_height / height
    new_size = (int(width * scale), max_height)
    return image.resize(new_size, Image.LANCZOS)


def open_pdf_page(
    pdf_path: str,
    page_number: int
) -> Image.Image:
    """
    Load a single page from a local PDF file as a PIL Image.
    """
    pages = convert_from_path(pdf_path)
    if page_number < 0 or page_number >= len(pages):
        raise IndexError(
            f"Page {page_number} out of range for {pdf_path} ({len(pages)} pages)"
        )
    return pages[page_number]


def get_base64_image(
    image: Image.Image
) -> str:
    """
    Convert a PIL Image to a JPEG Base64-encoded string.
    """
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sha_id(
    name: str,
    page: int
) -> str:
    """
    Return a deterministic SHA-256 hex digest for a document page ID.
    """
    return hashlib.sha256(f"{name}_{page}".encode()).hexdigest()


def image_to_base64(
    image: Image.Image,
    max_height: int = 800
) -> str:
    """
    Resize an image and return a Base64-encoded JPEG string.

    Reuses resize_image and get_base64_image.
    """
    resized = resize_image(image, max_height)
    return get_base64_image(resized)
