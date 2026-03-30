"""
Extract structured content from the FTT book PDF.

Classifies text by font/size into semantic elements (h1-h3, p, blockquote, etc.)
and extracts images with captions. Outputs structured.json for downstream processing.
"""

import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Chapter map
# ---------------------------------------------------------------------------

CHAPTERS = [
    {"id": "cover", "title": "Cover", "pages": (1, 1)},
    {"id": "toc", "title": "Table of Contents", "pages": (2, 8), "skip_translate": True},
    {"id": "preface", "title": "Preface", "title_zh": "前言", "pages": (9, 19)},
    {"id": "contact", "title": "Forehand Contact", "title_zh": "正手击球接触", "pages": (20, 47)},
    {"id": "kinetic_chain", "title": "The Rotational Kinetic Chain Provides Unparalleled Speed", "title_zh": "旋转动力链带来无与伦比的速度", "pages": (48, 55)},
    {"id": "preparation", "title": "Forehand Preparation", "title_zh": "正手准备阶段", "pages": (56, 80)},
    {"id": "forward_swing", "title": "The Forward Swing", "title_zh": "前挥阶段", "pages": (81, 122)},
    {"id": "training", "title": "Training the Forehand", "title_zh": "正手训练", "pages": (123, 148)},
    {"id": "closing", "title": "Closing Thoughts", "title_zh": "结语", "pages": (149, 152)},
]

# ---------------------------------------------------------------------------
# Font classification helpers
# ---------------------------------------------------------------------------

def _approx(size: float, target: float, tolerance: float = 1.0) -> bool:
    """Check if a font size is approximately equal to a target."""
    return abs(size - target) <= tolerance


def _is_bold(font_name: str) -> bool:
    return "Bold" in font_name and "Ital" not in font_name


def _is_italic(font_name: str) -> bool:
    return "Ital" in font_name or "Italic" in font_name


def _is_bold_italic(font_name: str) -> bool:
    return "Bold" in font_name and ("Ital" in font_name or "Italic" in font_name)


def _is_times(font_name: str) -> bool:
    return "TimesNewRoman" in font_name or "Times" in font_name


def _is_dubai(font_name: str) -> bool:
    return "Dubai" in font_name


def _is_page_number(text: str) -> bool:
    """Check if text is just a standalone page number."""
    return bool(re.fullmatch(r"\s*\d{1,3}\s*", text))


def classify_span(font_name: str, font_size: float, text: str) -> str | None:
    """
    Classify a single text span into a semantic type.

    Returns one of: h1, h2, h3, blockquote, p, strong, figcaption, footnote, None (skip).
    """
    text_stripped = text.strip()
    if not text_stripped:
        return None

    # 42pt TimesNewRoman Bold → h1
    if _approx(font_size, 42, 2) and _is_times(font_name) and _is_bold(font_name):
        return "h1"

    # 28pt TimesNewRoman Bold → h2
    if _approx(font_size, 28, 2) and _is_times(font_name) and _is_bold(font_name):
        return "h2"

    # 19.6pt TimesNewRoman Bold → h3
    if _approx(font_size, 19.6, 1.5) and _is_times(font_name) and _is_bold(font_name):
        return "h3"

    # 14pt TimesNewRoman BoldItalic → blockquote
    if _approx(font_size, 14, 1) and _is_times(font_name) and _is_bold_italic(font_name):
        return "blockquote"

    # 12pt TimesNewRoman (non-bold, non-italic) + standalone number → page number, skip
    if _approx(font_size, 12, 0.5) and _is_times(font_name) and not _is_italic(font_name):
        if _is_page_number(text_stripped):
            return None

    # 12pt TimesNewRoman Italic → figcaption
    if _approx(font_size, 12, 0.5) and _is_times(font_name) and _is_italic(font_name):
        return "figcaption"

    # 7.5pt → footnote
    if _approx(font_size, 7.5, 1.0) and font_size < 9:
        return "footnote"

    # <11pt (but not footnote range) → skip (TOC entries, small annotations)
    if font_size < 11:
        return None

    # 13pt Dubai-Bold → strong (bold body text)
    if _approx(font_size, 13, 1.5) and _is_dubai(font_name) and _is_bold(font_name):
        return "strong"

    # 13pt Dubai-Regular → p (body text)
    if _approx(font_size, 13, 1.5) and _is_dubai(font_name):
        return "p"

    # 12pt TimesNewRoman (non-italic, non-page-number) → treat as body text
    if _approx(font_size, 12, 0.5) and _is_times(font_name):
        return "p"

    # Fallback: anything >= 11pt that we didn't classify → body text
    if font_size >= 11:
        return "p"

    return None


# ---------------------------------------------------------------------------
# Inline HTML wrapping
# ---------------------------------------------------------------------------

def _wrap_inline(text: str, font_name: str, base_type: str) -> str:
    """Wrap text with inline HTML tags if the font differs from the base type."""
    if base_type in ("h1", "h2", "h3", "blockquote", "figcaption", "footnote"):
        return text
    if _is_bold_italic(font_name):
        return f"<strong><em>{text}</em></strong>"
    if _is_bold(font_name):
        return f"<strong>{text}</strong>"
    if _is_italic(font_name):
        return f"<em>{text}</em>"
    return text


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _scan_decoration_xrefs(doc: fitz.Document, max_pages: int = 5) -> set[int]:
    """Pre-scan to find image xrefs that appear on many pages (decorations)."""
    from collections import Counter
    xref_page_count = Counter()
    for pnum in range(doc.page_count):
        seen_on_page = set()
        for img in doc[pnum].get_images(full=True):
            seen_on_page.add(img[0])
        for xref in seen_on_page:
            xref_page_count[xref] += 1
    # Images appearing on more than max_pages pages are decorations
    return {xref for xref, count in xref_page_count.items() if count > max_pages}


def extract_images(doc: fitz.Document, page: fitz.Page, page_num: int,
                   output_dir: str, decoration_xrefs: set[int] = None) -> list[dict]:
    """
    Extract images from a page.

    Saves to {output_dir}/images/fig_{page_num:03d}_{idx:02d}.jpg.
    Filters out decorations (xrefs appearing on many pages) and tiny images.
    Resizes to max width 800px.
    Returns list of image element dicts.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    if decoration_xrefs is None:
        decoration_xrefs = set()

    results = []
    image_list = page.get_images(full=True)

    for idx, img_info in enumerate(image_list):
        xref = img_info[0]

        # Skip decoration images (appear on many pages)
        if xref in decoration_xrefs:
            continue

        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        if not base_image or not base_image.get("image"):
            continue

        width = base_image.get("width", 0)
        height = base_image.get("height", 0)

        if width < 200 or height < 200:
            continue

        # Reconstruct as pixmap for resizing
        pix = fitz.Pixmap(doc, xref)

        # Convert CMYK or other colorspaces to RGB
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        # Resize if wider than 800px
        if pix.width > 800:
            scale = 800 / pix.width
            new_w = 800
            new_h = int(pix.height * scale)
            # Create a new pixmap at the target size via an intermediate step
            import io
            from PIL import Image as PILImage

            img_bytes = pix.tobytes("png")
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            filename = f"fig_{page_num:03d}_{idx:02d}.jpg"
            filepath = os.path.join(images_dir, filename)
            pil_img.save(filepath, "JPEG", quality=85)
        else:
            filename = f"fig_{page_num:03d}_{idx:02d}.jpg"
            filepath = os.path.join(images_dir, filename)
            pix.save(filepath)

        results.append({
            "type": "img",
            "src": f"images/{filename}",
            "caption": "",
            "page": page_num,
            "_y": _get_image_y_position(page, xref),
        })

    return results


def _get_image_y_position(page: fitz.Page, xref: int) -> float:
    """Get the y position of an image on the page for caption association."""
    for img_block in page.get_text("dict")["blocks"]:
        if img_block.get("type") == 1:  # image block
            # Match by checking if this block references our xref
            # Image blocks have a bbox we can use for y-position
            return img_block.get("bbox", (0, 0, 0, 0))[3]  # bottom y
    return 0.0


# ---------------------------------------------------------------------------
# Page text extraction
# ---------------------------------------------------------------------------

def _extract_page_elements(page: fitz.Page, page_num: int) -> list[dict]:
    """
    Extract text elements from a single page.

    Returns a list of element dicts with type, text, and page number.
    Merges consecutive same-type spans on the same line into paragraphs
    with inline HTML tags for bold/italic.
    """
    page_dict = page.get_text("dict")
    raw_spans = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                font_name = span.get("font", "")
                font_size = span.get("size", 0)
                bbox = span.get("bbox", (0, 0, 0, 0))
                span_type = classify_span(font_name, font_size, text)
                if span_type is None:
                    continue
                raw_spans.append({
                    "type": span_type,
                    "text": text,
                    "font": font_name,
                    "size": font_size,
                    "y_top": bbox[1],
                    "y_bottom": bbox[3],
                    "x_left": bbox[0],
                })

    if not raw_spans:
        return []

    # Merge spans into paragraphs.
    # Spans on the same line (similar y position) get merged into one element.
    # Consecutive elements of the same type get merged across lines.
    elements = []
    current = None

    for span in raw_spans:
        # Normalize strong into p (strong is just bold body text)
        element_type = "p" if span["type"] == "strong" else span["type"]

        # Determine if this span continues the current element
        if current is not None:
            same_type = (current["_base_type"] == element_type)
            same_line = abs(span["y_top"] - current["_last_y"]) < 3
            consecutive_line = (
                span["y_top"] > current["_last_y"]
                and span["y_top"] - current["_last_y"] < current["_line_height"] * 2
            )

            if same_type and (same_line or consecutive_line):
                # Merge: add inline markup if needed
                inline_text = _wrap_inline(span["text"], span["font"], element_type)
                if same_line:
                    current["text"] += inline_text
                else:
                    current["text"] += " " + inline_text
                current["_last_y"] = span["y_top"]
                current["_line_height"] = max(
                    current["_line_height"],
                    span["y_bottom"] - span["y_top"]
                )
                continue

            # Flush current element
            elements.append(_finalize_element(current))

        # Start new element
        inline_text = _wrap_inline(span["text"], span["font"], element_type)
        current = {
            "type": element_type,
            "text": inline_text,
            "page": page_num,
            "_base_type": element_type,
            "_last_y": span["y_top"],
            "_line_height": span["y_bottom"] - span["y_top"],
        }

    if current is not None:
        elements.append(_finalize_element(current))

    return elements


def _finalize_element(elem: dict) -> dict:
    """Strip internal fields and clean up text."""
    text = elem["text"].strip()
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return {
        "type": elem["type"],
        "text": text,
        "page": elem["page"],
    }


# ---------------------------------------------------------------------------
# Caption association
# ---------------------------------------------------------------------------

def _associate_captions(elements: list[dict]) -> list[dict]:
    """
    Associate figcaption elements with preceding img elements.

    If a figcaption immediately follows an img on the same page,
    set the img's caption and remove the figcaption element.
    """
    result = []
    i = 0
    while i < len(elements):
        el = elements[i]
        if el["type"] == "img" and i + 1 < len(elements):
            next_el = elements[i + 1]
            if next_el["type"] == "figcaption" and next_el["page"] == el["page"]:
                el["caption"] = next_el["text"]
                result.append(el)
                i += 2
                continue
        result.append(el)
        i += 1
    return result


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_all(pdf_path: str, output_dir: str) -> dict:
    """
    Extract structured content from the FTT book PDF.

    Returns structured data dict and saves to {output_dir}/structured.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # Pre-scan to identify decoration images (appear on many pages)
    print("Scanning for decoration images...")
    decoration_xrefs = _scan_decoration_xrefs(doc, max_pages=5)
    print(f"Found {len(decoration_xrefs)} decoration xrefs to skip")

    structured = {
        "source": os.path.basename(pdf_path),
        "total_pages": len(doc),
        "chapters": [],
    }

    for chapter in CHAPTERS:
        ch_id = chapter["id"]
        start_page, end_page = chapter["pages"]

        ch_data = {
            "id": ch_id,
            "title": chapter["title"],
            "title_zh": chapter.get("title_zh", ""),
            "pages": [start_page, end_page],
            "skip_translate": chapter.get("skip_translate", False),
            "elements": [],
        }

        # Skip content extraction for TOC pages
        if ch_id == "toc":
            structured["chapters"].append(ch_data)
            continue

        all_elements = []

        for page_num in range(start_page, end_page + 1):
            page_idx = page_num - 1  # fitz uses 0-based indexing
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]

            # Extract text elements
            text_elements = _extract_page_elements(page, page_num)

            # Extract images
            img_elements = extract_images(doc, page, page_num, output_dir, decoration_xrefs)

            # Interleave images and text by y-position
            # Images have _y field; text elements don't have explicit y,
            # so we insert images before the first text element that follows them
            if img_elements:
                combined = _interleave_images_and_text(text_elements, img_elements, page)
            else:
                combined = text_elements

            all_elements.extend(combined)

        # Associate captions with images
        all_elements = _associate_captions(all_elements)

        # Clean up internal fields from image elements
        for el in all_elements:
            el.pop("_y", None)

        ch_data["elements"] = all_elements
        structured["chapters"].append(ch_data)

    doc.close()

    # Save structured.json
    output_path = os.path.join(output_dir, "structured.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"Extracted {sum(len(c['elements']) for c in structured['chapters'])} elements "
          f"across {len(structured['chapters'])} chapters → {output_path}")

    return structured


def _interleave_images_and_text(
    text_elements: list[dict],
    img_elements: list[dict],
    page: fitz.Page,
) -> list[dict]:
    """
    Interleave image elements into the text element list based on y-position.

    Images are inserted before the first text element whose page position
    comes after the image's y-position.
    """
    if not img_elements:
        return text_elements

    # We need y-positions for text elements to interleave.
    # Since we lost exact y during merging, insert images at a reasonable spot.
    # Simple approach: insert all images before the text for that page,
    # then rely on caption association to pair them.
    result = []
    img_y_sorted = sorted(img_elements, key=lambda e: e.get("_y", 0))

    # Get approximate y positions for text elements by re-scanning the page
    text_y_map = _estimate_text_y_positions(text_elements, page)

    img_idx = 0
    for i, tel in enumerate(text_elements):
        text_y = text_y_map.get(i, 999999)
        while img_idx < len(img_y_sorted) and img_y_sorted[img_idx].get("_y", 0) <= text_y:
            result.append(img_y_sorted[img_idx])
            img_idx += 1
        result.append(tel)

    # Append any remaining images
    while img_idx < len(img_y_sorted):
        result.append(img_y_sorted[img_idx])
        img_idx += 1

    return result


def _estimate_text_y_positions(elements: list[dict], page: fitz.Page) -> dict[int, float]:
    """
    Estimate y-positions for text elements by matching their text against page content.

    Returns a dict mapping element index to approximate y-position.
    """
    result = {}
    page_dict = page.get_text("dict")

    # Build a list of (y, text_snippet) from page blocks
    block_positions = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
        block_positions.append((bbox[1], block_text.strip()))

    for i, el in enumerate(elements):
        el_text_start = el["text"][:40].replace("<strong>", "").replace("</strong>", "")
        el_text_start = el_text_start.replace("<em>", "").replace("</em>", "").strip()
        best_y = 999999
        for y, bt in block_positions:
            if el_text_start and el_text_start[:20] in bt:
                best_y = y
                break
        result[i] = best_y

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/qsy/Desktop/tennis/"
        "The Fault Tolerant Forehand_ Succeed Under Imperfect Conditions_nodrm.pdf"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else "/Users/qsy/Desktop/tennis/scripts/epub_pipeline/build"

    extract_all(pdf, out)
