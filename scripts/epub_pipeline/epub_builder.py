"""Assemble the final bilingual EPUB from structured content + translations."""

import html
import json
import mimetypes
from pathlib import Path

from ebooklib import epub

from .css import STYLESHEET


def build_epub(
    structured_data: dict,
    translations_dir: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Create the bilingual EPUB file."""
    translations_dir = Path(translations_dir)
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    book = epub.EpubBook()

    # -- Metadata --
    book.set_identifier("ftt-bilingual-v1")
    book.set_title("The Fault Tolerant Forehand 容错型正手")
    book.set_language("en")
    book.add_author("John Kumpf")

    # -- CSS --
    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=STYLESHEET.encode("utf-8"),
    )
    book.add_item(css_item)

    # -- Images --
    _add_images(book, images_dir)

    # -- Chapters --
    chapters = structured_data.get("chapters", [])
    epub_chapters: list[epub.EpubHtml] = []

    for idx, chapter in enumerate(chapters):
        ch_id = chapter.get("id", f"ch{idx}")

        # Skip TOC chapter — EPUB has its own nav
        if ch_id == "toc":
            continue

        filename = f"ch{idx:02d}_{ch_id}.xhtml"

        # Load translations for this chapter
        translations = _load_translations(translations_dir, ch_id)

        # Cover chapter gets special formatting
        if ch_id == "cover":
            chapter_html = _build_cover_html()
        else:
            chapter_html = _build_chapter_html(chapter, translations)

        item = epub.EpubHtml(
            title=chapter.get("title", ch_id),
            file_name=filename,
            lang="en",
        )
        # ebooklib expects body content wrapped in minimal HTML
        wrapped = (
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops">\n'
            "<head></head>\n"
            "<body>\n"
            f"{chapter_html}\n"
            "</body>\n"
            "</html>"
        )
        item.content = wrapped.encode("utf-8")

        book.add_item(item)
        epub_chapters.append(item)

    # -- TOC --
    book.toc = _build_toc(chapters, epub_chapters)

    # -- Navigation --
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # -- Spine --
    book.spine = ["nav"] + epub_chapters

    # -- Write --
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book, {})


def _add_images(book: epub.EpubBook, images_dir: Path) -> None:
    if not images_dir.is_dir():
        return
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        media_type, _ = mimetypes.guess_type(str(img_path))
        if media_type is None or not media_type.startswith("image/"):
            continue
        img_item = epub.EpubImage(
            uid=img_path.stem,
            file_name=f"images/{img_path.name}",
            media_type=media_type,
            content=img_path.read_bytes(),
        )
        book.add_item(img_item)


def _load_translations(translations_dir: Path, chapter_id: str) -> dict:
    """Load translations for a chapter. Returns {element_index_str: chinese_text}."""
    json_path = translations_dir / f"{chapter_id}.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("translations", {})


def _build_chapter_html(chapter: dict, translations: dict) -> str:
    """Generate XHTML body content for one chapter."""
    skip_translate = chapter.get("skip_translate", False)
    parts: list[str] = []

    # Chapter title (h1)
    title = html.escape(chapter.get("title", ""))
    title_zh = chapter.get("title_zh", "")
    if title_zh and not skip_translate:
        parts.append(
            f'<h1>{title}<span class="heading-zh">'
            f"{html.escape(title_zh)}</span></h1>"
        )
    else:
        parts.append(f"<h1>{title}</h1>")

    # Process elements — translations keyed by string index of the element
    elements = chapter.get("elements", [])
    for i, elem in enumerate(elements):
        etype = elem.get("type", "")
        text = elem.get("text", "")
        idx_str = str(i)
        zh = translations.get(idx_str, "")

        if etype in ("h1",):
            # Skip h1 within elements (already handled above)
            continue

        if etype in ("h2", "h3", "h4"):
            if zh and not skip_translate:
                parts.append(
                    f"<{etype}>{_safe(text)}"
                    f'<span class="heading-zh">{html.escape(zh)}</span>'
                    f"</{etype}>"
                )
            else:
                parts.append(f"<{etype}>{_safe(text)}</{etype}>")

        elif etype == "p":
            parts.append(f'<p class="en">{_safe(text)}</p>')
            if zh and not skip_translate:
                parts.append(f'<p class="zh">{_safe(zh)}</p>')

        elif etype == "blockquote":
            parts.append("<blockquote>")
            parts.append(f'  <p class="en"><em>{_safe(text)}</em></p>')
            if zh and not skip_translate:
                parts.append(f'  <p class="zh">{_safe(zh)}</p>')
            parts.append("</blockquote>")

        elif etype == "img":
            src = elem.get("src", "")
            caption = elem.get("caption", "")
            parts.append("<figure>")
            parts.append(f'  <img src="{html.escape(src)}" alt=""/>')
            if caption:
                # Caption translation might be in the next element's slot
                cap_zh = ""
                parts.append(f"  <figcaption>{_safe(caption)}</figcaption>")
            parts.append("</figure>")

        elif etype == "figcaption":
            # Standalone figcaption (not attached to image)
            parts.append(f'<p class="en" style="font-size:0.85em; font-style:italic; color:#666;">{_safe(text)}</p>')
            if zh and not skip_translate:
                parts.append(f'<p class="zh" style="font-size:0.8em; font-style:normal; color:#888;">{_safe(zh)}</p>')

        elif etype == "footnote":
            parts.append(f'<div class="footnote">{_safe(text)}</div>')
            if zh and not skip_translate:
                parts.append(f'<div class="footnote" style="color:#888;">{_safe(zh)}</div>')

    return "\n".join(parts)


def _safe(text: str) -> str:
    """Escape text for XHTML but preserve inline HTML tags like <strong>, <em>."""
    # The text may already contain <strong>/<em> tags from extraction
    # We need to escape & < > but NOT our known inline tags
    # Simple approach: escape everything then unescape known tags
    escaped = html.escape(text)
    escaped = escaped.replace("&lt;strong&gt;", "<strong>")
    escaped = escaped.replace("&lt;/strong&gt;", "</strong>")
    escaped = escaped.replace("&lt;em&gt;", "<em>")
    escaped = escaped.replace("&lt;/em&gt;", "</em>")
    return escaped


def _build_cover_html() -> str:
    return (
        '<div class="cover-title">The Fault Tolerant Forehand</div>\n'
        '<div class="cover-title" style="font-size:1.5em; color:#555;">'
        "容错型正手</div>\n"
        '<div class="cover-subtitle">Succeed Under Imperfect Conditions</div>\n'
        '<div class="cover-subtitle" style="font-style:normal; color:#888;">'
        "在不完美的条件下取得成功</div>\n"
        '<div class="cover-author">John Kumpf</div>\n'
        '<div class="cover-edition">中英双语版 Bilingual Edition · 2026</div>'
    )


def _build_toc(
    chapters: list[dict],
    epub_chapters: list[epub.EpubHtml],
) -> list:
    """Build nested TOC from chapter structure."""
    toc: list = []

    # Build map: chapter id -> epub item
    # epub_chapters skips TOC, so we need to match carefully
    ch_item_map: dict[str, epub.EpubHtml] = {}
    non_toc = [c for c in chapters if c.get("id") != "toc"]
    for ch, item in zip(non_toc, epub_chapters):
        ch_item_map[ch["id"]] = item

    for chapter in chapters:
        ch_id = chapter.get("id", "")
        if ch_id == "toc":
            continue

        item = ch_item_map.get(ch_id)
        if item is None:
            continue

        title = chapter.get("title", ch_id)
        title_zh = chapter.get("title_zh", "")
        display_title = f"{title} {title_zh}" if title_zh else title

        # Collect h2 sub-sections
        children: list[epub.Link] = []
        for elem in chapter.get("elements", []):
            if elem.get("type") == "h2":
                section_text = elem.get("text", "")
                anchor = _make_anchor(section_text)
                children.append(
                    epub.Link(
                        f"{item.file_name}#{anchor}",
                        section_text,
                        f"{item.file_name}_{anchor}",
                    )
                )

        if children:
            section = epub.Section(display_title)
            toc.append((section, children))
        else:
            toc.append(epub.Link(item.file_name, display_title, item.file_name))

    return toc


def _make_anchor(text: str) -> str:
    anchor = text.lower().replace(" ", "-")
    return "".join(c for c in anchor if c.isalnum() or c == "-")
