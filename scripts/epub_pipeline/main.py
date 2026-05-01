"""Bilingual EPUB Pipeline — orchestrator (PDF → 中英对照 EPUB)."""
import argparse, os, sys, json

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.epub_pipeline.pdf_extractor import extract_all
from scripts.epub_pipeline.translator import translate_all
from scripts.epub_pipeline.epub_builder import build_epub

# Default = FTT (preserves backward compatibility with `python -m scripts.epub_pipeline.main`)
DEFAULT_PDF = os.path.join(
    PROJECT_ROOT,
    "The Fault Tolerant Forehand_ Succeed Under Imperfect Conditions_nodrm.pdf",
)
DEFAULT_BUILD_DIR = os.path.join(PROJECT_ROOT, "output", "epub_build")
DEFAULT_EPUB = os.path.join(PROJECT_ROOT, "output", "The_Fault_Tolerant_Forehand_Bilingual.epub")


def step_extract(pdf_path, build_dir):
    print("=" * 60)
    print(f"STEP 1: Extract  {os.path.basename(pdf_path)}")
    print("=" * 60)
    data = extract_all(pdf_path, build_dir)
    total_elements = sum(len(ch["elements"]) for ch in data["chapters"])
    total_images = sum(1 for ch in data["chapters"] for e in ch["elements"] if e["type"] == "img")
    print(f"\nExtraction complete: {len(data['chapters'])} chapters, {total_elements} elements, {total_images} images")
    return data


def step_translate(build_dir, structured_data=None, from_chapter=None, force=False):
    print("=" * 60)
    print("STEP 2: Translate with Gemini")
    print("=" * 60)
    if structured_data is None:
        with open(os.path.join(build_dir, "structured.json"), encoding="utf-8") as f:
            structured_data = json.load(f)
    translate_all(structured_data, build_dir, force=force, from_chapter=from_chapter)
    print("\nTranslation complete.")


def step_build(build_dir, epub_path, structured_data=None):
    print("=" * 60)
    print(f"STEP 3: Build EPUB → {os.path.basename(epub_path)}")
    print("=" * 60)
    if structured_data is None:
        with open(os.path.join(build_dir, "structured.json"), encoding="utf-8") as f:
            structured_data = json.load(f)
    build_epub(
        structured_data,
        os.path.join(build_dir, "translations"),
        os.path.join(build_dir, "images"),
        epub_path,
    )
    size_mb = os.path.getsize(epub_path) / (1024 * 1024)
    print(f"\nEPUB saved: {epub_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Bilingual EPUB Pipeline")
    parser.add_argument("--step", choices=["extract", "translate", "build", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Source PDF path")
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR,
                        help="Working directory for extracted JSON / images / translations")
    parser.add_argument("--epub-out", default=DEFAULT_EPUB,
                        help="Output EPUB filepath")
    parser.add_argument("--from-chapter", help="Resume translation from this chapter ID")
    parser.add_argument("--force", action="store_true", help="Force re-translate cached chapters")
    args = parser.parse_args()

    os.makedirs(args.build_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.epub_out) or ".", exist_ok=True)

    if args.step == "extract":
        step_extract(args.pdf, args.build_dir)
    elif args.step == "translate":
        step_translate(args.build_dir, from_chapter=args.from_chapter, force=args.force)
    elif args.step == "build":
        step_build(args.build_dir, args.epub_out)
    else:  # all
        data = step_extract(args.pdf, args.build_dir)
        step_translate(args.build_dir, data)
        step_build(args.build_dir, args.epub_out, data)


if __name__ == "__main__":
    main()
