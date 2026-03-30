"""FTT Bilingual EPUB Pipeline — orchestrator."""
import argparse, os, sys, json

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.epub_pipeline.pdf_extractor import extract_all
from scripts.epub_pipeline.translator import translate_all
from scripts.epub_pipeline.epub_builder import build_epub

PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "The Fault Tolerant Forehand_ Succeed Under Imperfect Conditions_nodrm.pdf",
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "epub_build")
EPUB_OUTPUT = os.path.join(PROJECT_ROOT, "output", "The_Fault_Tolerant_Forehand_Bilingual.epub")


def step_extract():
    print("=" * 60)
    print("STEP 1: Extracting text and images from PDF")
    print("=" * 60)
    data = extract_all(PDF_PATH, OUTPUT_DIR)
    total_elements = sum(len(ch["elements"]) for ch in data["chapters"])
    total_images = sum(1 for ch in data["chapters"] for e in ch["elements"] if e["type"] == "img")
    print(f"\nExtraction complete: {len(data['chapters'])} chapters, {total_elements} elements, {total_images} images")
    return data


def step_translate(structured_data=None, from_chapter=None, force=False):
    print("=" * 60)
    print("STEP 2: Translating with Gemini")
    print("=" * 60)
    if structured_data is None:
        json_path = os.path.join(OUTPUT_DIR, "structured.json")
        with open(json_path, encoding="utf-8") as f:
            structured_data = json.load(f)
    translate_all(structured_data, OUTPUT_DIR, force=force, from_chapter=from_chapter)
    print("\nTranslation complete.")


def step_build(structured_data=None):
    print("=" * 60)
    print("STEP 3: Building EPUB")
    print("=" * 60)
    if structured_data is None:
        json_path = os.path.join(OUTPUT_DIR, "structured.json")
        with open(json_path, encoding="utf-8") as f:
            structured_data = json.load(f)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    translations_dir = os.path.join(OUTPUT_DIR, "translations")
    build_epub(structured_data, translations_dir, images_dir, EPUB_OUTPUT)
    size_mb = os.path.getsize(EPUB_OUTPUT) / (1024 * 1024)
    print(f"\nEPUB saved: {EPUB_OUTPUT} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="FTT Bilingual EPUB Pipeline")
    parser.add_argument("--step", choices=["extract", "translate", "build", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--from-chapter", help="Resume translation from this chapter ID")
    parser.add_argument("--force", action="store_true", help="Force re-translate cached chapters")
    args = parser.parse_args()

    if args.step == "extract":
        step_extract()
    elif args.step == "translate":
        step_translate(from_chapter=args.from_chapter, force=args.force)
    elif args.step == "build":
        step_build()
    else:  # all
        data = step_extract()
        step_translate(data)
        step_build(data)


if __name__ == "__main__":
    main()
