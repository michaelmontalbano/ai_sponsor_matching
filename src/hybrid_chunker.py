from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from semantic_text_splitter import SemanticTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import re

# --------------------------
# CONFIG
# --------------------------
PDF_DIR = Path("../data/papers")
CHUNK_DIR = Path("../data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight + semantic
embedder = SentenceTransformer(MODEL_NAME)
splitter = SemanticTextSplitter.from_huggingface_model(MODEL_NAME)

MIN_SECTION_LENGTH = 80   # characters
MAX_CHUNK_SIZE = 700      # chars per semantic subchunk

# --------------------------
# CLEANUP + HELPERS
# --------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# MAIN CHUNKING FUNCTION
# --------------------------
def hybrid_chunk_pdf(pdf_path: Path):
    """Extract structured content from PDF, semantically chunk long sections."""
    print(f"📘 Processing {pdf_path.name}")

    # Step 1: Parse structured elements
    elements = partition_pdf(filename=pdf_path)
    chunks = []

    for el in elements:
        text = clean_text(el.text or "")
        if len(text) < MIN_SECTION_LENGTH:
            continue

        section = el.metadata.get("section_title", "Unknown")

        # Step 2: If section is long, semantically split
        if len(text) > MAX_CHUNK_SIZE:
            semantic_chunks = splitter.chunks(text, max_characters=MAX_CHUNK_SIZE)
        else:
            semantic_chunks = [text]

        # Step 3: Store chunks
        for ch in semantic_chunks:
            chunk_entry = {
                "section": section,
                "chunk": ch,
                "tokens": len(ch.split()),
                "source_file": pdf_path.name
            }
            chunks.append(chunk_entry)

    # Step 4: Save JSON
    output_path = CHUNK_DIR / f"{pdf_path.stem}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ {len(chunks)} chunks created → {output_path.name}")
    return len(chunks)

# --------------------------
# RUN ALL
# --------------------------
def process_all_pdfs():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    print(f"📂 Found {len(pdfs)} PDFs in {PDF_DIR}")
    results = {}

    for pdf in tqdm(pdfs, desc="Hybrid Semantic Chunking"):
        try:
            num_chunks = hybrid_chunk_pdf(pdf)
            results[pdf.name] = num_chunks
        except Exception as e:
            print(f"⚠️ Error processing {pdf.name}: {e}")

    print("\n🎯 Chunking complete.")
    for k, v in results.items():
        print(f"  - {k}: {v} chunks")

if __name__ == "__main__":
    process_all_pdfs()
