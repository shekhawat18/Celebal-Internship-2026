"""
Stage 1: Document Ingestion

Loads PDF and TXT documents from a directory and extracts raw text.
Each successfully loaded document is stored in a dictionary:

{
    "filename.pdf": "...text..."
}
"""

import os
import time
from pypdf import PdfReader


# ==========================================================
# LOAD PDF
# ==========================================================

def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    reader = PdfReader(file_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


# ==========================================================
# LOAD TXT
# ==========================================================

def load_txt(file_path: str) -> str:
    """
    Read a UTF-8 text file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        return f.read()


# ==========================================================
# LOAD ALL DOCUMENTS
# ==========================================================

def load_documents(data_dir: str) -> dict:

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Directory not found: {data_dir}"
        )

    print("\nLoading Documents")
    print("-" * 60)

    start = time.time()

    documents = {}

    total_characters = 0

    supported = (".pdf", ".txt")

    for filename in sorted(os.listdir(data_dir)):

        path = os.path.join(data_dir, filename)

        if not filename.lower().endswith(supported):
            continue

        try:

            if filename.lower().endswith(".pdf"):
                text = load_pdf(path)

            else:
                text = load_txt(path)

            text = text.strip()

            if len(text) == 0:
                print(f"⚠ Skipped empty file: {filename}")
                continue

            documents[filename] = text

            characters = len(text)

            total_characters += characters

            print(
                f"✓ {filename:<30}"
                f"{characters:>8,} chars"
            )

        except Exception as e:

            print(f"✗ Failed to load {filename}")

            print(f"  Reason: {e}")

    if len(documents) == 0:
        raise ValueError(
            "No readable PDF or TXT files were found."
        )

    elapsed = time.time() - start

    print("-" * 60)
    print(f"Documents Loaded : {len(documents)}")
    print(f"Characters       : {total_characters:,}")
    print(f"Load Time        : {elapsed:.2f} sec")
    print("-" * 60)

    return documents