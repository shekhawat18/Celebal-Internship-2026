"""
Stage 2: Text Chunking

Splits long documents into overlapping chunks for efficient retrieval.
Each chunk keeps metadata so answers can be traced back to the original
document.
"""

from typing import List, Dict


# ==========================================================
# CHUNK A SINGLE DOCUMENT
# ==========================================================

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Parameters
    ----------
    text : str
        Document text.

    chunk_size : int
        Maximum characters per chunk.

    overlap : int
        Characters shared between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = text.replace("\n", " ").strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ==========================================================
# CHUNK MULTIPLE DOCUMENTS
# ==========================================================

def chunk_documents(
    documents: Dict[str, str],
    chunk_size: int,
    overlap: int,
) -> List[dict]:
    """
    Chunk all documents.

    Returns
    -------
    List[dict]

    Example

    {
        "text": "...",
        "source": "notes.pdf",
        "chunk_id": 0,
        "start_char": 0,
        "end_char": 500,
        "length": 500
    }
    """

    all_chunks = []

    total_characters = 0

    print("\nCreating Chunks")
    print("-" * 60)

    for source, text in documents.items():

        pieces = chunk_text(
            text,
            chunk_size,
            overlap,
        )

        print(
            f"{source:<25}"
            f"{len(pieces):>4} chunks"
        )

        for i, piece in enumerate(pieces):

            start_char = i * (chunk_size - overlap)
            end_char = start_char + len(piece)

            all_chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "chunk_id": i,
                    "start_char": start_char,
                    "end_char": end_char,
                    "length": len(piece),
                }
            )

            total_characters += len(piece)

    print("-" * 60)
    print(f"Total Chunks      : {len(all_chunks)}")
    print(f"Total Characters  : {total_characters:,}")
    print("-" * 60)

    return all_chunks