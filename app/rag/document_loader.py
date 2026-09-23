from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


def load_and_split_document(file_path, chunk_size=800, chunk_overlap=100):
    """
    Load the uploaded PDF and split it into smaller chunks.

    Uses lightweight pypdf instead of PyPDFLoader and a simple
    Python-based splitter to reduce memory usage on Render.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    print(f"Loading uploaded file: {file_path.name}")

    reader = PdfReader(str(file_path))

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        # Clean unnecessary whitespace
        text = " ".join(text.split())

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": str(file_path),
                            "page": page_number
                        }
                    )
                )

            if end >= text_length:
                break

            start = max(0, end - chunk_overlap)

    print(f"Loaded {len(reader.pages)} pages from {file_path.name}")
    print(f"Created {len(chunks)} chunks.")

    return chunks