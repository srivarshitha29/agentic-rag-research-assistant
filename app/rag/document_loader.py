from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path):
    """
    Load only the PDF file provided by the user.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    print(f"Loading uploaded file: {file_path.name}")

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from {file_path.name}")

    return documents


def split_documents(documents):
    """
    Split the uploaded document into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    print(f"Created {len(chunks)} chunks.")

    return chunks


def load_and_split_document(file_path):
    """
    Load and split ONLY the uploaded PDF.
    """

    documents = load_document(file_path)

    chunks = split_documents(documents)

    return chunks