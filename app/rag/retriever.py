
from pathlib import Path
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.rag.document_loader import load_and_split_document


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

# Make sure the directory exists
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CURRENT ACTIVE COLLECTION
# ==========================================

ACTIVE_COLLECTION = None


# ==========================================
# CACHE EMBEDDING MODEL
# ==========================================

_embeddings = None


def get_embeddings():
    """
    Load the embedding model only once.

    This prevents a new Sentence Transformer
    model from being loaded every time a
    retriever is requested.
    """

    global _embeddings

    if _embeddings is None:

        print("\nLoading embedding model...")

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 8
            }
        )

        print("Embedding model loaded.")

    return _embeddings


# ==========================================
# CREATE UNIQUE ID FROM PDF CONTENT
# ==========================================

def get_file_hash(file_path):

    file_path = Path(file_path)

    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()[:16]


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_vector_database(file_path):

    global ACTIVE_COLLECTION

    file_path = Path(file_path)

    print("\n======================================")
    print("PROCESSING UPLOADED PDF")
    print("======================================")

    print(f"PDF: {file_path.name}")

    # --------------------------------------
    # Check file
    # --------------------------------------

    if not file_path.exists():

        print("PDF file not found.")

        return None

    if file_path.suffix.lower() != ".pdf":

        print("Only PDF files are supported.")

        return None

    # --------------------------------------
    # Create unique collection
    # --------------------------------------

    file_hash = get_file_hash(file_path)

    collection_name = f"pdf_{file_hash}"

    print(f"Collection: {collection_name}")

    # --------------------------------------
    # Load ONLY this PDF
    # --------------------------------------

    print("\nReading uploaded PDF...")

    chunks = load_and_split_document(
        str(file_path)
    )

    if not chunks:

        print(
            "No text could be extracted from this PDF."
        )

        return None

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------
    # Check whether collection already exists
    # --------------------------------------

    print("\nChecking existing vector database...")

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    existing_count = vector_store._collection.count()

    print(
        f"Existing vectors: {existing_count}"
    )

    # --------------------------------------
    # Avoid rebuilding the same PDF
    # --------------------------------------

    if existing_count > 0:

        print(
            "\nPDF already exists in ChromaDB."
        )

        ACTIVE_COLLECTION = collection_name

        print(
            f"Using existing collection: "
            f"{ACTIVE_COLLECTION}"
        )

        return vector_store

    # --------------------------------------
    # Create embeddings
    # --------------------------------------

    print("\nCreating embeddings...")

    # Process documents in smaller batches
    batch_size = 16

    for start in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]

        print(
            f"Processing chunks "
            f"{start + 1}-"
            f"{min(start + batch_size, len(chunks))}"
            f" of {len(chunks)}"
        )

        vector_store.add_documents(
            batch
        )

    # --------------------------------------
    # Set active collection
    # --------------------------------------

    ACTIVE_COLLECTION = collection_name

    print("\n======================================")
    print("ACTIVE PDF UPDATED")
    print("======================================")

    print(f"PDF: {file_path.name}")

    print(
        f"Collection: {ACTIVE_COLLECTION}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "======================================"
    )

    return vector_store


# ==========================================
# GET RETRIEVER
# ==========================================

def get_retriever(file_path=None):

    global ACTIVE_COLLECTION

    # --------------------------------------
    # Make sure a PDF has been uploaded
    # --------------------------------------

    if ACTIVE_COLLECTION is None:

        raise ValueError(
            "No PDF has been uploaded yet."
        )

    print("\n======================================")
    print("LOADING ACTIVE PDF RETRIEVER")
    print("======================================")

    print(
        f"Collection: {ACTIVE_COLLECTION}"
    )

    print(
        "======================================"
    )

    embeddings = get_embeddings()

    vector_store = Chroma(

        collection_name=ACTIVE_COLLECTION,

        persist_directory=str(CHROMA_DIR),

        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(

        search_kwargs={
            "k": 5
        }

    )

    return retriever


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    file_path = input(
        "\nEnter PDF file path: "
    ).strip()

    vector_store = create_vector_database(
        file_path
    )

    if vector_store:

        retriever = get_retriever()

        question = input(
            "\nAsk a question about the PDF: "
        ).strip()

        results = retriever.invoke(
            question
        )

        print(
            "\n========== RETRIEVED RESULTS ==========\n"
        )

        if not results:

            print(
                "No relevant information found."
            )

        else:

            for i, doc in enumerate(
                results,
                start=1
            ):

                print(
                    f"--- Result {i} ---"
                )

                print(
                    doc.page_content[:1000]
                )

                print(
                    "\nSOURCE:",
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                )

                print()

        print(
            "========================================"
        )
