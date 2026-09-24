from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from sklearn.feature_extraction.text import HashingVectorizer

from app.rag.document_loader import load_and_split_document


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CURRENT ACTIVE COLLECTION
# ==========================================

ACTIVE_COLLECTION = None


# ==========================================
# LIGHTWEIGHT LOCAL EMBEDDINGS
# ==========================================

class LocalEmbeddings(Embeddings):

    def __init__(self):

        # HashingVectorizer does not need a trained model.
        # It is lightweight and uses fixed-size vectors.

        self.vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2)
        )

    def embed_documents(self, texts):

        if not texts:
            return []

        vectors = self.vectorizer.transform(texts)

        return vectors.toarray().tolist()

    def embed_query(self, text):

        vector = self.vectorizer.transform([text])

        return vector.toarray()[0].tolist()


# ==========================================
# SINGLE EMBEDDING INSTANCE
# ==========================================

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print("\nLoading lightweight local embeddings...")

        _embeddings = LocalEmbeddings()

        print("Local embeddings ready.")

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
    # Validate file
    # --------------------------------------

    if not file_path.exists():

        print("PDF file not found.")

        return None

    if file_path.suffix.lower() != ".pdf":

        print("Only PDF files are supported.")

        return None

    # --------------------------------------
    # Unique collection
    # --------------------------------------

    file_hash = get_file_hash(file_path)

    # IMPORTANT:
    # local_v1 prevents mixing old Gemini
    # embeddings with the new local embeddings.

    collection_name = f"pdf_local_v1_{file_hash}"

    print(f"Collection: {collection_name}")

    # --------------------------------------
    # Load PDF
    # --------------------------------------

    print("\nReading uploaded PDF...")

    chunks = load_and_split_document(
        str(file_path)
    )

    if not chunks:

        print(
            "\nNo readable text was found in this PDF."
        )

        print(
            "OCR will be required for scanned/image PDFs."
        )

        return None

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------
    # Local embeddings
    # --------------------------------------

    embeddings = get_embeddings()

    # --------------------------------------
    # Existing Chroma collection
    # --------------------------------------

    print(
        "\nChecking existing vector database..."
    )

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
    # Already processed
    # --------------------------------------

    if existing_count > 0:

        print(
            "\nPDF already exists in ChromaDB."
        )

        ACTIVE_COLLECTION = collection_name

        print(
            f"Active collection: {ACTIVE_COLLECTION}"
        )

        return vector_store

    # --------------------------------------
    # Create embeddings
    # --------------------------------------

    print(
        "\nCreating local embeddings..."
    )

    batch_size = 32

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

        vector_store.add_documents(batch)

    # --------------------------------------
    # Activate collection
    # --------------------------------------

    ACTIVE_COLLECTION = collection_name

    print("\n======================================")
    print("ACTIVE PDF UPDATED")
    print("======================================")

    print(
        f"PDF: {file_path.name}"
    )

    print(
        f"Collection: {ACTIVE_COLLECTION}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "Embedding method: Local HashingVectorizer"
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