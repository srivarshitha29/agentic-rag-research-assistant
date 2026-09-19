from dotenv import load_dotenv
import os
from pathlib import Path

from google import genai

from app.rag.retriever import get_retriever


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=API_KEY
)


# ==========================================
# RESEARCH ANSWER
# ==========================================

def research_answer(
    question,
    file_path,
    chat_history=None
):

    """
    Answer the user's question using the
    currently uploaded PDF.
    """

    print("\n======================================")
    print("RESEARCH AGENT")
    print("======================================")

    print(f"Question: {question}")
    print(f"PDF: {Path(file_path).name}")


    # ======================================
    # INITIALIZE HISTORY
    # ======================================

    if chat_history is None:
        chat_history = []


    # ======================================
    # GET RETRIEVER
    # ======================================

    try:

        retriever = get_retriever(
            file_path
        )

    except Exception as e:

        print(f"Retriever error: {e}")

        return (
            f"Error loading PDF retriever: {e}"
        )


    # ======================================
    # RETRIEVE DOCUMENT INFORMATION
    # ======================================

    try:

        docs = retriever.invoke(
            question
        )

    except Exception as e:

        print(f"Retrieval error: {e}")

        return (
            f"Error retrieving information: {e}"
        )


    # ======================================
    # CHECK RESULTS
    # ======================================

    if not docs:

        return (
            "The uploaded document does not contain "
            "enough information to answer this question."
        )


    # ======================================
    # CREATE DOCUMENT CONTEXT
    # ======================================

    context_parts = []

    for doc in docs:

        text = doc.page_content.strip()

        if text:

            context_parts.append(text)


    context = "\n\n".join(
        context_parts
    )


    if not context:

        return (
            "The uploaded document does not contain "
            "enough information to answer this question."
        )


    # ======================================
    # CREATE CONVERSATION CONTEXT
    # ======================================

    conversation_context = ""

    if chat_history:

        conversation_parts = []

        for item in chat_history:

            previous_question = item.get(
                "question",
                ""
            )

            previous_answer = item.get(
                "answer",
                ""
            )

            conversation_parts.append(
                f"Previous User Question:\n"
                f"{previous_question}\n\n"
                f"Previous Assistant Answer:\n"
                f"{previous_answer}"
            )

        conversation_context = "\n\n".join(
            conversation_parts
        )

    else:

        conversation_context = (
            "No previous conversation."
        )


    # ======================================
    # COLLECT SOURCES
    # ======================================

    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            file_path
        )

        source = os.path.basename(
            source
        )

        page = doc.metadata.get(
            "page",
            None
        )

        if isinstance(page, int):

            page = page + 1


        if page is not None:

            source_info = (
                f"{source} - Page {page}"
            )

        else:

            source_info = source


        if source_info not in sources:

            sources.append(
                source_info
            )


    # ======================================
    # CREATE PROMPT
    # ======================================

    prompt = f"""
You are an AI Research Assistant.

Answer the user's question using ONLY the
provided document context.

IMPORTANT FORMATTING RULES:

1. Give a short direct answer first.
2. Use headings when appropriate.
3. Always present important information
   in numbered points or bullet points.
4. Do NOT write the entire answer as a paragraph.
5. For types, categories, features,
   advantages, disadvantages, or steps,
   use separate numbered points.
6. Keep each point short and clear.
7. Use **bold** for important terms.
8. Include examples if they are present
   in the document.
9. Do not invent information.
10. If the document does not contain enough
    information, clearly say so.
11. Make the answer easy for students to
    understand and remember.
12. Use Markdown formatting.

Use this structure when appropriate:

### Direct Answer

Give the answer in 1–2 sentences.

### Key Points

1. **Point 1:** Explanation.
2. **Point 2:** Explanation.
3. **Point 3:** Explanation.

### Example

Give an example if available.

### Summary

Give a short summary.

Document Context:
{context}

Previous Conversation:
{conversation_context}

User Question:
{question}

Now provide the answer in clear,
point-wise Markdown format.
"""


    # ======================================
    # GEMINI MODEL
    # ======================================

    model_name = "gemini-3.6-flash"


    # ======================================
    # GENERATE ANSWER
    # ======================================

    try:

        print(
            f"\nTrying Gemini model: {model_name}"
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        print(
            f"Successfully used: {model_name}"
        )


    except Exception as e:

        print(
            f"{model_name} failed:"
        )

        print(e)


        if (
            "429" in str(e)
            or "RESOURCE_EXHAUSTED" in str(e)
        ):

            return (
                "⚠️ **Gemini API quota reached.**\n\n"
                "Your PDF processing, embeddings, "
                "vector database, and RAG retrieval "
                "are working correctly.\n\n"
                "Only Gemini answer generation is "
                "currently unavailable because the "
                "API quota has been reached."
            )


        return (
            f"Error while generating answer: {e}"
        )


    # ======================================
    # GET ANSWER
    # ======================================

    answer = response.text


    # ======================================
    # ADD SOURCES
    # ======================================

    if sources:

        answer += "\n\n---\n"

        answer += (
            "### 📚 Sources from Uploaded PDF\n\n"
        )


        for i, source in enumerate(
            sources,
            start=1
        ):

            answer += (
                f"{i}. 📄 {source}\n"
            )


    print(
        "\nAnswer generated successfully."
    )


    return answer


# ==========================================
# TERMINAL TEST
# ==========================================

if __name__ == "__main__":

    file_path = input(
        "\nEnter PDF file path: "
    ).strip()


    question = input(
        "\nAsk your question: "
    ).strip()


    answer = research_answer(
        question,
        file_path,
        []
    )


    print(
        "\n======================================"
    )

    print(
        "AI RESEARCH ASSISTANT"
    )

    print(
        "======================================"
    )

    print(
        "\nAnswer:\n"
    )

    print(answer)