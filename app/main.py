import os
from pathlib import Path

from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from markdown import markdown as convert_markdown

from app.agents.research_agent import research_answer
from app.rag.retriever import create_vector_database


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Secret key
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "ai-research-assistant-secret-key"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CURRENT PDF
# ============================================================

current_file_path = None


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    """Check whether the uploaded file is a PDF."""

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    global current_file_path

    answer = ""
    message = ""

    # --------------------------------------------------------
    # GET CHAT HISTORY
    # --------------------------------------------------------

    chat_history = session.get(
        "chat_history",
        []
    )

    print("\n========================================")
    print("HOME REQUEST")
    print("========================================")

    print(
        "CURRENT FILE:",
        current_file_path
    )

    print(
        "CHAT HISTORY LENGTH:",
        len(chat_history)
    )

    # ========================================================
    # QUESTION SUBMISSION
    # ========================================================

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        print(
            "QUESTION:",
            question
        )

        # ----------------------------------------------------
        # CHECK PDF
        # ----------------------------------------------------

        if current_file_path is None:

            message = (
                "Please upload a PDF file first."
            )

        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        elif not question:

            message = (
                "Please enter a question."
            )

        else:

            try:

                print(
                    "\nGenerating answer..."
                )

                # ------------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------------

                answer = research_answer(
                    question,
                    current_file_path,
                    chat_history
                )

                # ------------------------------------------------
                # SAVE CHAT HISTORY
                # ------------------------------------------------

                chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

                # Keep only latest 10 conversations
                chat_history = chat_history[-10:]

                session["chat_history"] = chat_history

                # Make session data persistent
                session.modified = True

                # ------------------------------------------------
                # CONVERT MARKDOWN TO HTML
                # ------------------------------------------------

                answer = convert_markdown(
                    answer,
                    extensions=["extra"]
                )

                print(
                    "Answer generated successfully."
                )

            except Exception as e:

                print(
                    "\nERROR WHILE ANSWERING:"
                )

                print(
                    repr(e)
                )

                message = (
                    "Error while answering: "
                    + str(e)
                )

    # ========================================================
    # RENDER PAGE
    # ========================================================

    return render_template(
        "index.html",

        answer=answer,

        message=message,

        current_file=(
            Path(current_file_path).name
            if current_file_path
            else None
        ),

        chat_history=chat_history
    )


# ============================================================
# PDF UPLOAD
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    global current_file_path

    print("\n========================================")
    print("PDF UPLOAD REQUEST")
    print("========================================")

    # --------------------------------------------------------
    # RESET CHAT FOR NEW PDF
    # --------------------------------------------------------

    session["chat_history"] = []

    session.modified = True

    # --------------------------------------------------------
    # CHECK FILE FIELD
    # --------------------------------------------------------

    if "file" not in request.files:

        message = (
            "No file was received by Flask."
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=None,

            chat_history=[]
        )

    # --------------------------------------------------------
    # GET FILE
    # --------------------------------------------------------

    file = request.files["file"]

    print(
        "Filename received:",
        file.filename
    )

    # --------------------------------------------------------
    # CHECK EMPTY FILE
    # --------------------------------------------------------

    if not file.filename:

        message = (
            "Please select a PDF file."
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=None,

            chat_history=[]
        )

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        message = (
            "Only PDF files are allowed."
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=None,

            chat_history=[]
        )

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    if not filename:

        message = (
            "Invalid file name."
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=None,

            chat_history=[]
        )

    # --------------------------------------------------------
    # CREATE FILE PATH
    # --------------------------------------------------------

    file_path = DOCUMENTS_DIR / filename

    print(
        "Saving PDF to:",
        file_path
    )

    try:

        # ----------------------------------------------------
        # SAVE PDF
        # ----------------------------------------------------

        file.save(file_path)

        print(
            "PDF saved successfully."
        )

        # ----------------------------------------------------
        # CREATE VECTOR DATABASE
        # ----------------------------------------------------

        print(
            "\nProcessing uploaded PDF..."
        )

        create_vector_database(
            str(file_path)
        )

        print(
            "Vector database created successfully."
        )

        # ----------------------------------------------------
        # SET CURRENT PDF
        # ----------------------------------------------------

        current_file_path = str(
            file_path
        )

        print(
            "\nCURRENT FILE SET TO:"
        )

        print(
            current_file_path
        )

        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        message = (
            f"PDF uploaded successfully: {filename}. "
            "Chat history has been reset. "
            "You can now ask questions about this PDF."
        )

        print(
            message
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=filename,

            chat_history=[]
        )

    except Exception as e:

        print(
            "\nERROR PROCESSING PDF:"
        )

        print(
            repr(e)
        )

        # ----------------------------------------------------
        # REMOVE PARTIALLY SAVED FILE
        # ----------------------------------------------------

        try:

            if file_path.exists():

                file_path.unlink()

        except Exception:

            pass

        current_file_path = None

        message = (
            "Error processing PDF: "
            + str(e)
        )

        return render_template(
            "index.html",

            answer="",

            message=message,

            current_file=None,

            chat_history=[]
        )


# ============================================================
# CLEAR CHAT
# ============================================================

@app.route("/clear-chat", methods=["POST"])
def clear_chat():

    # --------------------------------------------------------
    # CLEAR SESSION CHAT
    # --------------------------------------------------------

    session["chat_history"] = []

    session.modified = True

    print(
        "\nCHAT HISTORY CLEARED"
    )

    # --------------------------------------------------------
    # KEEP CURRENT PDF
    # --------------------------------------------------------

    current_file = (
        Path(current_file_path).name
        if current_file_path
        else None
    )

    return render_template(
        "index.html",

        answer="",

        message="Chat history cleared successfully.",

        current_file=current_file,

        chat_history=[]
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return {
        "status": "healthy",
        "application": "AI Research Assistant"
    }, 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "index.html",

        answer="",

        message="Page not found.",

        current_file=(
            Path(current_file_path).name
            if current_file_path
            else None
        ),

        chat_history=session.get(
            "chat_history",
            []
        )
    ), 404


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "\nINTERNAL SERVER ERROR:"
    )

    print(
        repr(error)
    )

    return render_template(
        "index.html",

        answer="",

        message="An internal server error occurred.",

        current_file=(
            Path(current_file_path).name
            if current_file_path
            else None
        ),

        chat_history=session.get(
            "chat_history",
            []
        )
    ), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )