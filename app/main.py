
from flask import Flask, render_template, request, session
from app.agents.research_agent import research_answer
from app.rag.retriever import create_vector_database
from markdown import markdown as convert_markdown

from pathlib import Path
from werkzeug.utils import secure_filename


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# Secret key is required for Flask session
app.secret_key = "ai-research-assistant-secret-key"


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

DOCUMENTS_DIR.mkdir(exist_ok=True)


# ==========================================
# CURRENT UPLOADED PDF
# ==========================================

current_file_path = None


# ==========================================
# ALLOWED FILE TYPE
# ==========================================

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    global current_file_path

    answer = ""
    message = ""

    # --------------------------------------
    # GET CHAT HISTORY
    # --------------------------------------

    chat_history = session.get(
        "chat_history",
        []
    )

    print(
        "\nCURRENT FILE:",
        current_file_path
    )

    print(
        "CHAT HISTORY LENGTH:",
        len(chat_history)
    )


    # ======================================
    # QUESTION SUBMISSION
    # ======================================

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        print(
            "Question received:",
            question
        )


        # ----------------------------------
        # CHECK PDF
        # ----------------------------------

        if current_file_path is None:

            message = (
                "Please upload a PDF file first."
            )


        elif not question:

            message = (
                "Please enter a question."
            )


        else:

            try:

                print(
                    "\nGenerating answer..."
                )


                # --------------------------
                # SEND QUESTION + HISTORY
                # --------------------------

                answer = research_answer(

                    question,

                    current_file_path,

                    chat_history

                )


                # --------------------------
                # SAVE CONVERSATION
                # --------------------------

                chat_history.append({

                    "question": question,

                    "answer": answer

                })


                # Keep latest 10 conversations
                chat_history = chat_history[-10:]


                session["chat_history"] = chat_history


                # --------------------------
                # CONVERT MARKDOWN
                # --------------------------

                answer = convert_markdown(

                    answer,

                    extensions=["extra"]

                )


                print(
                    "Answer generated successfully."
                )


            except Exception as e:

                print(
                    "ERROR WHILE ANSWERING:",
                    e
                )

                message = (
                    f"Error while answering: {e}"
                )


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


# ==========================================
# PDF UPLOAD
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_file():

    global current_file_path

    print(
        "\n===================================="
    )

    print(
        "       UPLOAD REQUEST RECEIVED"
    )

    print(
        "===================================="
    )


    # --------------------------------------
    # NEW PDF = NEW CONVERSATION
    # --------------------------------------

    session["chat_history"] = []


    # --------------------------------------
    # CHECK FILE FIELD
    # --------------------------------------

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


    # --------------------------------------
    # GET FILE
    # --------------------------------------

    file = request.files["file"]


    print(
        "Filename received:",
        file.filename
    )


    # --------------------------------------
    # CHECK EMPTY FILE
    # --------------------------------------

    if file.filename == "":

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


    # --------------------------------------
    # CHECK PDF EXTENSION
    # --------------------------------------

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


    # --------------------------------------
    # SECURE FILE NAME
    # --------------------------------------

    filename = secure_filename(
        file.filename
    )


    file_path = DOCUMENTS_DIR / filename


    print(
        "Saving PDF to:",
        file_path
    )


    # --------------------------------------
    # SAVE PDF
    # --------------------------------------

    file.save(file_path)


    print(
        "PDF saved successfully."
    )


    # --------------------------------------
    # CREATE VECTOR DATABASE
    # --------------------------------------

    try:

        print(
            "\nProcessing uploaded PDF..."
        )


        create_vector_database(
            str(file_path)
        )


        # -------------------------------
        # SET CURRENT PDF
        # -------------------------------

        current_file_path = str(
            file_path
        )


        print(
            "\nCURRENT FILE SET TO:"
        )

        print(
            current_file_path
        )


        message = (

            f"PDF uploaded successfully: "

            f"{filename}. "

            f"Chat history has been reset. "

            f"You can now ask questions "

            f"about this PDF."

        )


        print(
            message
        )


    except Exception as e:

        print(
            "\nERROR PROCESSING PDF:"
        )

        print(e)


        message = (
            f"Error processing PDF: {e}"
        )


    return render_template(

        "index.html",

        answer="",

        message=message,

        current_file=filename,

        chat_history=[]

    )


# ==========================================
# CLEAR CHAT
# ==========================================

@app.route("/clear-chat", methods=["POST"])
def clear_chat():

    # Clear conversation history
    session["chat_history"] = []

    print(
        "\nCHAT HISTORY CLEARED"
    )

    # Return the same page with the PDF
    # still available
    return render_template(

        "index.html",

        answer="",

        message="Chat history cleared successfully.",

        current_file=(

            Path(current_file_path).name

            if current_file_path

            else None

        ),

        chat_history=[]

    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

