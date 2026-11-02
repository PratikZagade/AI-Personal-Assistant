
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq


# =================================
# 1. Flask Application
# =================================

app = Flask(__name__)

# Maximum request size: 64 KB
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# Load local environment variables
load_dotenv()

# Get Groq API Key
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=api_key) if api_key else None

# Model Configuration
MODEL = "openai/gpt-oss-20b"


# =================================
# 2. Home Page
# =================================

@app.route("/")
def hello_world():

    return render_template("index.html")


# =================================
# 3. Ask Anything
# =================================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if client is None:
        app.logger.error("Groq API key is not configured.")

        return jsonify({
            "error": "AI service is not configured. Please contact the administrator."
        }), 500

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": "Act like a helpful personal assistant."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0.7,
            max_completion_tokens=512
        )

        answer = response.choices[0].message.content or ""

        return jsonify({
            "response": answer.strip()
        }), 200

    except Exception:

        app.logger.exception("Ask API Error")

        return jsonify({
            "error": "Unable to generate an answer. Please try again."
        }), 500


# =================================
# 4. Summarize Email
# =================================

@app.route("/summarize", methods=["POST"])
def summarize():

    email_text = request.form.get("email", "").strip()

    if not email_text:
        return jsonify({
            "error": "Please enter an email to summarize."
        }), 400

    if client is None:
        app.logger.error("Groq API key is not configured.")

        return jsonify({
            "error": "AI service is not configured. Please contact the administrator."
        }), 500

    try:

        prompt = (
            "Summarize the following email in 2-3 sentences. "
            "Include the main purpose, important details, "
            "and any required actions.\n\n"
            f"Email:\n{email_text}"
        )

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": "Act like an expert email assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_completion_tokens=512
        )

        summary = response.choices[0].message.content or ""

        return jsonify({
            "response": summary.strip()
        }), 200

    except Exception:

        app.logger.exception("Summarize API Error")

        return jsonify({
            "error": "Unable to summarize the email. Please try again."
        }), 500


# =================================
# 5. Request Error Handling
# =================================

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Request is too large. Please reduce the input size."
    }), 413


# =================================
# 6. Run Flask Application
# =================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
