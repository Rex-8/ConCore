from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# AI APIs
import openai
import anthropic
import google.generativeai as genai

# File parsing logic (make sure this path is correct)
from data_parse.main_parser import parse_file

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = parse_file(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"})


@app.route("/chat", methods=["POST"])
def chat_with_model():
    data = request.json
    model = data.get("model")
    api_key = data.get("apiKey")
    user_input = data.get("message")

    if not model or not api_key or not user_input:
        return jsonify({"reply": "Missing required parameters."})

    try:
        if model == "openai":
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_input}]
            )
            reply = response.choices[0].message["content"]

        elif model == "claude":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": user_input}]
            )
            reply = response.content[0].text

        elif model == "gemini":
            genai.configure(api_key=api_key)
            model_client = genai.GenerativeModel("gemini-2.5-flash")
            response = model_client.generate_content(user_input)
            reply = response.text

        else:
            reply = "Invalid model selected."

    except Exception as e:
        reply = f"Error: {str(e)}"

    return jsonify({"reply": reply})


# Run server
if __name__ == "__main__":
    app.run(debug=True)
