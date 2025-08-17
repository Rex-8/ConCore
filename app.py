from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os

# AI APIs
import openai
import anthropic
import google.generativeai as genai

# File parsing logic
from data_parse.main_parser import parse_file

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'json', 'csv', 'xlsx', 'xls', 'db', 'sqlite'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Save file
        file.save(filepath)

        # Parse file
        result = parse_file(filepath)
        
        # Add file info to result
        result["filename"] = filename
        result["filepath"] = filepath
        result["size"] = os.path.getsize(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat_with_model():
    try:
        data = request.json
        model = data.get("model")
        api_key = data.get("apiKey")
        user_input = data.get("message")

        if not model or not api_key or not user_input:
            return jsonify({"reply": "Missing required parameters."}), 400

        # Enhanced error handling for different models
        if model == "openai":
            try:
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=1000,
                    temperature=0.7
                )
                reply = response.choices[0].message["content"]
            except openai.error.AuthenticationError:
                reply = "Invalid OpenAI API key. Please check your key and try again."
            except openai.error.RateLimitError:
                reply = "OpenAI API rate limit exceeded. Please try again later."
            except Exception as e:
                reply = f"OpenAI Error: {str(e)}"

        elif model == "claude":
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": user_input}]
                )
                reply = response.content[0].text
            except anthropic.AuthenticationError:
                reply = "Invalid Claude API key. Please check your key and try again."
            except anthropic.RateLimitError:
                reply = "Claude API rate limit exceeded. Please try again later."
            except Exception as e:
                reply = f"Claude Error: {str(e)}"

        elif model == "gemini":
            try:
                genai.configure(api_key=api_key)
                model_client = genai.GenerativeModel("gemini-1.5-flash")
                response = model_client.generate_content(
                    user_input,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.7,
                    )
                )
                reply = response.text
            except Exception as e:
                if "API_KEY_INVALID" in str(e):
                    reply = "Invalid Gemini API key. Please check your key and try again."
                elif "QUOTA_EXCEEDED" in str(e):
                    reply = "Gemini API quota exceeded. Please try again later."
                else:
                    reply = f"Gemini Error: {str(e)}"

        else:
            reply = "Invalid model selected. Please choose OpenAI, Claude, or Gemini."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

@app.route("/files", methods=["GET"])
def list_uploaded_files():
    """List all uploaded files"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath) and allowed_file(filename):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(filepath),
                    "uploaded": os.path.getctime(filepath)
                })
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete an uploaded file"""
    try:
        if not allowed_file(filename):
            return jsonify({"error": "Invalid file type"}), 400
            
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"message": f"File {filename} deleted successfully"})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)