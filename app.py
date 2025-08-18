from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os

# AI APIs
import openai
import anthropic
import google.generativeai as genai

# File parsing logic
from data_parse.main_parser import parse_file

# Context Manager
from context.manager import ContextManager

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'json', 'csv', 'xlsx', 'xls', 'db', 'sqlite'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize app and context manager
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize context manager
cm = ContextManager()

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

        # Parse file - now returns standardized format
        result = parse_file(filepath)
        
        # Add file info to result
        result["filename"] = filename
        result["filepath"] = filepath
        result["size"] = os.path.getsize(filepath)

        # Store in context manager as file-content
        file_context = {
            "filename": filename,
            "features": result.get("features", []),
            "population": result.get("population", 0),
            "file_type": os.path.splitext(filename)[1].lower(),
            "tables": result.get("tables", [])  # For SQLite files
        }
        cm.upload("file-content", file_context)

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

        # Get current context and append to user message
        context_data = cm.get_all()
        enhanced_message = user_input
        
        if context_data:
            context_str = "Available file context:\n"
            for content in context_data:
                if content["type"] == "file-content":
                    data_info = content["data"]
                    context_str += f"- File: {data_info['filename']} ({data_info['population']} rows)\n"
                    
                    # Handle different feature structures
                    features = data_info['features']
                    if isinstance(features, dict):  # SQLite case
                        context_str += f"  Tables and Features:\n"
                        for table, cols in features.items():
                            context_str += f"    {table}: {', '.join(cols)}\n"
                    else:  # CSV, Excel, JSON case
                        context_str += f"  Features: {', '.join(features)}\n"
                    
                    if data_info.get('tables'):
                        context_str += f"  Table Names: {', '.join(data_info['tables'])}\n"
            
            enhanced_message = f"{context_str}\nUser question: {user_input}"

        # Enhanced error handling for different models
        if model == "openai":
            try:
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_message}],
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
                    messages=[{"role": "user", "content": enhanced_message}]
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
                    enhanced_message,
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

@app.route("/context", methods=["GET"])
def get_context():
    """Get all stored context data"""
    try:
        return jsonify(cm.get_all())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/context/clear", methods=["POST"])
def clear_context():
    """Clear all stored context data"""
    try:
        cm.clear()
        return jsonify({"message": "Context cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    """Delete an uploaded file and remove from context"""
    try:
        if not allowed_file(filename):
            return jsonify({"error": "Invalid file type"}), 400
            
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            
            # Remove from context manager
            current_context = cm.get_all()
            for i, content in enumerate(current_context):
                if (content["type"] == "file-content" and 
                    content["data"]["filename"] == filename):
                    current_context.pop(i)
                    break
            
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
    app.run(debug=True, host='0.0.0.0', port=8000)