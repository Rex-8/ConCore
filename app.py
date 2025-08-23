from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json

# AI APIs
import openai
import anthropic
import google.generativeai as genai

# File parsing logic
from data_parse.main_parser import parse_file

# Context Manager
from context.manager import ContextManager

# Data Access Tools
from data_access_tools import DataAccessTools, LLMToolIntegration

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'json', 'csv', 'xlsx', 'xls', 'db', 'sqlite'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize app and components
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize managers
cm = ContextManager()
data_tools = DataAccessTools(UPLOAD_FOLDER)

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
    """Upload and parse file - stores only metadata in context"""
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

        # Parse file - returns standardized metadata format
        result = parse_file(filepath)
        
        # Add file info to result
        result["filename"] = filename
        result["filepath"] = filepath
        result["size"] = os.path.getsize(filepath)

        # Store ONLY metadata in context manager
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
    """Enhanced chat endpoint with tool calling support"""
    try:
        data = request.json
        model = data.get("model")
        api_key = data.get("apiKey")
        user_input = data.get("message")

        if not model or not api_key or not user_input:
            return jsonify({"reply": "Missing required parameters."}), 400

        # Get current context (metadata only)
        context_data = cm.get_all()
        enhanced_message = user_input
        
        if context_data:
            context_str = "Available file context (metadata only - use tools to access actual data):\n"
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
            
            context_str += "\nTo access actual data, use the available tools:\n"
            context_str += "- get_data_sample: Get sample rows\n"
            context_str += "- get_column_data: Get specific columns\n" 
            context_str += "- get_statistics: Get descriptive statistics\n"
            context_str += "- search_data: Search for specific values\n"
            context_str += "- query_sqlite: Run SQL queries on database files\n"
            
            enhanced_message = f"{context_str}\nUser question: {user_input}"

        # Get tool definitions
        tools_definition = data_tools.get_tools_definition()

        # Route to appropriate model with tool support
        if model == "openai":
            try:
                client = openai.OpenAI(api_key=api_key)
                
                # Format tools for OpenAI
                tools = LLMToolIntegration.format_tools_for_openai(tools_definition)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_message}],
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1000,
                    temperature=0.7
                )
                
                message = response.choices[0].message
                
                # Handle tool calls
                if message.tool_calls:
                    tool_call_messages = []
                    
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        # Execute tool
                        result = data_tools.execute_tool(tool_name, tool_args)
                        
                        # Add tool call result message
                        tool_call_messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result)
                        })
                    
                    # Get follow-up response with tool results
                    messages = [
                        {"role": "user", "content": enhanced_message},
                        message,  # Assistant message with tool calls
                    ] + tool_call_messages
                    
                    final_response = client.chat.completions.create(
                        model="gpt-4o-mini", 
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    reply = final_response.choices[0].message.content
                else:
                    reply = message.content
                    
            except Exception as e:
                error_str = str(e).lower()
                if "incorrect_api_key" in error_str or "invalid" in error_str:
                    reply = "Invalid OpenAI API key. Please check your key and try again."
                elif "rate_limit" in error_str:
                    reply = "OpenAI API rate limit exceeded. Please try again later."
                else:
                    reply = f"OpenAI Error: {str(e)}"

        elif model == "claude":
            try:
                client = anthropic.Anthropic(api_key=api_key)
                
                # Format tools for Claude
                tools = LLMToolIntegration.format_tools_for_claude(tools_definition)
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.7,
                    tools=tools,
                    messages=[{"role": "user", "content": enhanced_message}]
                )
                
                # Handle tool calls
                if response.stop_reason == "tool_use":
                    tool_call_messages = []
                    assistant_content = []
                    
                    # Collect assistant content and tool calls
                    for content_block in response.content:
                        if content_block.type == "text":
                            assistant_content.append(content_block.text)
                        elif content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_args = content_block.input
                            
                            # Execute tool
                            result = data_tools.execute_tool(tool_name, tool_args)
                            
                            # Add tool result message
                            tool_call_messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content_block.id,
                                        "content": json.dumps(result)
                                    }
                                ]
                            })
                    
                    # Get follow-up response with tool results
                    messages = [
                        {"role": "user", "content": enhanced_message},
                        {"role": "assistant", "content": response.content}
                    ] + tool_call_messages
                    
                    final_response = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        temperature=0.7,
                        messages=messages
                    )
                    
                    reply = final_response.content[0].text
                else:
                    reply = response.content[0].text
                    
            except Exception as e:
                error_str = str(e).lower()
                if "authentication" in error_str or "invalid" in error_str:
                    reply = "Invalid Claude API key. Please check your key and try again."
                elif "rate_limit" in error_str:
                    reply = "Claude API rate limit exceeded. Please try again later."
                else:
                    reply = f"Claude Error: {str(e)}"

        elif model == "gemini":
            try:
                genai.configure(api_key=api_key)
                
                # For Gemini, we'll implement a simpler approach without native tool calling
                # since Gemini's tool calling API is different
                model_client = genai.GenerativeModel("gemini-1.5-flash")
                
                # Add tool instructions to the prompt
                tool_instructions = "\n\nIf you need to access actual data from the files, please ask me to use one of these tools:\n"
                for tool in tools_definition:
                    func = tool["function"]
                    tool_instructions += f"- {func['name']}: {func['description']}\n"
                
                enhanced_message += tool_instructions
                
                response = model_client.generate_content(
                    enhanced_message,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.7,
                    )
                )
                reply = response.text
                
            except Exception as e:
                error_str = str(e)
                if "API_KEY_INVALID" in error_str:
                    reply = "Invalid Gemini API key. Please check your key and try again."
                elif "QUOTA_EXCEEDED" in error_str:
                    reply = "Gemini API quota exceeded. Please try again later."
                else:
                    reply = f"Gemini Error: {str(e)}"

        else:
            reply = "Invalid model selected. Please choose OpenAI, Claude, or Gemini."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

@app.route("/tools/execute", methods=["POST"])
def execute_data_tool():
    """Manual tool execution endpoint for testing or direct access"""
    try:
        data = request.json
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            return jsonify({"error": "tool_name is required"}), 400
            
        result = data_tools.execute_tool(tool_name, arguments)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools/list", methods=["GET"])
def list_available_tools():
    """Get list of available data access tools"""
    try:
        tools_definition = data_tools.get_tools_definition()
        simplified_tools = []
        
        for tool in tools_definition:
            func = tool["function"]
            simplified_tools.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": list(func["parameters"]["properties"].keys())
            })
            
        return jsonify({"tools": simplified_tools})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/context", methods=["GET"])
def get_context():
    """Get all stored context data (metadata only)"""
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
            cm.remove_file(filename)
            
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