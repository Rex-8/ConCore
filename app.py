from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import time

# AI APIs
import openai
import anthropic
import google.generativeai as genai

# File parsing logic
from data_parse.main_parser import parse_file

# Enhanced Context Manager
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
    """Enhanced chat endpoint with conversation history and tool calling"""
    start_time = time.time()
    
    try:
        data = request.json
        model = data.get("model")
        api_key = data.get("apiKey")
        user_input = data.get("message")
        include_history = data.get("include_history", True)  # New parameter
        history_turns = data.get("history_turns", 5)  # How many turns to include

        if not model or not api_key or not user_input:
            return jsonify({"reply": "Missing required parameters."}), 400

        # Get current context (metadata only)
        context_data = cm.get_all()
        enhanced_message = user_input
        
        # Build file context
        files_context = []
        if context_data:
            context_str = "Available file context (metadata only - use tools to access actual data):\n"
            for content in context_data:
                if content["type"] == "file-content":
                    data_info = content["data"]
                    files_context.append(data_info["filename"])
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
            context_str += "- query_sqlite: Run SQL queries on database files\n\n"
            
            enhanced_message = context_str + enhanced_message

        # Add conversation history if requested
        if include_history:
            conversation_context = cm.get_conversation_context_for_llm(
                turns_to_include=history_turns,
                include_tool_calls=True
            )
            if conversation_context:
                enhanced_message = conversation_context + "\n" + enhanced_message

        # Get tool definitions
        tools_definition = data_tools.get_tools_definition()
        tool_calls_made = []

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
                        
                        # Execute tool and track calls
                        result = data_tools.execute_tool(tool_name, tool_args)
                        tool_calls_made.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result
                        })
                        
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
                            
                            # Execute tool and track calls
                            result = data_tools.execute_tool(tool_name, tool_args)
                            tool_calls_made.append({
                                "tool": tool_name,
                                "arguments": tool_args,
                                "result": result
                            })
                            
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

        # Calculate response time
        response_time = time.time() - start_time
        
        # Store conversation turn in context manager
        conversation_id = cm.add_conversation_turn(
            user_message=user_input,
            assistant_response=reply,
            model_used=model,
            files_context=files_context,
            tool_calls=tool_calls_made,
            metadata={
                "response_time_seconds": response_time,
                "message_length": len(enhanced_message),
                "tools_available": len(tools_definition),
                "include_history": include_history,
                "history_turns_included": history_turns if include_history else 0
            }
        )

        return jsonify({
            "reply": reply,
            "conversation_id": conversation_id,
            "response_time": f"{response_time:.2f}s",
            "tools_used": len(tool_calls_made)
        })

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

# New conversation management endpoints
@app.route("/conversation/history", methods=["GET"])
def get_conversation_history():
    """Get conversation history with optional filtering"""
    try:
        limit = request.args.get("limit", type=int)
        session_id = request.args.get("session_id")
        include_context = request.args.get("include_context", "true").lower() == "true"
        
        history = cm.get_conversation_history(
            limit=limit,
            session_id=session_id,
            include_context=include_context
        )
        
        return jsonify({
            "history": history,
            "total_conversations": len(history),
            "current_session": cm.session_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/conversation/search", methods=["POST"])
def search_conversations():
    """Search through conversation history"""
    try:
        data = request.json
        query = data.get("query", "")
        search_in = data.get("search_in", "both")  # "user", "assistant", "both"
        case_sensitive = data.get("case_sensitive", False)
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        results = cm.search_conversations(
            query=query,
            search_in=search_in,
            case_sensitive=case_sensitive
        )
        
        return jsonify({
            "results": results,
            "total_matches": len(results),
            "query": query,
            "search_in": search_in
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/conversation/summary", methods=["GET"])
def get_conversation_summary():
    """Get conversation summary and statistics"""
    try:
        summary = cm.get_conversation_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/conversation/export", methods=["GET"])
def export_conversation():
    """Export conversation history"""
    try:
        format_type = request.args.get("format", "json")  # "json" or "markdown"
        
        if format_type not in ["json", "markdown"]:
            return jsonify({"error": "Format must be 'json' or 'markdown'"}), 400
        
        exported_data = cm.export_conversation_history(format=format_type)
        
        # Set appropriate content type and filename
        if format_type == "json":
            response = app.response_class(
                response=exported_data,
                status=200,
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename=conversation_history_{cm.session_id}.json'
                }
            )
        else:  # markdown
            response = app.response_class(
                response=exported_data,
                status=200,
                mimetype='text/markdown',
                headers={
                    'Content-Disposition': f'attachment; filename=conversation_history_{cm.session_id}.md'
                }
            )
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    """Get all stored context data (metadata only) and conversation summary"""
    try:
        context_data = cm.get_all()
        conversation_summary = cm.get_conversation_summary()
        
        return jsonify({
            "file_context": context_data,
            "conversation_summary": conversation_summary,
            "session_id": cm.session_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/context/clear", methods=["POST"])
def clear_context():
    """Clear context data with options"""
    try:
        data = request.json or {}
        clear_type = data.get("type", "all")  # "all", "files", "conversation"
        
        if clear_type == "all":
            cm.clear()
            message = "All context and conversation history cleared successfully"
        elif clear_type == "files":
            cm.clear_by_type("file-content")
            message = "File context cleared successfully"
        elif clear_type == "conversation":
            cm.clear_conversation_only()
            message = "Conversation history cleared successfully"
        else:
            return jsonify({"error": "Invalid clear type. Use 'all', 'files', or 'conversation'"}), 400
        
        return jsonify({
            "message": message,
            "new_session_id": cm.session_id
        })
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

# Session management endpoints
@app.route("/session/info", methods=["GET"])
def get_session_info():
    """Get current session information"""
    try:
        return jsonify({
            "session_id": cm.session_id,
            "session_start": cm.session_start_time.isoformat(),
            "conversation_count": len(cm.conversation_history),
            "files_loaded": len(cm.get_files()),
            "summary": cm.get_conversation_summary()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/session/new", methods=["POST"])
def start_new_session():
    """Start a new session (clears conversation but keeps files)"""
    try:
        data = request.json or {}
        keep_files = data.get("keep_files", True)
        
        old_session_id = cm.session_id
        
        if keep_files:
            cm.clear_conversation_only()
            message = f"New session started. Files retained from previous session."
        else:
            cm.clear()
            message = f"New session started. All data cleared."
        
        return jsonify({
            "message": message,
            "old_session_id": old_session_id,
            "new_session_id": cm.session_id,
            "files_retained": keep_files
        })
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
    print(f"ConCore starting with session: {cm.session_id}")
    app.run(debug=True, host='0.0.0.0', port=8000)