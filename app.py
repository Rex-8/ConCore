from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import time
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import traceback
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# AI APIs
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'json', 'csv', 'xlsx', 'xls', 'db', 'sqlite'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# In-memory storage for this session
session_data = {
    "files": {},
    "conversation_history": [],
    "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
}

class DataProcessor:
    """Handle file parsing and data operations"""
    
    @staticmethod
    def parse_file(filepath):
        """Parse uploaded file and return metadata"""
        try:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == ".csv":
                df = pd.read_csv(filepath)
                return {
                    "filename": filename,
                    "file_type": "csv",
                    "population": len(df),
                    "features": df.columns.tolist(),
                    "sample_data": df.head(3).to_dict('records'),
                    "data_types": df.dtypes.astype(str).to_dict()
                }
                
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(filepath)
                return {
                    "filename": filename,
                    "file_type": "excel", 
                    "population": len(df),
                    "features": df.columns.tolist(),
                    "sample_data": df.head(3).to_dict('records'),
                    "data_types": df.dtypes.astype(str).to_dict()
                }
                
            elif ext == ".json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    return {
                        "filename": filename,
                        "file_type": "json",
                        "population": len(df),
                        "features": df.columns.tolist(),
                        "sample_data": df.head(3).to_dict('records'),
                        "data_types": df.dtypes.astype(str).to_dict()
                    }
                else:
                    return {
                        "filename": filename,
                        "file_type": "json",
                        "population": 1,
                        "features": list(data.keys()) if isinstance(data, dict) else ["value"],
                        "sample_data": [data] if isinstance(data, dict) else [{"value": data}],
                        "data_types": {}
                    }
                    
            elif ext in [".db", ".sqlite"]:
                conn = sqlite3.connect(filepath)
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_info = {}
                total_rows = 0
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    total_rows += row_count
                    
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    table_info[table] = columns
                
                conn.close()
                
                return {
                    "filename": filename,
                    "file_type": "sqlite",
                    "population": total_rows,
                    "features": table_info,
                    "tables": tables,
                    "sample_data": []
                }
                
            else:
                raise ValueError(f"Unsupported file type: {ext}")
                
        except Exception as e:
            raise Exception(f"Failed to parse file: {str(e)}")

class ScriptExecutor:
    """Execute Python scripts safely"""
    
    def __init__(self):
        self.upload_folder = UPLOAD_FOLDER
        
    def execute_script(self, script, filename=None, description="", output_type="both"):
        """Execute Python script with optional data loading"""
        try:
            # Prepare execution environment
            exec_globals = {
                'pd': pd, 'pandas': pd,
                'np': np, 'numpy': np,
                'plt': plt, 'matplotlib': matplotlib,
                'sns': sns, 'seaborn': sns,
                'json': json,
                'datetime': datetime
            }
            
            exec_locals = {}
            
            # Load data if filename provided
            if filename and filename in session_data["files"]:
                filepath = os.path.join(self.upload_folder, filename)
                if os.path.exists(filepath):
                    df = self._load_dataframe(filepath)
                    exec_locals['df'] = df
                    exec_locals['data'] = df
            
            # Capture stdout
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Execute script
            plt.figure(figsize=(10, 6))
            exec(script, exec_globals, exec_locals)
            
            # Get output
            output_text = captured_output.getvalue()
            sys.stdout = old_stdout
            
            # Check for plot
            plot_data = None
            if plt.get_fignums():
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close('all')
            
            return {
                "success": True,
                "description": description,
                "script": script,
                "filename": filename,
                "output_type": output_type,
                "execution_time": datetime.now().isoformat(),
                "printed_output": output_text,
                "has_plot": plot_data is not None,
                "plot_base64": plot_data,
                "variables_created": [k for k in exec_locals.keys() if not k.startswith('_')]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "script": script,
                "filename": filename,
                "description": description
            }
        finally:
            plt.close('all')
    
    def _load_dataframe(self, filepath):
        """Load file into pandas DataFrame"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".csv":
            return pd.read_csv(filepath)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(filepath)
        elif ext == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                return pd.DataFrame([{"value": data}])
        elif ext in [".db", ".sqlite"]:
            conn = sqlite3.connect(filepath)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            if tables:
                table_name = tables[0][0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", conn)
            else:
                df = pd.DataFrame()
            conn.close()
            return df
        else:
            raise ValueError(f"Unsupported file type: {ext}")

class DataTools:
    """Data analysis tools"""
    
    def __init__(self):
        self.upload_folder = UPLOAD_FOLDER
    
    def get_sample(self, filename, rows=10, offset=0):
        """Get sample rows from dataset"""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            df = self._load_dataframe(filepath)
            
            sample_df = df.iloc[offset:offset+rows]
            
            return {
                "success": True,
                "data": sample_df.to_dict('records'),
                "rows_returned": len(sample_df),
                "total_rows": len(df),
                "columns": df.columns.tolist()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_statistics(self, filename, columns=None):
        """Get descriptive statistics"""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            df = self._load_dataframe(filepath)
            
            if columns:
                df = df[columns]
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {"success": False, "error": "No numeric columns found"}
            
            stats = numeric_df.describe()
            
            return {
                "success": True,
                "statistics": stats.to_dict(),
                "columns_analyzed": stats.columns.tolist(),
                "total_rows": len(df)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_data(self, filename, column, value, limit=20):
        """Search for specific values"""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            df = self._load_dataframe(filepath)
            
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found"}
            
            mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
            matching_df = df[mask].head(limit)
            
            return {
                "success": True,
                "data": matching_df.to_dict('records'),
                "rows_returned": len(matching_df),
                "total_matches": mask.sum(),
                "search_column": column,
                "search_value": value
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def query_sqlite(self, filename, query, limit=50):
        """Execute SQL query"""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            
            if not query.strip().upper().startswith("SELECT"):
                return {"success": False, "error": "Only SELECT queries are allowed"}
            
            conn = sqlite3.connect(filepath)
            
            if "LIMIT" not in query.upper():
                query = f"{query} LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return {
                "success": True,
                "data": df.to_dict('records'),
                "rows_returned": len(df),
                "columns": df.columns.tolist(),
                "query": query
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_dataframe(self, filepath):
        """Load file into DataFrame"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".csv":
            return pd.read_csv(filepath)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(filepath)
        elif ext == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported file type: {ext}")

# Initialize components
data_processor = DataProcessor()
script_executor = ScriptExecutor()
data_tools = DataTools()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload and parse file"""
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
        result = data_processor.parse_file(filepath)
        
        # Store in session data
        session_data["files"][filename] = {
            "filepath": filepath,
            "metadata": result,
            "uploaded_at": datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_with_model():
    """Chat endpoint with AI models"""
    try:
        data = request.json
        model = data.get("model")
        api_key = data.get("apiKey")
        user_input = data.get("message")

        if not model or not api_key or not user_input:
            return jsonify({"reply": "Missing required parameters."}), 400

        # Build context from uploaded files
        context_str = ""
        if session_data["files"]:
            context_str = "Available datasets:\n"
            for filename, file_info in session_data["files"].items():
                metadata = file_info["metadata"]
                context_str += f"- {filename}: {metadata.get('population', 0)} rows"
                
                features = metadata.get('features', [])
                if isinstance(features, dict):  # SQLite
                    context_str += f", Tables: {list(features.keys())}\n"
                else:  # CSV, Excel, JSON
                    context_str += f", {len(features)} columns\n"
            
            context_str += "\nYou can analyze this data, create visualizations, or generate Python/SQL code.\n\n"

        enhanced_message = context_str + user_input

        # Route to appropriate model
        if model == "openai" and openai:
            try:
                client = openai.OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_message}],
                    max_tokens=1500,
                    temperature=0.7
                )
                
                reply = response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e).lower()
                if "incorrect_api_key" in error_str or "invalid" in error_str:
                    reply = "Invalid OpenAI API key. Please check your key and try again."
                elif "rate_limit" in error_str:
                    reply = "OpenAI API rate limit exceeded. Please try again later."
                else:
                    reply = f"OpenAI Error: {str(e)}"

        elif model == "claude" and anthropic:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": enhanced_message}]
                )
                
                reply = response.content[0].text
                
            except Exception as e:
                error_str = str(e).lower()
                if "authentication" in error_str or "invalid" in error_str:
                    reply = "Invalid Claude API key. Please check your key and try again."
                elif "rate_limit" in error_str:
                    reply = "Claude API rate limit exceeded. Please try again later."
                else:
                    reply = f"Claude Error: {str(e)}"

        elif model == "gemini" and genai:
            try:
                genai.configure(api_key=api_key)
                model_client = genai.GenerativeModel("gemini-1.5-flash")
                
                response = model_client.generate_content(
                    enhanced_message,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1500,
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
            reply = f"Model '{model}' is not available or API library not installed."

        # Store conversation
        session_data["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_input,
            "assistant_response": reply,
            "model_used": model
        })

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

@app.route("/execute_script", methods=["POST"])
def execute_script():
    """Execute Python script"""
    try:
        data = request.json
        script = data.get("script", "")
        filename = data.get("filename")
        description = data.get("description", "Script execution")
        output_type = data.get("output_type", "both")
        
        if not script:
            return jsonify({"success": False, "error": "No script provided"})
        
        result = script_executor.execute_script(script, filename, description, output_type)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/data_tools/<tool_name>", methods=["POST"])
def data_tool(tool_name):
    """Execute data analysis tools"""
    try:
        data = request.json
        
        if tool_name == "sample":
            filename = data.get("filename")
            rows = data.get("rows", 10)
            offset = data.get("offset", 0)
            return jsonify(data_tools.get_sample(filename, rows, offset))
            
        elif tool_name == "statistics":
            filename = data.get("filename")
            columns = data.get("columns")
            return jsonify(data_tools.get_statistics(filename, columns))
            
        elif tool_name == "search":
            filename = data.get("filename")
            column = data.get("column")
            value = data.get("value")
            limit = data.get("limit", 20)
            return jsonify(data_tools.search_data(filename, column, value, limit))
            
        elif tool_name == "query":
            filename = data.get("filename")
            query = data.get("query")
            limit = data.get("limit", 50)
            return jsonify(data_tools.query_sqlite(filename, query, limit))
            
        else:
            return jsonify({"success": False, "error": f"Unknown tool: {tool_name}"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/files", methods=["GET"])
def list_files():
    """List uploaded files"""
    try:
        files_info = []
        for filename, file_data in session_data["files"].items():
            metadata = file_data["metadata"]
            files_info.append({
                "filename": filename,
                "file_type": metadata.get("file_type"),
                "population": metadata.get("population", 0),
                "features": len(metadata.get("features", [])) if isinstance(metadata.get("features"), list) else len(metadata.get("features", {})),
                "uploaded_at": file_data["uploaded_at"]
            })
        
        return jsonify({
            "files": files_info,
            "total_files": len(files_info),
            "session_id": session_data["session_id"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete uploaded file"""
    try:
        if filename in session_data["files"]:
            filepath = session_data["files"][filename]["filepath"]
            if os.path.exists(filepath):
                os.remove(filepath)
            del session_data["files"][filename]
            return jsonify({"message": f"File {filename} deleted successfully"})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/session/clear", methods=["POST"])
def clear_session():
    """Clear session data"""
    try:
        # Clear conversation history
        session_data["conversation_history"].clear()
        
        # Optionally clear files
        clear_files = request.json.get("clear_files", False) if request.json else False
        if clear_files:
            for filename, file_data in session_data["files"].items():
                filepath = file_data["filepath"]
                if os.path.exists(filepath):
                    os.remove(filepath)
            session_data["files"].clear()
        
        # Generate new session ID
        session_data["session_id"] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            "message": "Session cleared successfully",
            "session_id": session_data["session_id"],
            "files_cleared": clear_files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/session/info", methods=["GET"])
def session_info():
    """Get session information"""
    try:
        return jsonify({
            "session_id": session_data["session_id"],
            "total_files": len(session_data["files"]),
            "conversation_turns": len(session_data["conversation_history"]),
            "files": list(session_data["files"].keys())
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

if __name__ == "__main__":
    print(f"ConCore starting with session: {session_data['session_id']}")
    app.run(debug=True, host='0.0.0.0', port=8000)