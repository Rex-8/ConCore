import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
import base64
from io import StringIO, BytesIO
import warnings
warnings.filterwarnings('ignore')

class ScriptExecutor:
    """
    Safe Python script executor for data analysis.
    Executes LLM-generated scripts with sandboxing and security controls.
    """
    
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        self.allowed_imports = {
            'pandas', 'pd', 'numpy', 'np', 'matplotlib.pyplot', 'plt',
            'seaborn', 'sns', 'json', 'datetime', 'math', 'statistics',
            'collections', 're', 'itertools'
        }
        self.forbidden_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'globals()', 'locals()', 'vars()', 'dir()',
            'getattr', 'setattr', 'delattr', 'hasattr'
        ]
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for LLM integration"""
        return {
            "type": "function",
            "function": {
                "name": "execute_script",
                "description": "Execute Python script for data analysis and visualization. Can generate plots, perform calculations, and analyze data from uploaded files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Python script to execute. Can use pandas, numpy, matplotlib, seaborn. Must use 'df' variable for dataframes loaded from files."
                        },
                        "filename": {
                            "type": "string", 
                            "description": "Primary file to load as 'df' variable (optional if script doesn't need file data)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what the script does"
                        },
                        "output_type": {
                            "type": "string",
                            "enum": ["analysis", "plot", "both"],
                            "description": "Expected output type",
                            "default": "analysis"
                        }
                    },
                    "required": ["script", "description"]
                }
            }
        }
    
    def _load_dataframe(self, filename: str) -> pd.DataFrame:
        """Load file into pandas DataFrame"""
        filepath = os.path.join(self.upload_folder, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} not found")
            
        ext = os.path.splitext(filename)[1].lower()
        
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
            # Get first table for default load
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
    
    def _validate_script(self, script: str) -> bool:
        """Basic security validation of script"""
        script_lower = script.lower()
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.lower() in script_lower:
                raise ValueError(f"Forbidden operation detected: {pattern}")
        
        # Check for file operations
        dangerous_ops = ['open(', 'file(', 'write(', 'delete(', 'remove(']
        for op in dangerous_ops:
            if op in script_lower:
                raise ValueError(f"File operation not allowed: {op}")
        
        return True
    
    def _capture_output(self, script: str, local_vars: Dict) -> Dict[str, Any]:
        """Execute script and capture outputs"""
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Capture any plots
        plt.figure(figsize=(10, 6))
        
        try:
            # Execute the script
            exec(script, {
                'pd': pd, 'pandas': pd,
                'np': np, 'numpy': np,
                'plt': plt, 'matplotlib': matplotlib,
                'sns': sns, 'seaborn': sns,
                'json': json,
                'datetime': datetime
            }, local_vars)
            
            # Get printed output
            output_text = captured_output.getvalue()
            
            # Check if a plot was created
            plot_data = None
            if plt.get_fignums():
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close('all')
            
            return {
                "output": output_text,
                "plot": plot_data,
                "variables": {k: str(v) for k, v in local_vars.items() 
                            if not k.startswith('_') and not callable(v)}
            }
            
        finally:
            sys.stdout = old_stdout
            plt.close('all')
    
    def execute_script(self, script: str, filename: Optional[str] = None, 
                      description: str = "", output_type: str = "analysis") -> Dict[str, Any]:
        """
        Execute a Python script safely with optional data loading
        
        Args:
            script: Python code to execute
            filename: Optional file to load as 'df' variable
            description: Description of what script does
            output_type: Expected output type
            
        Returns:
            Dict with execution results, output, and any plots
        """
        try:
            # Validate script
            self._validate_script(script)
            
            # Prepare local variables
            local_vars = {}
            
            # Load data if filename provided
            if filename:
                df = self._load_dataframe(filename)
                local_vars['df'] = df
                local_vars['data'] = df  # Alternative name
            
            # Execute script and capture output
            execution_result = self._capture_output(script, local_vars)
            
            # Prepare result
            result = {
                "success": True,
                "description": description,
                "script": script,
                "filename": filename,
                "output_type": output_type,
                "execution_time": datetime.now().isoformat(),
                "printed_output": execution_result["output"],
                "has_plot": execution_result["plot"] is not None,
                "variables_created": list(execution_result["variables"].keys())
            }
            
            if execution_result["plot"]:
                result["plot_base64"] = execution_result["plot"]
            
            # Include variable values if they're small
            for var_name, var_value in execution_result["variables"].items():
                if len(str(var_value)) < 500:  # Only include small variables
                    result[f"variable_{var_name}"] = var_value
            
            return result
            
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