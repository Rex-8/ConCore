from typing import Dict, List, Any, Optional
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime

class DataAccessTools:
    """
    Provides function tools that LLMs can call to access actual file data.
    Data is only loaded when explicitly requested through these functions.
    """
    
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        
    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        Returns the function definitions that can be passed to LLM APIs
        for tool calling (OpenAI format, adaptable to Claude/Gemini)
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_data_sample",
                    "description": "Get a sample of rows from a dataset file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the file to sample from"
                            },
                            "rows": {
                                "type": "integer", 
                                "description": "Number of rows to return (default 10, max 100)",
                                "default": 10
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Starting row offset (default 0)",
                                "default": 0
                            }
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_column_data",
                    "description": "Get data from specific columns of a dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the file"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to retrieve"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of rows (default 50, max 200)",
                                "default": 50
                            }
                        },
                        "required": ["filename", "columns"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_statistics",
                    "description": "Get descriptive statistics for numeric columns",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the file"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Column names to analyze (optional - all numeric columns if not specified)"
                            }
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_data", 
                    "description": "Search for rows matching specific criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string", 
                                "description": "Name of the file"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column name to search in"
                            },
                            "value": {
                                "type": "string",
                                "description": "Value to search for (supports partial matches)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return (default 20)",
                                "default": 20
                            }
                        },
                        "required": ["filename", "column", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_sqlite",
                    "description": "Execute SQL query on SQLite database file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of the SQLite file"
                            },
                            "query": {
                                "type": "string", 
                                "description": "SQL query to execute (SELECT statements only)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Row limit for results (default 50, max 200)",
                                "default": 50
                            }
                        },
                        "required": ["filename", "query"]
                    }
                }
            }
        ]
    
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
            # Convert JSON to DataFrame based on structure
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                return pd.DataFrame([{"value": data}])
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _convert_to_json_safe(self, obj):
        """Convert pandas/numpy types to JSON-safe types"""
        if pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool function and return results.
        This is called by the chat endpoint when LLM requests tool usage.
        """
        try:
            if tool_name == "get_data_sample":
                return self.get_data_sample(**arguments)
            elif tool_name == "get_column_data":
                return self.get_column_data(**arguments)
            elif tool_name == "get_statistics":
                return self.get_statistics(**arguments)
            elif tool_name == "search_data":
                return self.search_data(**arguments)
            elif tool_name == "query_sqlite":
                return self.query_sqlite(**arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def get_data_sample(self, filename: str, rows: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get sample rows from dataset"""
        rows = min(max(1, rows), 100)  # Clamp between 1-100
        offset = max(0, offset)
        
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in [".db", ".sqlite"]:
            return {"error": "Use query_sqlite for database files"}
            
        df = self._load_dataframe(filename)
        
        # Apply offset and limit
        sample_df = df.iloc[offset:offset+rows]
        
        # Convert to JSON-safe format
        result_data = []
        for _, row in sample_df.iterrows():
            row_dict = {}
            for col, val in row.items():
                row_dict[col] = self._convert_to_json_safe(val)
            result_data.append(row_dict)
        
        return {
            "data": result_data,
            "rows_returned": len(result_data),
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "offset": offset
        }
    
    def get_column_data(self, filename: str, columns: List[str], limit: int = 50) -> Dict[str, Any]:
        """Get data from specific columns"""
        limit = min(max(1, limit), 200)
        
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".db", ".sqlite"]:
            return {"error": "Use query_sqlite for database files"}
            
        df = self._load_dataframe(filename)
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}"}
        
        # Select columns and limit rows
        subset_df = df[columns].head(limit)
        
        # Convert to JSON-safe format
        result_data = []
        for _, row in subset_df.iterrows():
            row_dict = {}
            for col in columns:
                row_dict[col] = self._convert_to_json_safe(row[col])
            result_data.append(row_dict)
        
        return {
            "data": result_data,
            "columns": columns,
            "rows_returned": len(result_data),
            "total_rows": len(df)
        }
    
    def get_statistics(self, filename: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get descriptive statistics for numeric columns"""
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".db", ".sqlite"]:
            return {"error": "Use query_sqlite for database files"}
            
        df = self._load_dataframe(filename)
        
        # Get numeric columns
        if columns:
            # Validate specified columns exist and are numeric
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"error": f"Columns not found: {missing_cols}"}
            numeric_df = df[columns].select_dtypes(include=['number'])
        else:
            numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found"}
        
        # Calculate statistics
        stats = numeric_df.describe()
        
        # Convert to JSON-safe format
        result = {}
        for col in stats.columns:
            result[col] = {}
            for stat_name in stats.index:
                result[col][stat_name] = self._convert_to_json_safe(stats.loc[stat_name, col])
        
        return {
            "statistics": result,
            "columns_analyzed": list(stats.columns),
            "total_rows": len(df)
        }
    
    def search_data(self, filename: str, column: str, value: str, limit: int = 20) -> Dict[str, Any]:
        """Search for rows matching criteria"""
        limit = min(max(1, limit), 200)
        
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".db", ".sqlite"]:
            return {"error": "Use query_sqlite for database files"}
            
        df = self._load_dataframe(filename)
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        # Perform search (case-insensitive partial match)
        mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
        matching_df = df[mask].head(limit)
        
        # Convert to JSON-safe format
        result_data = []
        for _, row in matching_df.iterrows():
            row_dict = {}
            for col, val in row.items():
                row_dict[col] = self._convert_to_json_safe(val)
            result_data.append(row_dict)
        
        return {
            "data": result_data,
            "rows_returned": len(result_data),
            "total_matches": mask.sum(),
            "search_column": column,
            "search_value": value
        }
    
    def query_sqlite(self, filename: str, query: str, limit: int = 50) -> Dict[str, Any]:
        """Execute SQL query on SQLite database"""
        limit = min(max(1, limit), 200)
        
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".db", ".sqlite"]:
            return {"error": "File is not a SQLite database"}
        
        # Security: Only allow SELECT statements
        if not query.strip().upper().startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed"}
        
        filepath = os.path.join(self.upload_folder, filename)
        
        try:
            conn = sqlite3.connect(filepath)
            
            # Add LIMIT to query if not present
            if "LIMIT" not in query.upper():
                query = f"{query} LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert to JSON-safe format
            result_data = []
            for _, row in df.iterrows():
                row_dict = {}
                for col, val in row.items():
                    row_dict[col] = self._convert_to_json_safe(val)
                result_data.append(row_dict)
            
            return {
                "data": result_data,
                "rows_returned": len(result_data),
                "columns": df.columns.tolist(),
                "query": query
            }
            
        except Exception as e:
            return {"error": f"SQL query failed: {str(e)}"}


# Integration helper for different LLM APIs
class LLMToolIntegration:
    """Helper class to integrate tools with different LLM APIs"""
    
    @staticmethod
    def format_tools_for_openai(tools_definition: List[Dict]) -> List[Dict]:
        """Format tools for OpenAI API"""
        return tools_definition
    
    @staticmethod  
    def format_tools_for_claude(tools_definition: List[Dict]) -> List[Dict]:
        """Format tools for Claude API (Anthropic format)"""
        claude_tools = []
        for tool in tools_definition:
            claude_tools.append({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"], 
                "input_schema": tool["function"]["parameters"]
            })
        return claude_tools
    
    @staticmethod
    def format_tools_for_gemini(tools_definition: List[Dict]) -> List[Dict]:
        """Format tools for Gemini API (Google format)"""
        # Gemini uses a similar structure but may need adaptation
        gemini_tools = []
        for tool in tools_definition:
            gemini_tools.append({
                "function_declarations": [{
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"]
                }]
            })
        return gemini_tools