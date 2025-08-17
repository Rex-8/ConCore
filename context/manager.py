from typing import List, Dict, Any, Optional
from datetime import datetime

class ContextManager:
    """
    Manages context data for the application, storing information about
    uploaded files and other contextual information.
    """
    
    def __init__(self):
        self.context_store: List[Dict[str, Any]] = []
    
    def upload(self, content_type: str, data: Dict[str, Any]) -> None:
        """
        Store new context data.
        
        Args:
            content_type: Type of content ("file-content", "conversation", etc.)
            data: The data to store
        """
        context_entry = {
            "type": content_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.context_store)  # Simple ID based on position
        }
        
        # For file-content, check if we already have this file and update it
        if content_type == "file-content":
            filename = data.get("filename")
            if filename:
                # Remove existing entry for the same file
                self.context_store = [
                    entry for entry in self.context_store 
                    if not (entry["type"] == "file-content" and 
                           entry["data"].get("filename") == filename)
                ]
        
        self.context_store.append(context_entry)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all stored context data.
        
        Returns:
            List of all context entries
        """
        return self.context_store.copy()
    
    def get_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        Get context data filtered by type.
        
        Args:
            content_type: The type to filter by
            
        Returns:
            List of context entries matching the type
        """
        return [
            entry for entry in self.context_store 
            if entry["type"] == content_type
        ]
    
    def get_files(self) -> List[Dict[str, Any]]:
        """
        Get all file-content entries.
        
        Returns:
            List of file context entries
        """
        return self.get_by_type("file-content")
    
    def remove_file(self, filename: str) -> bool:
        """
        Remove a specific file from context.
        
        Args:
            filename: Name of file to remove
            
        Returns:
            True if file was found and removed, False otherwise
        """
        original_length = len(self.context_store)
        self.context_store = [
            entry for entry in self.context_store 
            if not (entry["type"] == "file-content" and 
                   entry["data"].get("filename") == filename)
        ]
        return len(self.context_store) < original_length
    
    def clear(self) -> None:
        """Clear all stored context data."""
        self.context_store.clear()
    
    def clear_by_type(self, content_type: str) -> None:
        """
        Clear all context data of a specific type.
        
        Args:
            content_type: The type to clear
        """
        self.context_store = [
            entry for entry in self.context_store 
            if entry["type"] != content_type
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of stored context data.
        
        Returns:
            Summary including counts by type and total storage
        """
        summary = {
            "total_entries": len(self.context_store),
            "by_type": {},
            "files": []
        }
        
        for entry in self.context_store:
            entry_type = entry["type"]
            if entry_type not in summary["by_type"]:
                summary["by_type"][entry_type] = 0
            summary["by_type"][entry_type] += 1
            
            # Add file details for file-content entries
            if entry_type == "file-content":
                file_data = entry["data"]
                summary["files"].append({
                    "filename": file_data.get("filename"),
                    "population": file_data.get("population", 0),
                    "columns": len(file_data.get("row_headings", [])),
                    "file_type": file_data.get("file_type"),
                    "timestamp": entry["timestamp"]
                })
        
        return summary