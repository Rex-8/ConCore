from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ContextManager:
    """
    Enhanced context manager that stores:
    1. File metadata (existing functionality)
    2. Full conversation history 
    3. Session management
    """
    
    def __init__(self):
        self.context_store: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 10000}"
    
    def upload(self, content_type: str, data: Dict[str, Any]) -> None:
        """
        Store new context data (existing functionality).
        
        Args:
            content_type: Type of content ("file-content", "conversation", etc.)
            data: The data to store
        """
        context_entry = {
            "type": content_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.context_store),
            "session_id": self.session_id
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
    
    def add_conversation_turn(
        self, 
        user_message: str, 
        assistant_response: str, 
        model_used: str,
        files_context: Optional[List[str]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a complete conversation turn to history.
        
        Args:
            user_message: The user's input message
            assistant_response: The AI's response
            model_used: Which model was used (openai, claude, gemini)
            files_context: List of files that were available during this turn
            tool_calls: Any tool calls made during this turn
            metadata: Additional metadata (tokens used, response time, etc.)
            
        Returns:
            conversation_id: Unique ID for this conversation turn
        """
        conversation_id = f"conv_{len(self.conversation_history)}_{hash(user_message + assistant_response) % 10000}"
        
        conversation_turn = {
            "conversation_id": conversation_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "turn_number": len(self.conversation_history) + 1,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "model_used": model_used,
            "files_context": files_context or [],
            "tool_calls": tool_calls or [],
            "metadata": metadata or {},
            "message_length": {
                "user": len(user_message),
                "assistant": len(assistant_response)
            }
        }
        
        self.conversation_history.append(conversation_turn)
        return conversation_id
    
    def get_conversation_history(
        self, 
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with optional filtering.
        
        Args:
            limit: Maximum number of recent conversations to return
            session_id: Filter by specific session (default: current session)
            include_context: Whether to include file context info
            
        Returns:
            List of conversation turns
        """
        # Filter by session if specified
        if session_id:
            filtered_history = [
                conv for conv in self.conversation_history 
                if conv.get("session_id") == session_id
            ]
        else:
            filtered_history = [
                conv for conv in self.conversation_history 
                if conv.get("session_id") == self.session_id
            ]
        
        # Apply limit
        if limit:
            filtered_history = filtered_history[-limit:]
        
        # Add context information if requested
        if include_context:
            for conv in filtered_history:
                conv["available_files"] = [
                    entry["data"]["filename"] 
                    for entry in self.context_store 
                    if entry["type"] == "file-content"
                ]
        
        return filtered_history
    
    def get_conversation_context_for_llm(
        self, 
        turns_to_include: int = 5,
        include_tool_calls: bool = False
    ) -> str:
        """
        Generate conversation context string for LLM consumption.
        
        Args:
            turns_to_include: Number of recent conversation turns to include
            include_tool_calls: Whether to include tool call details
            
        Returns:
            Formatted conversation context string
        """
        recent_history = self.get_conversation_history(limit=turns_to_include)
        
        if not recent_history:
            return ""
        
        context_parts = []
        context_parts.append("Recent conversation history:")
        
        for i, conv in enumerate(recent_history):
            turn_num = conv["turn_number"]
            timestamp = conv["timestamp"][:19]  # Remove milliseconds
            
            context_parts.append(f"\n--- Turn {turn_num} ({timestamp}) ---")
            context_parts.append(f"User: {conv['user_message']}")
            context_parts.append(f"Assistant ({conv['model_used']}): {conv['assistant_response']}")
            
            # Include tool calls if requested
            if include_tool_calls and conv["tool_calls"]:
                context_parts.append("Tool calls made:")
                for tool_call in conv["tool_calls"]:
                    context_parts.append(f"  - {tool_call.get('tool', 'unknown')}: {tool_call.get('arguments', {})}")
            
            # Include files that were available
            if conv["files_context"]:
                context_parts.append(f"Files available: {', '.join(conv['files_context'])}")
        
        context_parts.append("\n--- End of conversation history ---\n")
        
        return "\n".join(context_parts)
    
    def search_conversations(
        self, 
        query: str, 
        search_in: str = "both",
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search through conversation history.
        
        Args:
            query: Search term
            search_in: "user", "assistant", or "both"
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching conversation turns
        """
        if not case_sensitive:
            query = query.lower()
        
        results = []
        
        for conv in self.conversation_history:
            user_msg = conv["user_message"]
            assistant_msg = conv["assistant_response"]
            
            if not case_sensitive:
                user_msg = user_msg.lower()
                assistant_msg = assistant_msg.lower()
            
            match = False
            
            if search_in in ["user", "both"] and query in user_msg:
                match = True
            
            if search_in in ["assistant", "both"] and query in assistant_msg:
                match = True
            
            if match:
                results.append(conv)
        
        return results
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of conversation history.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {
                "total_conversations": 0,
                "session_id": self.session_id,
                "session_duration": "0 minutes",
                "models_used": [],
                "total_tokens": {"user": 0, "assistant": 0},
                "files_discussed": [],
                "tools_used": []
            }
        
        # Calculate session duration
        session_duration = datetime.now() - self.session_start_time
        duration_minutes = int(session_duration.total_seconds() / 60)
        
        # Collect statistics
        models_used = list(set(conv["model_used"] for conv in self.conversation_history))
        total_user_chars = sum(conv["message_length"]["user"] for conv in self.conversation_history)
        total_assistant_chars = sum(conv["message_length"]["assistant"] for conv in self.conversation_history)
        
        # Get unique files mentioned
        files_discussed = set()
        for conv in self.conversation_history:
            files_discussed.update(conv["files_context"])
        
        # Get unique tools used
        tools_used = set()
        for conv in self.conversation_history:
            for tool_call in conv["tool_calls"]:
                tools_used.add(tool_call.get("tool", "unknown"))
        
        return {
            "total_conversations": len(self.conversation_history),
            "session_id": self.session_id,
            "session_duration": f"{duration_minutes} minutes",
            "session_start": self.session_start_time.isoformat(),
            "models_used": models_used,
            "total_characters": {
                "user": total_user_chars,
                "assistant": total_assistant_chars
            },
            "files_discussed": list(files_discussed),
            "tools_used": list(tools_used),
            "latest_conversation": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def export_conversation_history(self, format: str = "json") -> str:
        """
        Export conversation history in specified format.
        
        Args:
            format: "json" or "markdown"
            
        Returns:
            Formatted conversation history
        """
        if format == "json":
            return json.dumps({
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.session_start_time.isoformat(),
                    "export_time": datetime.now().isoformat()
                },
                "conversation_history": self.conversation_history,
                "summary": self.get_conversation_summary()
            }, indent=2)
        
        elif format == "markdown":
            lines = []
            lines.append(f"# Conversation Export - {self.session_id}")
            lines.append(f"**Session Start:** {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"**Export Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            summary = self.get_conversation_summary()
            lines.append("## Session Summary")
            lines.append(f"- **Total Conversations:** {summary['total_conversations']}")
            lines.append(f"- **Models Used:** {', '.join(summary['models_used'])}")
            lines.append(f"- **Files Discussed:** {', '.join(summary['files_discussed']) if summary['files_discussed'] else 'None'}")
            lines.append(f"- **Tools Used:** {', '.join(summary['tools_used']) if summary['tools_used'] else 'None'}")
            lines.append("")
            
            lines.append("## Conversation History")
            
            for conv in self.conversation_history:
                lines.append(f"### Turn {conv['turn_number']} - {conv['timestamp'][:19]}")
                lines.append(f"**Model:** {conv['model_used']}")
                lines.append("")
                lines.append("**User:**")
                lines.append(conv['user_message'])
                lines.append("")
                lines.append("**Assistant:**")
                lines.append(conv['assistant_response'])
                
                if conv['tool_calls']:
                    lines.append("")
                    lines.append("**Tools Used:**")
                    for tool_call in conv['tool_calls']:
                        lines.append(f"- {tool_call.get('tool', 'unknown')}")
                
                lines.append("")
                lines.append("---")
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError("Format must be 'json' or 'markdown'")
    
    # Existing methods remain unchanged
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all stored context data."""
        return self.context_store.copy()
    
    def get_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """Get context data filtered by type."""
        return [
            entry for entry in self.context_store 
            if entry["type"] == content_type
        ]
    
    def get_files(self) -> List[Dict[str, Any]]:
        """Get all file-content entries."""
        return self.get_by_type("file-content")
    
    def remove_file(self, filename: str) -> bool:
        """Remove a specific file from context."""
        original_length = len(self.context_store)
        self.context_store = [
            entry for entry in self.context_store 
            if not (entry["type"] == "file-content" and 
                   entry["data"].get("filename") == filename)
        ]
        return len(self.context_store) < original_length
    
    def clear(self) -> None:
        """Clear all stored context data and conversation history."""
        self.context_store.clear()
        self.conversation_history.clear()
        # Start new session
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
    
    def clear_conversation_only(self) -> None:
        """Clear only conversation history, keep file context."""
        self.conversation_history.clear()
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
    
    def clear_by_type(self, content_type: str) -> None:
        """Clear all context data of a specific type."""
        self.context_store = [
            entry for entry in self.context_store 
            if entry["type"] != content_type
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of stored context data including conversation stats."""
        summary = {
            "total_entries": len(self.context_store),
            "by_type": {},
            "files": [],
            "conversation_summary": self.get_conversation_summary()
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
                    "columns": len(file_data.get("features", [])) if isinstance(file_data.get("features", []), list) else len(file_data.get("features", {})),
                    "file_type": file_data.get("file_type"),
                    "timestamp": entry["timestamp"]
                })
        
        return summary