"""
Document Registry - Track indexed documents to avoid reprocessing
"""
import os
import json
import hashlib
from typing import Dict, List, Set, Optional
from datetime import datetime

class DocumentRegistry:
    """Manages registry of indexed documents to prevent unnecessary reprocessing"""
    
    def __init__(self, registry_file: str = "indexed_documents.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing document registry"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save document registry to file"""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving registry: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"⚠️ Error hashing file {file_path}: {e}")
            return ""
    
    def get_document_status(self, docs_folder: str) -> Dict[str, str]:
        """
        Check status of all documents in folder
        Returns: {'filename': 'status'} where status is:
        - 'indexed': Already indexed, no changes
        - 'changed': File modified since last indexing
        - 'new': New file, not indexed
        - 'missing': Was indexed but file no longer exists
        """
        current_files = {}
        status = {}
        
        # Get current files and their hashes
        if os.path.exists(docs_folder):
            for filename in os.listdir(docs_folder):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(docs_folder, filename)
                    current_files[filename] = self._get_file_hash(file_path)
        
        # Check status of current files
        for filename, current_hash in current_files.items():
            if filename in self.registry:
                if self.registry[filename]['hash'] == current_hash:
                    status[filename] = 'indexed'
                else:
                    status[filename] = 'changed'
            else:
                status[filename] = 'new'
        
        # Check for missing files (were indexed but no longer exist)
        for filename in self.registry:
            if filename not in current_files:
                status[filename] = 'missing'
        
        return status
    
    def get_files_to_process(self, docs_folder: str) -> List[str]:
        """Get list of files that need processing (new or changed)"""
        status = self.get_document_status(docs_folder)
        return [filename for filename, stat in status.items() 
                if stat in ['new', 'changed']]
    
    def mark_document_indexed(
        self,
        filename: str,
        file_path: str,
        chunk_count: int,
        embedding_model: str = "unknown",
        embedding_dimension: int = 0,
    ):
        """Mark a document as successfully indexed, recording the embedding model used."""
        file_hash = self._get_file_hash(file_path)
        self.registry[filename] = {
            'hash':                file_hash,
            'indexed_at':          datetime.now().isoformat(),
            'chunk_count':         chunk_count,
            'file_size':           os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            # ── LLMOps: index lifecycle tracking ──────────────────────────────
            'embedding_model':     embedding_model,
            'embedding_dimension': embedding_dimension,
        }
        self._save_registry()

    def needs_reindex(
        self,
        filename: str,
        current_embedding_model: str,
        current_dimension: int = 0,
    ) -> bool:
        """
        Return True if the document must be re-embedded because the embedding
        model or dimension has changed since it was last indexed.

        LLMOps: vectors built with an old model are incompatible with a new one.
        """
        entry = self.registry.get(filename)
        if not entry:
            return True  # Never indexed
        stored_model = entry.get("embedding_model", "")
        stored_dim   = entry.get("embedding_dimension", 0)
        if stored_model and stored_model != current_embedding_model:
            print(
                f"⚠️  [Registry] Embedding model changed "
                f"({stored_model} → {current_embedding_model}). "
                f"Re-indexing '{filename}'."
            )
            return True
        if current_dimension and stored_dim and stored_dim != current_dimension:
            print(
                f"⚠️  [Registry] Embedding dimension changed "
                f"({stored_dim} → {current_dimension}). "
                f"Re-indexing '{filename}'."
            )
            return True
        return False

    
    def remove_document(self, filename: str):
        """Remove document from registry"""
        if filename in self.registry:
            del self.registry[filename]
            self._save_registry()
    
    def get_registry_summary(self) -> Dict:
        """Get summary of indexed documents"""
        if not self.registry:
            return {"total_documents": 0, "total_chunks": 0}
        
        total_chunks = sum(doc.get('chunk_count', 0) for doc in self.registry.values())
        return {
            "total_documents": len(self.registry),
            "total_chunks": total_chunks,
            "documents": self.registry
        }
    
    def clear_registry(self):
        """Clear all registry entries"""
        self.registry = {}
        self._save_registry()
