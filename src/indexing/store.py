"""
Module: faiss_storage.py
Functionality: Enhanced FAISS vector storage with improved metadata handling.
"""
import os
import json
import pickle
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """Enhanced FAISS vector store with better metadata and error handling."""
    
    def __init__(self, index_name: str = "policy-index", storage_dir: str = "faiss_storage"):
        self.index_name = index_name
        self.storage_dir = storage_dir
        self.dimension = 1024
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        self.metadata_path = os.path.join(storage_dir, f"{index_name}_metadata.json")
        self.id_map_path = os.path.join(storage_dir, f"{index_name}_id_map.pkl")
        
        # Initialize or load components
        self.index = None
        self.metadata = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new one."""
        try:
            if os.path.exists(self.index_path):
                # Load FAISS index
                print(f"🔍 Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                # Verify dimension
                if self.index.d != self.dimension:
                    print(f"⚠️ Index dimension mismatch: {self.index.d} vs {self.dimension}")
                    self._create_new_index()
                    return
                
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.metadata)} chunks")
            
            # Load ID mapping
            if os.path.exists(self.id_map_path):
                with open(self.id_map_path, 'rb') as f:
                    id_map_data = pickle.load(f)
                    self.id_to_idx = id_map_data.get('id_to_idx', {})
                    self.idx_to_id = id_map_data.get('idx_to_id', {})
                    self.next_idx = id_map_data.get('next_idx', 0)
                print(f"✅ Loaded ID mappings for {len(self.id_to_idx)} chunks")
            
        except Exception as e:
            print(f"❌ Error loading FAISS components: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        print(f"🆕 Creating new FAISS index with {self.dimension} dimensions")
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
    
    def add_vectors(self, vectors: np.ndarray, chunk_data: List[Dict[str, Any]]) -> bool:
        """
        Add vectors with associated chunk data to the index.
        
        Args:
            vectors: Numpy array of vectors to add
            chunk_data: List of dictionaries containing chunk metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(vectors) != len(chunk_data):
                print(f"❌ Vector count ({len(vectors)}) doesn't match chunk data count ({len(chunk_data)})")
                return False
            
            # Normalize vectors for cosine similarity
            normalized_vectors = vectors.copy()
            norms = np.linalg.norm(normalized_vectors, axis=1, keepdims=True)
            normalized_vectors = np.divide(normalized_vectors, norms, 
                                         out=np.zeros_like(normalized_vectors), 
                                         where=norms!=0)
            
            # Add to FAISS index
            start_idx = self.next_idx
            self.index.add(normalized_vectors.astype(np.float32))
            
            # Update mappings and metadata
            for i, chunk in enumerate(chunk_data):
                idx = start_idx + i
                chunk_id = chunk.get('chunk_id', f"chunk_{idx}")
                
                # Store mappings
                self.id_to_idx[chunk_id] = idx
                self.idx_to_id[idx] = chunk_id
                
                # Store metadata with enhanced information
                metadata = {
                    'chunk_id': chunk_id,
                    'document_name': chunk.get('document_name', 'unknown'),
                    'page_number': chunk.get('page_number', 0),
                    'content': chunk.get('content', '')[:2000],  # Store first 2000 chars
                    'content_length': len(chunk.get('content', '')),
                    'section': chunk.get('section', 'content'),
                    'chunk_index': chunk.get('chunk_index', i),
                    'source_document': chunk.get('source_document', chunk.get('document_name', 'unknown')),
                    'content_hash': hashlib.md5(chunk.get('content', '').encode()).hexdigest(),
                    'timestamp': time.time()
                }
                
                # Add any additional metadata from chunk
                for key, value in chunk.items():
                    if key not in metadata and key != 'content':
                        metadata[key] = value
                
                self.metadata[chunk_id] = metadata
            
            self.next_idx += len(chunk_data)
            
            print(f"✅ Added {len(chunk_data)} vectors to FAISS index (total: {self.index.ntotal})")
            return True
            
        except Exception as e:
            print(f"❌ Error adding vectors to FAISS: {e}")
            return False
    
    def query(self, query_vector: List[float], top_k: int = 20, 
              include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query the FAISS index for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Dictionary with matches and metadata
        """
        try:
            if self.index.ntotal == 0:
                print("❌ No vectors in FAISS index")
                return {'matches': [], 'total_vectors': 0}
            
            # Normalize query vector for cosine similarity
            query_array = np.array([query_vector], dtype=np.float32)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            
            # Search FAISS index
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            # Format results
            matches = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                chunk_id = self.idx_to_id.get(idx, f"unknown_{idx}")
                
                match_data = {
                    'id': chunk_id,
                    'score': float(score),
                    'similarity_score': float(score),  # For compatibility
                    'index': int(idx)
                }
                
                if include_metadata:
                    metadata = self.metadata.get(chunk_id, {})
                    match_data['metadata'] = metadata
                    match_data['content'] = metadata.get('content', '')
                    match_data['document_name'] = metadata.get('document_name', 'unknown')
                    match_data['page_number'] = metadata.get('page_number', 0)
                
                matches.append(match_data)
            
            print(f"✅ Found {len(matches)} matches from FAISS index")
            
            return {
                'matches': matches,
                'total_vectors': self.index.ntotal,
                'query_processed': True
            }
            
        except Exception as e:
            print(f"❌ Error querying FAISS index: {e}")
            return {'matches': [], 'total_vectors': 0, 'error': str(e)}
    
    def save(self):
        """Save all components to disk."""
        try:
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Save ID mappings
            id_map_data = {
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'next_idx': self.next_idx
            }
            with open(self.id_map_path, 'wb') as f:
                pickle.dump(id_map_data, f)
            
            print(f"💾 Saved FAISS index and metadata")
            
        except Exception as e:
            print(f"❌ Error saving FAISS components: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Count documents
            documents = set()
            content_types = {}
            
            for metadata in self.metadata.values():
                doc_name = metadata.get('document_name', 'unknown')
                documents.add(doc_name)
                
                content_type = metadata.get('content_type', 'text')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                'total_vectors': self.index.ntotal if self.index else 0,
                'total_chunks': len(self.metadata),
                'unique_documents': len(documents),
                'document_names': list(documents),
                'content_types': content_types,
                'dimension': self.dimension,
                'index_exists': self.index is not None,
                'storage_path': self.index_path
            }
            
        except Exception as e:
            return {
                'total_vectors': 0,
                'error': str(e),
                'index_exists': False
            }
    
    def clear(self) -> int:
        """Clear all data from the vector store."""
        try:
            total_vectors = self.index.ntotal if self.index else 0
            
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
            self.next_idx = 0
            
            # Remove files
            for path in [self.index_path, self.metadata_path, self.id_map_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            print(f"🗑️ Cleared FAISS index ({total_vectors} vectors removed)")
            return total_vectors
            
        except Exception as e:
            print(f"❌ Error clearing FAISS index: {e}")
            return 0


def check_or_create_faiss_index(index_name: str = "policy-index", 
                              required_dimension: int = 1024,
                              storage_dir: str = "faiss_storage") -> bool:
    """
    Check if FAISS index exists with correct dimensions, create if needed.
    
    Args:
        index_name: Name of the FAISS index
        required_dimension: Required vector dimension
        storage_dir: Storage directory for FAISS files
        
    Returns:
        True if index is ready, False otherwise
    """
    try:
        os.makedirs(storage_dir, exist_ok=True)
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                if index.d == required_dimension:
                    print(f"✅ FAISS index '{index_name}' already exists with correct {required_dimension} dimensions")
                    return True
                else:
                    print(f"⚠️ FAISS index dimension mismatch: {index.d} vs {required_dimension}, recreating...")
                    os.remove(index_path)
            except Exception as e:
                print(f"⚠️ Error reading existing index: {e}, creating new one...")
                if os.path.exists(index_path):
                    os.remove(index_path)
        
        # Create new index
        index = faiss.IndexFlatIP(required_dimension)
        faiss.write_index(index, index_path)
        print(f"✅ Created new FAISS index '{index_name}' with {required_dimension} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        return False
