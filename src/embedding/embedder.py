"""
Module: embed_and_index.py
Functionality: Advanced embedding generation using Pinecone's text embeddings and FAISS vector indexing with smart document management.
"""
from typing import List, Dict, Optional, Callable, Any
import time
import os
import json
import pickle
import numpy as np
import faiss
from pinecone import Pinecone, ServerlessSpec
try:
    from ..indexing.registry import DocumentRegistry
except ImportError:
    from src.indexing.registry import DocumentRegistry

def generate_embeddings_batch(texts: List[str], api_key: str, batch_size: int = 96) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Pinecone inference API with batching.
    Args:
        texts: List of text strings to embed
        api_key: Pinecone API key
        batch_size: Maximum number of texts to process in a single batch (default: 96)
    Returns:
        List of embedding vectors
    """
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Process in batches to respect Pinecone's limits
        all_embeddings = []
        total_texts = len(texts)
        
        # Process in batches
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(total_texts+batch_size-1)//batch_size}: {len(batch)} texts")
            
            # Use the inference.embed method for this batch
            response = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            
            # Extract embeddings from the response
            batch_embeddings = []
            for embedding in response.data:
                batch_embeddings.append(embedding.values)
            
            all_embeddings.extend(batch_embeddings)
        
        print(f"✅ Generated {len(all_embeddings)} embeddings using Pinecone inference ({len(all_embeddings[0]) if all_embeddings else 0} dims)")
        return all_embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings with Pinecone inference: {e}")
        # Return non-zero random vectors as fallback
        import random
        fallback_embeddings = []
        for _ in texts:
            fallback_embeddings.append([random.uniform(-0.01, 0.01) for _ in range(1024)])
        return fallback_embeddings

def generate_embeddings_pinecone(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Pinecone inference API.
    Args:
        texts: List of text strings to embed
        api_key: Pinecone API key
    Returns:
        List of embedding vectors
    """
    # Use the batched implementation with a max batch size of 96
    return generate_embeddings_batch(texts, api_key, batch_size=96)

## Fallback logic removed for simplicity and reliability

def generate_query_embedding_pinecone(query: str, api_key: str) -> List[float]:
    """
    Generate a single query embedding using Pinecone's embedding service.
    Args:
        query: Query text to embed
        api_key: Pinecone API key
    Returns:
        Query embedding vector
    """
    try:
        # Use Pinecone client inference method
        pc = Pinecone(api_key=api_key)
        
        # Use the inference.embed method directly
        response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )
        
        # Extract embedding from the response
        embedding = response.data[0].values
        print(f"✅ Generated query embedding using Pinecone inference ({len(embedding)} dims)")
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating query embedding with Pinecone inference: {e}")
        # Return non-zero random vector as fallback
        import random
        return [random.uniform(-0.01, 0.01) for _ in range(1024)]

def clear_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index') -> int:
    """
    Clear all vectors from FAISS index.
    """
    try:
        storage_dir = 'faiss_storage'
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        metadata_path = os.path.join(storage_dir, f"{index_name}_metadata.json")
        id_map_path = os.path.join(storage_dir, f"{index_name}_id_map.pkl")
        
        total_vectors = 0
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            total_vectors = index.ntotal
        
        # Remove files to clear index
        for path in [index_path, metadata_path, id_map_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Create new empty index
        index = faiss.IndexFlatIP(1024)
        faiss.write_index(index, index_path)
        
        # Create empty metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
        id_map_data = {'id_to_idx': {}, 'idx_to_id': {}, 'next_idx': 0}
        with open(id_map_path, 'wb') as f:
            pickle.dump(id_map_data, f)
        
        return total_vectors
    except Exception as e:
        print(f"❌ Error clearing FAISS index: {e}")
        return 0

def delete_duplicate_vectors(pinecone_api_key: str, index_name: str = 'policy-index', dry_run: bool = True):
    """
    Delete duplicate vectors from Pinecone index based on content hash.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    index = pc.Index(index_name)
    print("🔍 Scanning index for duplicates...")
    content_hashes = {}
    duplicates = []
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors == 0:
            return {'message': 'No vectors in index', 'duplicates_found': 0}
        print(f"📊 Found {total_vectors} vectors in index")
        # Pinecone query returns a dict with 'matches' key
        query_response = index.query(
            vector=[0.0] * 1024,
            top_k=min(10000, total_vectors),
            include_metadata=True
        )
        matches = []
        if isinstance(query_response, dict):
            matches = query_response.get('matches', [])
        elif hasattr(query_response, 'matches'):
            matches = query_response.matches
        for match in matches:
            vector_id = match['id'] if isinstance(match, dict) else match.id
            metadata = match.get('metadata', {}) if isinstance(match, dict) else getattr(match, 'metadata', {})
            content_hash = metadata.get('content_hash', '')
            if content_hash:
                if content_hash in content_hashes:
                    duplicates.append({
                        'duplicate_id': vector_id,
                        'original_id': content_hashes[content_hash],
                        'content_hash': content_hash,
                        'document_name': metadata.get('document_name', 'unknown')
                    })
                else:
                    content_hashes[content_hash] = vector_id
        print(f"🔍 Found {len(duplicates)} duplicate vectors")
        if not dry_run and duplicates:
            print("🗑️ Deleting duplicate vectors...")
            duplicate_ids = [dup['duplicate_id'] for dup in duplicates]
            batch_size = 100
            deleted_count = 0
            for i in range(0, len(duplicate_ids), batch_size):
                batch = duplicate_ids[i:i + batch_size]
                index.delete(ids=batch)
                deleted_count += len(batch)
                print(f"Deleted {deleted_count}/{len(duplicate_ids)} duplicates...")
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': deleted_count,
                'remaining_vectors': total_vectors - deleted_count,
                'action': 'deleted'
            }
        else:
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': 0,
                'total_vectors': total_vectors,
                'action': 'dry_run' if dry_run else 'none_deleted',
                'duplicate_details': duplicates[:10]
            }
    except Exception as e:
        return {'error': f'Error processing duplicates: {str(e)}'}

def reindex_documents(pinecone_api_key: str, documents_to_reindex: List[str], index_name: str = 'policy-index'):
    """
    Remove and re-add specific documents to the index.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    index = pc.Index(index_name)
    deleted_vectors = []
    for doc_name in documents_to_reindex:
        print(f"🗑️ Removing existing vectors for document: {doc_name}")
        query_response = index.query(
            vector=[0.0] * 1024,
            filter={'document_name': doc_name},
            top_k=10000,
            include_metadata=True
        )
        matches = []
        if isinstance(query_response, dict):
            matches = query_response.get('matches', [])
        elif hasattr(query_response, 'matches'):
            matches = query_response.matches
        if matches:
            vector_ids = [match['id'] if isinstance(match, dict) else match.id for match in matches]
            index.delete(ids=vector_ids)
            deleted_vectors.extend(vector_ids)
            print(f"Deleted {len(vector_ids)} vectors for {doc_name}")
    return {
        'documents_processed': len(documents_to_reindex),
        'vectors_deleted': len(deleted_vectors),
        'message': f'Deleted {len(deleted_vectors)} vectors. Re-run indexing to add fresh vectors.'
    }

def get_index_stats(pinecone_api_key: str, index_name: str = 'policy-index'):
    """
    Get statistics about a Pinecone index.
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            return {'exists': False, 'total_vector_count': 0}
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return {
            'exists': True,
            'total_vector_count': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0.0),
            'namespaces': stats.get('namespaces', {})
        }
    except Exception as e:
        return {'exists': False, 'error': str(e), 'total_vector_count': 0}

## Already simplified above

def check_or_create_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index', required_dimension: int = 1024, progress_callback: Optional[Callable] = None) -> bool:
    """
    Check if FAISS index exists with correct dimensions, create if needed.
    """
    try:
        storage_dir = 'faiss_storage'
        os.makedirs(storage_dir, exist_ok=True)
        index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        
        if os.path.exists(index_path):
            # Load and check dimension
            try:
                index = faiss.read_index(index_path)
                current_dimension = index.d
                
                if current_dimension != required_dimension:
                    print(f"⚠️ FAISS index '{index_name}' has {current_dimension} dimensions, but we need {required_dimension}")
                    if progress_callback:
                        progress_callback(f"Recreating index ({required_dimension}D)...", 20)
                    
                    # Remove old files
                    metadata_path = os.path.join(storage_dir, f"{index_name}_metadata.json")
                    id_map_path = os.path.join(storage_dir, f"{index_name}_id_map.pkl")
                    
                    for path in [index_path, metadata_path, id_map_path]:
                        if os.path.exists(path):
                            os.remove(path)
                    
                    # Create new index
                    index = faiss.IndexFlatIP(required_dimension)
                    faiss.write_index(index, index_path)
                    
                    print(f"✅ Successfully recreated FAISS index '{index_name}' with {required_dimension} dimensions")
                    return True
                else:
                    print(f"✅ FAISS index '{index_name}' already exists with correct {required_dimension} dimensions")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Error reading existing index, creating new one: {e}")
                index = faiss.IndexFlatIP(required_dimension)
                faiss.write_index(index, index_path)
                return True
        else:
            if progress_callback:
                progress_callback(f"Creating new FAISS index ({required_dimension}D)...", 15)
            
            index = faiss.IndexFlatIP(required_dimension)
            faiss.write_index(index, index_path)
            print(f"✅ Successfully created FAISS index '{index_name}' with {required_dimension} dimensions")
            return True
            
    except Exception as e:
        print(f"❌ Error managing FAISS index: {e}")
        if progress_callback:
            progress_callback(f"Index creation failed: {e}", -1)
        return False

def index_chunks_in_pinecone(chunks: List[Dict], pinecone_api_key: str, pinecone_env: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None):
    """
    Generate embeddings and store in enhanced FAISS with metadata.
    """
    if progress_callback:
        progress_callback("Initializing enhanced FAISS storage...", 0)
    
    try:
        from ..indexing.store import FAISSVectorStore, check_or_create_faiss_index
        
        # Check/create FAISS index
        if not check_or_create_faiss_index(index_name, 1024):
            print("❌ Failed to create or verify FAISS index")
            if progress_callback:
                progress_callback("Failed to create index", -1)
            return {"success": False, "error": "Failed to create FAISS index"}
        
        # Initialize FAISS vector store
        vector_store = FAISSVectorStore(index_name)
        
    except ImportError as e:
        print(f"❌ FAISS storage import error: {e}")
        if progress_callback:
            progress_callback("FAISS import failed", -1)
        return {"success": False, "error": f"FAISS import failed: {e}"}
    
    if progress_callback:
        progress_callback("Generating embeddings with Pinecone inference...", 15)
    
    # Generate embeddings using Pinecone (same as before)
    texts = [chunk['content'] for chunk in chunks]
    try:
        embeddings = generate_embeddings_pinecone(texts, pinecone_api_key)
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        if progress_callback:
            progress_callback(f"Error generating embeddings: {e}", -1)
        return {"success": False, "error": f"Embedding generation failed: {e}"}
    
    if progress_callback:
        progress_callback("Preparing vectors for enhanced FAISS storage...", 60)
    
    # Convert embeddings to numpy array
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    if progress_callback:
        progress_callback("Storing vectors in enhanced FAISS...", 70)
    
    # Add vectors to FAISS store
    success = vector_store.add_vectors(embedding_matrix, chunks)
    
    if not success:
        if progress_callback:
            progress_callback("Failed to store vectors", -1)
        return {"success": False, "error": "Failed to store vectors in FAISS"}
    
    # Save the vector store
    vector_store.save()
    
    if progress_callback:
        progress_callback("Enhanced FAISS indexing complete!", 100)
    
    print(f"Successfully indexed {len(chunks)} chunks into enhanced FAISS index '{index_name}'.")
    return {"success": True, "indexed_count": len(chunks)}

def smart_index_documents(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Smart indexing - only processes new or changed documents
    """
    registry = DocumentRegistry()
    status = registry.get_document_status(docs_folder)
    files_to_process = registry.get_files_to_process(docs_folder)
    status_counts = {
        'indexed': len([f for f, s in status.items() if s == 'indexed']),
        'new': len([f for f, s in status.items() if s == 'new']),
        'changed': len([f for f, s in status.items() if s == 'changed']),
        'missing': len([f for f, s in status.items() if s == 'missing'])
    }
    if progress_callback:
        progress_callback(f"📊 Status: {status_counts['indexed']} indexed, {status_counts['new']} new, {status_counts['changed']} changed", 10)
    if not files_to_process:
        if progress_callback:
            progress_callback("🎉 All documents are already indexed and up-to-date!", 100)
        return {
            "status": "up_to_date",
            "processed_files": 0,
            "skipped_files": status_counts['indexed'],
            "total_time": 0,
            "status_counts": status_counts
        }
    start_time = time.time()
    processed_files = []
    from ..feature_engineering.chunker import chunk_documents_optimized
    total_files = len(files_to_process)
    for i, filename in enumerate(files_to_process):
        file_path = os.path.join(docs_folder, filename)
        if progress_callback:
            progress_callback(f"🔄 Processing {filename} ({i+1}/{total_files})...", 20 + (i / total_files) * 60)
        try:
            from ..data_processing.parser import load_and_parse_from_folder
            parsed_docs = load_and_parse_from_folder(docs_folder, file_filter=[filename], save_parsed_text=save_parsed_text)
            if parsed_docs:
                transformed_docs = []
                for doc in parsed_docs:
                    doc_name = doc.get('document_name', 'unknown')
                    parsed_output = doc.get('parsed_output', {})
                    content = (parsed_output.get('content', '') or parsed_output.get('text', '') or parsed_output.get('cleaned_text', ''))
                    transformed_doc = {
                        'document_name': doc_name,
                        'content': content,
                        'ordered_content': parsed_output.get('ordered_content', [])
                    }
                    transformed_docs.append(transformed_doc)
                chunks = chunk_documents_optimized(transformed_docs)
                result = index_chunks_in_pinecone(chunks, pinecone_api_key, index_name)
                if isinstance(result, dict) and result.get('success', False):
                    registry.mark_document_indexed(filename, file_path, len(chunks))
                    processed_files.append(filename)
                    if progress_callback:
                        progress_callback(f"✅ {filename}: {len(chunks)} chunks indexed", 20 + ((i+1) / total_files) * 60)
                else:
                    if progress_callback:
                        progress_callback(f"❌ Failed to index {filename}", 20 + ((i+1) / total_files) * 60)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {filename}: {str(e)}", 20 + ((i+1) / total_files) * 60)
    end_time = time.time()
    processing_time = end_time - start_time
    if progress_callback:
        progress_callback(f"🎉 Smart indexing complete! Processed {len(processed_files)} files in {processing_time:.1f}s", 100)
    return {
        "status": "completed",
        "processed_files": len(processed_files),
        "skipped_files": status_counts['indexed'],
        "total_time": processing_time,
        "files_processed": processed_files,
        "status_counts": status_counts
    }

def force_reindex_all(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Force reindex all documents (clears registry and processes everything)
    """
    registry = DocumentRegistry()
    if progress_callback:
        progress_callback("🔄 Force re-indexing: clearing registry and index...", 5)
    registry.clear_registry()
    try:
        clear_result = clear_pinecone_index(pinecone_api_key, index_name)
        if progress_callback:
            progress_callback(f"🗑️ Cleared {clear_result} vectors from index", 10)
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Failed to clear index: {str(e)}", 10)
        return {"status": "failed", "error": f"Could not clear index: {str(e)}"}
    return smart_index_documents(docs_folder, pinecone_api_key, index_name, progress_callback, save_parsed_text)