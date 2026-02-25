"""
Single-click pipeline for complete document processing.
Combines parsing, chunking, embedding, and indexing into one streamlined operation.
Also provides query functionality for the complete RAG system.
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .data_processing.parser import parse_document_hybrid
from .feature_engineering.chunker import chunk_documents_optimized
from .embedding.embedder import index_chunks_in_pinecone
from .indexing.registry import DocumentRegistry


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    chunk_size: int = 800
    chunk_overlap: int = 150
    use_semantic: bool = True
    save_parsed_text: bool = False
    index_name: str = "policy-index"
    pinecone_env: str = "us-east-1"  # Kept for compatibility
    output_dir: str = "results"


class DocumentPipeline:
    """Complete document processing pipeline."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = DocumentRegistry()
    
    def _create_error_result(self, error_msg: str, step: str, start_time: float) -> Dict[str, Any]:
        """Create standardized error result with all required fields."""
        total_time = time.time() - start_time
        return {
            "success": False,
            "error": error_msg,
            "step": step,
            "statistics": {
                "total_files": 0,
                "processed_files": 0,
                "skipped_files": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "indexed_vectors": 0,
                "total_characters": 0,
                "processing_speed": "N/A",
            },
            "timing": {
                "parsing_time": "0.00s",
                "chunking_time": "0.00s", 
                "indexing_time": "0.00s",
                "total_time": f"{total_time:.2f}s"
            },
            "files": {
                "processed": [],
                "skipped": []
            },
            "processing_time": total_time
        }
    
    async def process_all_documents(self, 
                            docs_dir: str = "docs", 
                            pinecone_api_key: Optional[str] = None,
                            force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Complete single-click pipeline for all documents.
        
        Args:
            docs_dir: Directory containing PDF documents
            pinecone_api_key: Pinecone API key for indexing
            force_reprocess: Whether to reprocess already processed documents
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate inputs
            if not os.path.exists(docs_dir):
                return self._create_error_result(
                    f"Directory '{docs_dir}' not found!", 
                    "validation", 
                    start_time
                )
            
            if not pinecone_api_key:
                return self._create_error_result(
                    "Pinecone API key is required!", 
                    "validation", 
                    start_time
                )
            
            # Step 2: Find PDF files
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return self._create_error_result(
                    f"No PDF files found in '{docs_dir}' directory!", 
                    "file_discovery", 
                    start_time
                )
            
            # Step 3: Filter files based on registry (unless force reprocess)
            files_to_process = []
            skipped_files = []
            
            if force_reprocess:
                files_to_process = [os.path.join(docs_dir, f) for f in pdf_files]
            else:
                # Check registry status
                doc_status = self.registry.get_document_status(docs_dir)
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(docs_dir, pdf_file)
                    status = doc_status.get(pdf_file, 'new')
                    
                    if status in ['new', 'changed']:
                        files_to_process.append(pdf_path)
                    else:
                        skipped_files.append(pdf_file)
            
            if not files_to_process and not force_reprocess:
                return {
                    "success": True,
                    "message": "All documents already processed. Use force_reprocess=True to reprocess.",
                    "statistics": {
                        "total_files": len(pdf_files),
                        "processed_files": 0,
                        "skipped_files": len(skipped_files),
                        "total_documents": 0,
                        "total_chunks": 0,
                        "indexed_vectors": 0,
                        "total_characters": 0,
                        "processing_speed": "N/A",
                    },
                    "timing": {
                        "parsing_time": "0.00s",
                        "chunking_time": "0.00s", 
                        "indexing_time": "0.00s",
                        "total_time": f"{time.time() - start_time:.2f}s"
                    },
                    "files": {
                        "processed": [],
                        "skipped": skipped_files
                    },
                    "step": "registry_check",
                    "processing_time": time.time() - start_time
                }
            
            # Step 4: Parse documents (async)
            parsing_start = time.time()
            parsed_docs = await self._parse_documents_async(files_to_process)
            
            if not parsed_docs:
                return self._create_error_result(
                    "Failed to parse any documents!", 
                    "parsing", 
                    start_time
                )
            
            parsing_time = time.time() - parsing_start
            
            # Step 5: Chunk documents using the existing chunk_documents function (async)
            chunking_start = time.time()
            
            all_chunks = await self._chunk_documents_async(parsed_docs)
            
            chunking_time = time.time() - chunking_start
            
            if not all_chunks:
                return self._create_error_result(
                    "Failed to create any chunks!", 
                    "chunking", 
                    start_time
                )
            
            # Step 6: Generate embeddings and index (async)
            indexing_start = time.time()
            
            indexing_result = await self._index_documents_async(all_chunks, pinecone_api_key)
            
            indexing_time = time.time() - indexing_start
            
            if not indexing_result.get("success", False):
                return self._create_error_result(
                    f"Indexing failed: {indexing_result.get('error', 'Unknown error')}", 
                    "indexing", 
                    start_time
                )
            
            # Step 7: Update registry for processed files (async)
            await self._update_registry_async(files_to_process, all_chunks)
            
            # Calculate statistics
            total_time = time.time() - start_time
            total_chars = sum(len(doc.get('content', '')) for doc in parsed_docs)
            
            return {
                "success": True,
                "statistics": {
                    "total_files": len(pdf_files),
                    "processed_files": len(files_to_process),
                    "skipped_files": len(skipped_files),
                    "total_documents": len(parsed_docs),
                    "total_chunks": len(all_chunks),
                    "indexed_vectors": indexing_result.get("indexed_count", len(all_chunks)),
                    "total_characters": total_chars,
                    "processing_speed": f"{total_chars / total_time:.0f} chars/sec",
                },
                "timing": {
                    "parsing_time": f"{parsing_time:.2f}s",
                    "chunking_time": f"{chunking_time:.2f}s", 
                    "indexing_time": f"{indexing_time:.2f}s",
                    "total_time": f"{total_time:.2f}s"
                },
                "files": {
                    "processed": [os.path.basename(f) for f in files_to_process],
                    "skipped": skipped_files
                },
                "step": "completed"
            }
            
        except Exception as e:
            return self._create_error_result(
                f"Pipeline error: {str(e)}", 
                "unknown", 
                start_time
            )

    async def _parse_documents_async(self, files_to_process: List[str]) -> List[Dict]:
        """Async wrapper for document parsing."""
        # Run parsing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._parse_multiple_documents, files_to_process)
    
    def _parse_multiple_documents(self, file_paths: List[str]) -> List[Dict]:
        """Parse multiple documents using the hybrid parser."""
        results = []
        for pdf_path in file_paths:
            try:
                result = parse_document_hybrid(pdf_path, save_parsed_text=self.config.save_parsed_text)
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error parsing {pdf_path}: {e}")
                # Create error result
                results.append({
                    'document_name': os.path.basename(pdf_path),
                    'parsed_output': {'error': str(e)},
                    'metadata': {'status': 'error'}
                })
        return results
    
    async def _chunk_documents_async(self, parsed_docs: List[Dict]) -> List[Dict]:
        """Async wrapper for document chunking."""
        # Transform parsed_docs format to match what chunk_documents expects
        transformed_docs = []
        
        for doc in parsed_docs:
            doc_name = doc.get('document_name', 'unknown')
            
            # Check if this is direct output from parse_document_hybrid or wrapped format
            if 'parsed_output' in doc:
                # Old wrapped format
                parsed_output = doc.get('parsed_output', {})
                if isinstance(parsed_output, dict):
                    content = (parsed_output.get('content', '') or 
                              parsed_output.get('text', '') or 
                              parsed_output.get('cleaned_text', ''))
                    ordered_content = parsed_output.get('ordered_content', [])
                else:
                    content = str(parsed_output) if parsed_output else ''
                    ordered_content = []
            else:
                # Direct format from parse_document_hybrid
                content = doc.get('content', '')
                ordered_content = doc.get('ordered_content', [])
            
            # Skip documents with actual parsing errors or no content
            # Check for actual error conditions, not just the word "error" in content
            has_parsing_error = (
                doc.get('metadata', {}).get('error') or  # Check metadata for actual errors
                (isinstance(content, str) and content.strip().startswith('Error:')) or  # Content starts with "Error:"
                (isinstance(content, str) and 'parsing error' in content.lower()) or  # Explicit parsing errors
                (len(content.strip()) < 10)  # Content too short to be meaningful
            )
            
            if not content or content.strip() == '' or has_parsing_error:
                error_reason = "No content" if not content else "Parsing error detected"
                print(f"⚠️ Skipping document {doc_name}: {error_reason}")
                continue
            
            # Create the format expected by chunk_documents
            transformed_doc = {
                'document_name': doc_name,
                'content': content,
                'ordered_content': ordered_content
            }
            
            transformed_docs.append(transformed_doc)
        
        # Run chunking in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: chunk_documents_optimized(
                parsed_content=transformed_docs,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                save_parsed_text=self.config.save_parsed_text,
                output_dir=self.config.output_dir
            )
        )
    
    async def _index_documents_async(self, all_chunks: List[Dict], pinecone_api_key: str) -> Dict:
        """Async wrapper for document indexing using FAISS."""
        # Run indexing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: index_chunks_in_pinecone(
                chunks=all_chunks,
                pinecone_api_key=pinecone_api_key,
                pinecone_env=self.config.pinecone_env,
                index_name=self.config.index_name
            )
        )
    
    async def _update_registry_async(self, files_to_process: List[str], all_chunks: List[Dict]):
        """Async wrapper for registry updates."""
        loop = asyncio.get_event_loop()
        
        def update_registry():
            for file_path in files_to_process:
                filename = os.path.basename(file_path)
                # Count chunks for this specific document
                doc_chunks = [c for c in all_chunks if c.get('source_document', '').endswith(filename)]
                self.registry.mark_document_indexed(filename, file_path, len(doc_chunks))
        
        await loop.run_in_executor(None, update_registry)


# Convenience function for single-click operation
async def process_all_documents_pipeline(docs_dir: str = "docs",
                                 pinecone_api_key: Optional[str] = None,
                                 force_reprocess: bool = False,
                                 config: Optional[PipelineConfig] = None,
                                 index_name: str = "policy-index") -> Dict[str, Any]:
    """
    Single-click function to process all documents through the complete pipeline.

    Args:
        docs_dir: Directory containing PDF documents (default: "docs")
        pinecone_api_key: Pinecone API key for vector indexing
        force_reprocess: Whether to reprocess already processed documents
        config: Pipeline configuration (uses defaults if None)
        index_name: FAISS index name — use a per-request value for isolation

    Returns:
        Dictionary with processing results and detailed statistics
    """
    if config is None:
        config = PipelineConfig(index_name=index_name)
    else:
        config.index_name = index_name
    pipeline = DocumentPipeline(config)
    return await pipeline.process_all_documents(docs_dir, pinecone_api_key, force_reprocess)


# Alternative streamlined function for Streamlit
async def streamlit_single_click_pipeline(pinecone_api_key: Optional[str] = None,
                                  force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Streamlit-optimized single-click pipeline function.
    Uses sensible defaults and provides user-friendly output.
    
    Args:
        pinecone_api_key: Pinecone API key
        force_reprocess: Whether to reprocess all documents
        
    Returns:
        Dictionary with success status and user-friendly messages
    """
    # Use optimized config for Streamlit
    config = PipelineConfig(
        chunk_size=800,  # Optimal for retrieval
        chunk_overlap=150,
        use_semantic=True,
        save_parsed_text=False,
        index_name="policy-index",
        output_dir="results"
    )
    
    result = await process_all_documents_pipeline(
        docs_dir="docs",
        pinecone_api_key=pinecone_api_key,
        force_reprocess=force_reprocess,
        config=config
    )
    
    # Add user-friendly messages
    if result.get("success", False):
        stats = result.get("statistics", {})
        timing = result.get("timing", {})
        
        result["user_message"] = f"""
🎉 **Pipeline Complete!**

📊 **Processing Summary:**
- Processed {stats.get('processed_files', 0)} PDF files
- Created {stats.get('total_chunks', 0)} semantic chunks
- Indexed {stats.get('indexed_vectors', 0)} vectors in Pinecone
- Processing speed: {stats.get('processing_speed', 'N/A')}

⏱️ **Timing:**
- Parsing: {timing.get('parsing_time', 'N/A')}
- Chunking: {timing.get('chunking_time', 'N/A')}
- Indexing: {timing.get('indexing_time', 'N/A')}
- **Total: {timing.get('total_time', 'N/A')}**

✅ Your document search system is ready!
        """.strip()
        
        if stats.get('skipped_files', 0) > 0:
            result["user_message"] += f"\n\n📋 Skipped {stats.get('skipped_files', 0)} already processed files."
    
    else:
        result["user_message"] = f"❌ **Pipeline Failed**\n\nError in {result.get('step', 'unknown')} step:\n{result.get('error', 'Unknown error')}"
    
    return result


# Synchronous wrapper for Streamlit compatibility
def streamlit_single_click_pipeline_sync(pinecone_api_key: Optional[str] = None,
                                        force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Synchronous wrapper for Streamlit compatibility.
    Runs the async pipeline in a new event loop.
    
    Args:
        pinecone_api_key: Pinecone API key
        force_reprocess: Whether to reprocess all documents
        
    Returns:
        Dictionary with success status and user-friendly messages
    """
    try:
        # Create new event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async pipeline
        result = loop.run_until_complete(
            streamlit_single_click_pipeline(pinecone_api_key, force_reprocess)
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Pipeline execution error: {str(e)}",
            "step": "async_wrapper",
            "user_message": f"❌ **Pipeline Failed**\n\nAsync execution error:\n{str(e)}"
        }
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass


# Query functionality for complete RAG system
def query_documents_sync(
    query: str,
    pinecone_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    index_name: str = "policy-index",
    query_embedding: Optional[list] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous function to query processed documents.

    Args:
        query: The question to ask
        pinecone_api_key: Pinecone API key for vector search
        gemini_api_key: Gemini/Groq API key for LLM processing
        index_name: FAISS index name (use per-request value for isolation)
        query_embedding: Optional precomputed embedding for the query
        namespace: Passed through to process_query for API consistency

    Returns:
        Dictionary with query results and analysis
    """
    try:
        from .retrieval.searcher import FAISSQueryProcessor

        if not query or not query.strip():
            return {"success": False, "error": "Query cannot be empty", "query": query}

        if not pinecone_api_key:
            return {
                "success": False,
                "error": "Pinecone API key is required for embeddings (FAISS used for storage)",
                "query": query,
            }

        if not gemini_api_key:
            gemini_api_key = "dummy"

        processor = FAISSQueryProcessor(
            pinecone_api_key=pinecone_api_key,
            gemini_api_key=gemini_api_key,
            index_name=index_name,
        )

        # ❗ Pass namespace so per-request FAISS isolation flows end-to-end
        result = processor.process_query(query.strip(), namespace=namespace)
        result["success"] = result.get("status") == "success"
        return result

    except Exception as e:
        import traceback
        print(f"❌ query_documents_sync exception: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Query processing error: {str(e)}",
            "query": query,
            "status": "error",
        }


async def query_documents_batch_async(
    queries: List[str], 
    pinecone_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    index_name: str = "policy-index",
    query_embeddings: Optional[List[List[float]]] = None
) -> List[Dict[str, Any]]:
    """
    Asynchronous function to query multiple documents in parallel.
    
    Args:
        queries: List of questions to ask
        pinecone_api_key: Pinecone API key for vector search
        gemini_api_key: Gemini API key for LLM processing (optional, uses fallback if None)
        index_name: Name of the Pinecone index to query
        query_embeddings: Optional list of precomputed embeddings for each query
        
    Returns:
        List of dictionaries with query results and analysis in the same order as input
        
    Example:
        >>> results = await query_documents_batch_async(
        ...     ["What is covered?", "What are exclusions?"],
        ...     pinecone_api_key="your-pinecone-key",
        ...     gemini_api_key="your-gemini-key"
        ... )
        >>> for result in results:
        ...     print(result["evaluation"]["decision"])
    """
    try:
        from .retrieval.searcher import FAISSQueryProcessor

        # Validate inputs
        if not queries:
            return []
        
        # Validate that all queries are non-empty
        for i, query in enumerate(queries):
            if not query or not query.strip():
                return [{
                    "success": False,
                    "error": f"Query {i+1} cannot be empty",
                    "query": query,
                    "status": "error"
                }]

        if not pinecone_api_key:
            return [{
                "success": False,
                "error": "Pinecone API key is required for embeddings (FAISS used for storage)",
                "query": query,
                "status": "error"
            } for query in queries]

        # Use dummy key if Gemini API key not provided (fallback mode)
        if not gemini_api_key:
            gemini_api_key = "dummy"

        # Initialize FAISS query processor once for all queries
        processor = FAISSQueryProcessor(
            pinecone_api_key=pinecone_api_key,
            gemini_api_key=gemini_api_key,
            index_name=index_name
        )

        # Process queries in batch
        stripped_queries = [query.strip() for query in queries]
        results = await processor.process_queries_batch(stripped_queries, query_embeddings)
        
        # Add success flag to each result
        for result in results:
            result["success"] = result.get("status") == "success"

        return results

    except Exception as e:
        # Return error for all queries
        error_result = {
            "success": False,
            "error": f"Batch query processing error: {str(e)}",
            "status": "error"
        }
        return [dict(error_result, query=query) for query in queries]


def query_documents_batch_sync(
    queries: List[str], 
    pinecone_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    index_name: str = "policy-index",
    query_embeddings: Optional[List[List[float]]] = None
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for batch query processing.
    
    Args:
        queries: List of questions to ask
        pinecone_api_key: Pinecone API key for vector search
        gemini_api_key: Gemini API key for LLM processing (optional, uses fallback if None)
        index_name: Name of the Pinecone index to query
        query_embeddings: Optional list of precomputed embeddings for each query
        
    Returns:
        List of dictionaries with query results and analysis in the same order as input
    """
    import asyncio
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    query_documents_batch_async(queries, pinecone_api_key, gemini_api_key, index_name, query_embeddings)
                )
                return future.result()
        else:
            # We can run directly
            return loop.run_until_complete(
                query_documents_batch_async(queries, pinecone_api_key, gemini_api_key, index_name, query_embeddings)
            )
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(
            query_documents_batch_async(queries, pinecone_api_key, gemini_api_key, index_name, query_embeddings)
        )
