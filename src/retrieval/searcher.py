"""
Module: faiss_query_processor.py
Functionality: FAISS-based query processing that replaces Pinecone queries.
Keeps the same embedding and reranking logic but queries FAISS locally.
"""
import json
import re
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Suppress specific tokenizer warnings for BGE Reranker
warnings.filterwarnings(
    "ignore", 
    message=".*XLMRobertaTokenizerFast.*__call__.*method is faster.*", 
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*fast tokenizer.*__call__.*method is faster.*",
    category=UserWarning
)

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ── Load config.yaml (optional; falls back to defaults if not found) ─────────
import os as _os
import yaml as _yaml

def _load_cfg() -> dict:
    cfg_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "config.yaml")
    try:
        with open(cfg_path, "r") as _f:
            return _yaml.safe_load(_f) or {}
    except Exception:
        return {}

_CFG = _load_cfg()
_LLM_CFG = _CFG.get("llm", {})
_RET_CFG = _CFG.get("retrieval", {})

# FAISS storage and embeddings integration
from ..indexing.store import FAISSVectorStore, check_or_create_faiss_index
from ..embedding.embedder import generate_query_embedding_pinecone

# Inference stage — Groq LLM evaluation
from ..inference import GroqEvaluator, LLMAnswer


class FAISSQueryProcessor:
    """Query processor using FAISS for vector storage instead of Pinecone."""

    def __init__(self, pinecone_api_key: str, gemini_api_key: str = "unused",
                 index_name: str = "policy-index",
                 groq_api_key: Optional[str] = None):
        """
        Parameters
        ----------
        pinecone_api_key : Pinecone key for embeddings & BGE reranker.
        gemini_api_key   : Kept for backward-compat; ignored (Groq is used).
        index_name       : FAISS index name (must match config.yaml retrieval.index_name).
        groq_api_key     : Groq key. Falls back to GROQ_API_KEY env var if None.
        """
        import os
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.quota_exceeded = False
        self.fallback_reason = None

        # ── Resolve Groq key ──────────────────────────────────────────────
        self._groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = _LLM_CFG.get("model", "llama-3.3-70b-versatile")
        self._temperature = float(_LLM_CFG.get("temperature", 0.2))
        # max_tokens: None / null in config.yaml means no cap — do NOT cast None to int
        _cfg_max = _LLM_CFG.get("max_tokens", None)
        self._max_tokens = int(_cfg_max) if _cfg_max is not None else None
        self._max_retries = int(_LLM_CFG.get("max_retries", 2))
        # Adjacent chunk window — how many chunks around each top result to include in context
        self._adj_before = int(_RET_CFG.get("adjacent_chunks_before", 5))
        self._adj_after  = int(_RET_CFG.get("adjacent_chunks_after",  5))

        # ── Initialize FAISS vector store ─────────────────────────────────
        try:
            print("🔍 Checking/creating FAISS index with correct dimensions...")
            if check_or_create_faiss_index(index_name, 1024):
                self.vector_store = FAISSVectorStore(index_name)
                print("✅ Initialized FAISS vector store")
            else:
                print("❌ Failed to create/verify FAISS index")
                self.vector_store = None
        except Exception as e:
            print(f"FAISS initialization error: {e}")
            self.vector_store = None

        print("✅ Using Pinecone multilingual-e5-large embeddings")

        # ── Initialize BGE Reranker via Pinecone ──────────────────────────
        self.reranker_available = False
        try:
            if PINECONE_AVAILABLE and pinecone_api_key and pinecone_api_key != "dummy":
                self.pc = Pinecone(api_key=pinecone_api_key)
                if hasattr(self.pc, "inference"):
                    self.reranker_available = True
                    self.reranker_type = _RET_CFG.get("reranker_model", "bge-reranker-v2-m3")
                    print(f"✅ BGE Reranker ({self.reranker_type}) available")
                else:
                    self.reranker_type = "none"
                    self.pc = None
            else:
                self.reranker_type = "none"
                self.pc = None
        except Exception as e:
            print(f"⚠️ Reranker initialization failed: {e}")
            self.reranker_available = False
            self.reranker_type = "none"
            self.pc = None

        # ── Initialise inference stage (GroqEvaluator) ──────────────────────
        self.evaluator = GroqEvaluator(
            groq_api_key=self._groq_key,
            reranking_method=self.reranker_type,
            model=self.model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_retries=self._max_retries,
        )
    
    def search_similar_chunks(self, query: str, top_k: int = 20,
                              rerank_top_k: int = 5,
                              namespace: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using FAISS instead of Pinecone.
        Keeps the same reranking logic.
        `namespace` is accepted for API consistency (FAISS uses index_name for isolation).
        """
        if not self.vector_store:
            print("❌ FAISS vector store not available")
            return []
        
        try:
            # Generate query embedding using Pinecone (same as before)
            print(f"🔍 Generating query embedding...")
            query_embedding = generate_query_embedding_pinecone(query, self.pinecone_api_key)
            
            if not query_embedding:
                print("❌ Failed to generate query embedding")
                return []
            
            # Search FAISS index instead of Pinecone
            print(f"🔍 Searching FAISS index for top {top_k} similar chunks...")
            search_results = self.vector_store.query(
                query_vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not search_results or not search_results.get('matches'):
                print("❌ No similar chunks found in FAISS index")
                return []
            
            matches = search_results['matches']
            print(f"✅ Found {len(matches)} similar chunks in FAISS")
            
            # Convert FAISS results to same format as Pinecone
            chunks = []
            for match in matches:
                # Get text content from either 'content' or 'text' field in metadata
                metadata = match.get('metadata', {})
                text_content = metadata.get('content', '') or metadata.get('text', '') or match.get('content', '')
                
                chunk_data = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': metadata,
                    'text': text_content,  # Use the actual text content
                    'content': text_content,  # Also store as content for compatibility
                    'document_name': metadata.get('document_name', 'unknown'),
                    'page_number': metadata.get('page_number', 0),
                    'chunk_index': metadata.get('chunk_index', 0)  # Add chunk_index for adjacent chunks
                }
                chunks.append(chunk_data)
            
            # Apply reranking if available (same logic as before)
            if self.reranker_available and len(chunks) > 1:
                try:
                    print(f"🔄 Reranking {len(chunks)} chunks with BGE Reranker-v2-m3...")
                    reranked_chunks = self._rerank_chunks_bge(query, chunks, top_k=rerank_top_k)
                    final_chunks = reranked_chunks
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}, using similarity scores only")
                    final_chunks = chunks[:rerank_top_k]
            else:
                print(f"📊 Using similarity scores only, returning top {rerank_top_k} chunks")
                final_chunks = chunks[:rerank_top_k]

            # ── Guardrail: Abstain if similarity is too low ─────────────────
            abstain_thresh = float(_CFG.get("guardrails", {}).get("abstain_below_similarity", 0.15))
            if final_chunks:
                # Use best of similarity or rerank score
                top_score = max(
                    final_chunks[0].get("score", 0.0),
                    final_chunks[0].get("rerank_score", 0.0)
                )
                if top_score < abstain_thresh:
                    print(f"🚫 [Guardrail] Top score {top_score:.4f} < {abstain_thresh} (abstain).")
                    return []

            return final_chunks
                
        except Exception as e:
            print(f"❌ Error searching similar chunks: {e}")
            return []
    
    def _rerank_chunks_bge(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank chunks using BGE Reranker-v2-m3 via Pinecone inference.
        Same logic as original but works with FAISS search results.
        """
        if not chunks or not self.pc:
            return chunks[:top_k]
        
        try:
            # Prepare documents for reranking
            documents = []
            for chunk in chunks:
                text = chunk.get('text', '')
                if not text and 'metadata' in chunk:
                    text = chunk['metadata'].get('text', '')
                
                if text:
                    # Clean text but preserve full content
                    clean_text = re.sub(r'\s+', ' ', text).strip()
                    documents.append(clean_text)  # No character limit - preserve full content
                else:
                    documents.append("")
            
            if not documents:
                return chunks[:top_k]
            
            # Use Pinecone BGE reranker (without top_k parameter)
            rerank_response = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=documents,
                return_documents=False
            )
            
            # Process reranking results and limit to top_k
            reranked_chunks = []
            for result in rerank_response.data[:top_k]:  # Limit results here instead
                original_idx = result.index
                if 0 <= original_idx < len(chunks):
                    chunk = chunks[original_idx].copy()
                    chunk['rerank_score'] = result.score
                    chunk['original_similarity_score'] = chunk.get('score', 0.0)
                    reranked_chunks.append(chunk)
            
            print(f"🎯 BGE reranker processed {len(documents)} documents, returned {len(reranked_chunks)} reranked results")
            return reranked_chunks
            
        except Exception as e:
            print(f"⚠️ BGE reranking error: {e}")
            return chunks[:top_k]
    
    def process_query(self, query: str, search_top_k: int = 20, final_top_k: int = 5,
                     evaluation_method: str = "llm_with_quotes",
                     namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete query processing pipeline using FAISS.
        Same logic as original but uses FAISS for vector search.
        """
        start_time = time.time()

        # Step 1: Search for similar chunks using FAISS
        similar_chunks = self.search_similar_chunks(
            query, top_k=search_top_k, rerank_top_k=final_top_k, namespace=namespace
        )
        
        if not similar_chunks:
            return {
                "query": query,
                "search_results": [],
                "evaluation": {
                    "answer": "No relevant information found in the document database.",
                    "confidence": 0.0,
                    "search_method": "faiss_vector_search",
                    "reranking_method": self.reranker_type,
                    "model_used": self.model_name,
                    "no_results": True
                },
                "api_status": {
                    "faiss_search": "success",
                    "reranking": "not_applicable" if not self.reranker_available else "not_used",
                    "llm_evaluation": "not_used"
                },
                "timing": {
                    "total_time": time.time() - start_time,
                    "search_time": time.time() - start_time,
                    "rerank_time": 0,
                    "llm_time": 0
                },
                "status": "no_results",
                "success": True
            }
        
        search_time = time.time()
        
        # Step 2: Format search results — use consistent field names expected by the response builder
        formatted_results = []
        for i, chunk in enumerate(similar_chunks):
            result = {
                "rank":            i + 1,
                # These exact keys are read by final_backend.py's response builder:
                "id":              str(chunk.get('id', f'chunk_{i}')),
                "text":            chunk.get('text', chunk.get('content', '')),
                "score":           chunk.get('score', 0.0),
                "similarity_score": chunk.get('score', 0.0),
                "document_name":   chunk.get('document_name', 'unknown'),
                "page_number":     chunk.get('page_number', 0),
                "chunk_index":     chunk.get('chunk_index', 0),
                "metadata":        chunk.get('metadata', {}),
            }

            if 'rerank_score' in chunk:
                result["rerank_score"] = chunk['rerank_score']
                result["original_similarity_score"] = chunk.get('original_similarity_score', 0.0)

            formatted_results.append(result)
        
        # Step 3: LLM Evaluation — delegate to inference stage
        llm_start_time = time.time()
        context = self._create_comprehensive_context(similar_chunks)
        llm_answer: LLMAnswer = self.evaluator.evaluate(
            query=query,
            chunks=similar_chunks,
            context=context,
            method=evaluation_method,
        )
        evaluation_result = llm_answer.to_dict()
        llm_time = time.time() - llm_start_time
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "search_results": formatted_results,
            "evaluation": evaluation_result,
            "api_status": {
                "faiss_search": "success",
                "reranking": "success" if self.reranker_available else "not_available",
                "llm_evaluation": "success" if self.evaluator.available else "not_available"
            },
            "timing": {
                "total_time": total_time,
                "search_time": search_time - start_time,
                "rerank_time": llm_start_time - search_time,
                "llm_time": llm_time
            },
            "status": "success",
            "success": True
        }
    
    def _get_adjacent_chunks_extended(self, doc_name: str, chunk_index: int, chunks_before: int = 25, chunks_after: int = 25) -> List[Dict]:
        """Retrieve extended adjacent chunks (25 before + 25 after) from the same document."""
        try:
            if not self.vector_store:
                return []
            
            # Create a dummy vector for metadata-only search (correct dimension)
            dummy_vector = [0.0] * 1024  # Fixed dimension for multilingual-e5-large
            
            # Get all chunks from FAISS
            search_results = self.vector_store.query(dummy_vector, top_k=200)
            results = search_results.get('matches', [])
            
            # Filter and find adjacent chunks manually
            same_doc_chunks = []
            for match in results:
                metadata = match.get('metadata', {}) or {}
                if metadata.get("document_name") == doc_name:
                    # Get text content from multiple possible fields
                    text_content = metadata.get("content", "") or metadata.get("text", "") or match.get('content', '')
                    same_doc_chunks.append({
                        "text": text_content,
                        "chunk_index": metadata.get("chunk_index", 0),
                        "chunk_id": match.get('id', ''),
                        "score": match.get('score', 0.0)
                    })
            
            # Sort by chunk index to find adjacent chunks
            same_doc_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Find chunks adjacent to our target
            adjacent = []
            target_found = False
            target_position = None
            
            for i, chunk in enumerate(same_doc_chunks):
                if chunk["chunk_index"] == chunk_index:
                    target_found = True
                    target_position = i
                    break
            
            if target_found:
                # Get previous chunks
                start_idx = max(0, target_position - chunks_before)
                for j in range(start_idx, target_position):
                    prev_chunk = same_doc_chunks[j]
                    adjacent.append({
                        "text": prev_chunk["text"],
                        "chunk_index": prev_chunk["chunk_index"],
                        "position": "before",
                        "distance": target_position - j
                    })
                
                # Get next chunks
                end_idx = min(len(same_doc_chunks), target_position + chunks_after + 1)
                for j in range(target_position + 1, end_idx):
                    next_chunk = same_doc_chunks[j]
                    adjacent.append({
                        "text": next_chunk["text"],
                        "chunk_index": next_chunk["chunk_index"],
                        "position": "after",
                        "distance": j - target_position
                    })
            else:
                print(f"⚠️ Target chunk {chunk_index} not found in document {doc_name}")
                return []
            
            print(f"🔍 Found {len(adjacent)} adjacent chunks for {doc_name} chunk {chunk_index} (target: {chunks_before} before + {chunks_after} after)")
            return adjacent
            
        except Exception as e:
            print(f"⚠️ Could not retrieve adjacent chunks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_comprehensive_context(self, top_vectors: List[Dict]) -> str:
        """
        Build a deduplicated, document-ordered context from the top reranked vectors.

        Strategy
        --------
        1. Compute the desired chunk window [idx-adj_before … idx+adj_after] for every
           top vector and union those windows per document into a single set of indices.
        2. Fetch adjacent chunk text only for *unique* top-vector positions (avoids
           redundant FAISS calls when the same chunk appears in multiple top results).
        3. Emit each chunk exactly once, sorted by (document, chunk_index).
        4. Mark chunks that are top-reranked results with 🔑 so the LLM knows which
           sections are most relevant.
        """
        from collections import defaultdict

        if not top_vectors:
            return ""

        # ── 1. Mark top-result positions & compute desired window per document ──
        top_keys: set = set()             # (doc_name, chunk_index) for top-ranked chunks
        desired: dict = defaultdict(set)  # doc_name → set of chunk indices to include

        for vector in top_vectors:
            doc_name  = vector.get("document_name", "unknown")
            chunk_idx = vector.get("chunk_index", 0)
            top_keys.add((doc_name, chunk_idx))
            for ci in range(max(0, chunk_idx - self._adj_before),
                            chunk_idx + self._adj_after + 1):
                desired[doc_name].add(ci)

        # ── 2. Build chunk text lookup — seed with top-vector texts ─────────────
        chunk_lookup: dict = {}  # (doc_name, chunk_index) → text

        for vector in top_vectors:
            doc_name  = vector.get("document_name", "unknown")
            chunk_idx = vector.get("chunk_index", 0)
            text = (vector.get("text", "")
                    or vector.get("content", "")
                    or vector.get("metadata", {}).get("content", ""))
            chunk_lookup[(doc_name, chunk_idx)] = text

        # Fetch adjacent chunks — deduplicate calls by unique (doc, chunk) position
        fetched: set = set()
        for vector in top_vectors:
            doc_name  = vector.get("document_name", "unknown")
            chunk_idx = vector.get("chunk_index", 0)
            if (doc_name, chunk_idx) in fetched:
                continue                          # already fetched neighbours for this slot
            fetched.add((doc_name, chunk_idx))
            for adj in self._get_adjacent_chunks_extended(
                    doc_name, chunk_idx, self._adj_before, self._adj_after):
                key = (doc_name, adj["chunk_index"])
                if key not in chunk_lookup:
                    chunk_lookup[key] = adj["text"]

        # ── 3. Assemble context: one section per document, chunks in order ───────
        context_sections = []
        total_unique = 0

        for doc_name in sorted(desired):
            indices = sorted(desired[doc_name])
            available = [(ci, chunk_lookup[(doc_name, ci)])
                         for ci in indices if (doc_name, ci) in chunk_lookup]
            if not available:
                continue

            parts = [f"=== {doc_name} ===", ""]
            prev = None
            for ci, text in available:
                if prev is not None and ci > prev + 1:
                    parts.append(f"  [...gap: chunks {prev+1}–{ci-1} not in index...]")
                marker = "🔑 [TOP RESULT] " if (doc_name, ci) in top_keys else ""
                parts.append(f"[Chunk {ci}] {marker}{text}")
                prev = ci
                total_unique += 1
            parts.append("")
            context_sections.append("\n".join(parts))

        full_context = "\n".join(context_sections)
        print(f"📊 Deduplicated context: {total_unique} unique chunks "
              f"across {len(desired)} doc(s)  |  {len(full_context):,} chars")
        return full_context
    
    async def process_queries_batch(self, queries: List[str], 
                                  query_embeddings: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch using multithreading for better performance.
        
        Args:
            queries: List of query strings
            query_embeddings: Optional pre-computed embeddings for queries
            
        Returns:
            List of query results in the same order as input queries
        """
        import asyncio
        import concurrent.futures
        from threading import Thread
        import threading
        
        results = [None] * len(queries)
        
        def process_single_query(query_idx: int, query: str):
            """Process a single query in a thread."""
            thread_id = threading.get_ident()
            print(f"🧵 Thread {thread_id}: Starting query processing for query {query_idx + 1}")
            
            start_time = time.time()
            try:
                result = self.process_query(query)
                processing_time = time.time() - start_time
                print(f"✅ Thread {thread_id}: Completed query processing for query {query_idx + 1} in {processing_time:.2f}s")
                results[query_idx] = result
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ Thread {thread_id}: Error processing query {query_idx + 1}: {e}")
                results[query_idx] = {
                    "query": query,
                    "search_results": [],
                    "evaluation": {
                        "answer": f"Query processing failed: {str(e)}",
                        "confidence": 0.0,
                        "search_method": "faiss_vector_search",
                        "error": str(e)
                    },
                    "status": "error",
                    "success": False,
                    "timing": {"total_time": processing_time}
                }
        
        # Create and start threads for each query
        threads = []
        max_workers = min(len(queries), 10)  # Limit concurrent threads
        
        print(f"🚀 Starting multithreaded query processing for {len(queries)} queries with {max_workers} workers")
        
        # Process queries in batches to avoid too many concurrent threads
        for i in range(0, len(queries), max_workers):
            batch_queries = queries[i:i+max_workers]
            batch_threads = []
            
            for j, query in enumerate(batch_queries):
                query_idx = i + j
                thread = Thread(target=process_single_query, args=(query_idx, query))
                thread.start()
                batch_threads.append(thread)
            
            # Wait for this batch to complete
            for thread in batch_threads:
                thread.join()
        
        print(f"✅ Completed multithreaded query processing in batch mode")
        
        # Filter out None results (shouldn't happen but just in case)
        final_results = [r for r in results if r is not None]
        
        # Add batch processing info to each result
        for i, result in enumerate(final_results):
            if isinstance(result, dict):
                result["batch_info"] = {
                    "query_index": i,
                    "total_queries": len(queries),
                    "processing_mode": "multithreaded"
                }
        
        return final_results
