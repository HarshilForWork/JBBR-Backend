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
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# FAISS storage and embeddings integration
from .faiss_storage import FAISSVectorStore, check_or_create_faiss_index
from .embed_and_index import generate_query_embedding_pinecone


class FAISSQueryProcessor:
    """Query processor using FAISS for vector storage instead of Pinecone."""
    
    def __init__(self, pinecone_api_key: str, gemini_api_key: str, index_name: str = 'policy-index'):
        self.pinecone_api_key = pinecone_api_key
        self.gemini_api_key = gemini_api_key
        self.index_name = index_name
        self.quota_exceeded = False
        self.fallback_reason = None
        
        # Initialize FAISS vector store
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
        
        # Use Pinecone embeddings - no fallbacks
        print("✅ Using Pinecone multilingual-e5-large embeddings")
        
        # Initialize BGE Reranker with Pinecone
        self.reranker_available = False
        try:
            # Check if BGE reranker is available in Pinecone
            if PINECONE_AVAILABLE and pinecone_api_key and pinecone_api_key != 'dummy':
                self.pc = Pinecone(api_key=pinecone_api_key)
                if hasattr(self.pc, 'inference'):
                    print("🔍 Checking BGE Reranker-v2-m3 availability...")
                    self.reranker_available = True
                    self.reranker_type = "bge-reranker-v2-m3"
                    print("✅ BGE Reranker-v2-m3 available for reranking")
                else:
                    print("⚠️ BGE reranker not available, using similarity scores only")
                    self.reranker_type = "none"
                    self.pc = None
            else:
                print("⚠️ Pinecone not available for reranking, using similarity scores only")
                self.reranker_type = "none"
                self.pc = None
        except Exception as e:
            print(f"⚠️ Reranker initialization failed: {e}")
            self.reranker_available = False
            self.reranker_type = "none"
            self.pc = None
        
        # Initialize Gemini with better error handling and safety settings
        if GENAI_AVAILABLE and gemini_api_key and gemini_api_key != 'dummy':
            try:
                genai.configure(api_key=gemini_api_key)
                
                # Try different models without testing (avoid token limits)
                model_options = [
                    'gemini-2.5-flash',      # Most reliable
                    'gemini-2.5-pro',       # Good alternative
                ]
                
                self.model = None
                self.model_name = None
                
                # Try models without testing to avoid token limit issues
                for model_name in model_options:
                    try:
                        # Set generation config optimized for JSON responses (same as query_processor.py)
                        generation_config = {
                            "temperature": 0.7,  # Low temperature for consistent structured outputs
                        }
                        
                        self.model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config
                        )
                        self.model_name = model_name
                        print(f"✅ Successfully initialized Gemini model: {model_name} (no test performed)")
                        break
                            
                    except Exception as model_error:
                        print(f"⚠️ Model {model_name} not accessible: {model_error}")
                        continue
                
                if not self.model:
                    print("❌ No Gemini models are accessible with the provided API key")
                    self.model = None
                    
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self.model = None
        else:
            self.model = None
            print("⚠️ Gemini API not available")
    
    def search_similar_chunks(self, query: str, top_k: int = 20, rerank_top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using FAISS instead of Pinecone.
        Keeps the same reranking logic.
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
                    print(f"✅ Reranking complete, returning top {len(reranked_chunks)} chunks")
                    return reranked_chunks
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}, using similarity scores only")
                    return chunks[:rerank_top_k]
            else:
                print(f"📊 Using similarity scores only, returning top {rerank_top_k} chunks")
                return chunks[:rerank_top_k]
                
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
                     evaluation_method: str = "llm_with_quotes") -> Dict[str, Any]:
        """
        Complete query processing pipeline using FAISS.
        Same logic as original but uses FAISS for vector search.
        """
        start_time = time.time()
        
        # Step 1: Search for similar chunks using FAISS
        similar_chunks = self.search_similar_chunks(query, top_k=search_top_k, rerank_top_k=final_top_k)
        
        if not similar_chunks:
            return {
                "query": query,
                "search_results": [],
                "evaluation": {
                    "answer": "No relevant information found in the document database.",
                    "confidence": 0.0,
                    "search_method": "faiss_vector_search",
                    "reranking_method": self.reranker_type,
                    "model_used": self.model_name if self.model else "none",
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
        
        # Step 2: Format search results
        formatted_results = []
        for i, chunk in enumerate(similar_chunks):
            result = {
                "rank": i + 1,
                "chunk_id": chunk.get('id', f'chunk_{i}'),
                "content": chunk.get('text', ''),
                "document_name": chunk.get('document_name', 'unknown'),
                "page_number": chunk.get('page_number', 0),
                "similarity_score": chunk.get('score', 0.0),
                "metadata": chunk.get('metadata', {})
            }
            
            if 'rerank_score' in chunk:
                result["rerank_score"] = chunk['rerank_score']
                result["original_similarity_score"] = chunk.get('original_similarity_score', 0.0)
            
            formatted_results.append(result)
        
        # Step 3: LLM Evaluation (same as original)
        llm_start_time = time.time()
        evaluation_result = self._evaluate_with_llm(query, similar_chunks, evaluation_method)
        llm_time = time.time() - llm_start_time
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "search_results": formatted_results,
            "evaluation": evaluation_result,
            "api_status": {
                "faiss_search": "success",
                "reranking": "success" if self.reranker_available else "not_available",
                "llm_evaluation": "success" if self.model else "not_available"
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
    
    def _evaluate_with_llm(self, query: str, chunks: List[Dict], method: str = "llm_with_quotes") -> Dict[str, Any]:
        """
        LLM evaluation logic (same as original).
        """
        if not self.model:
            return {
                "answer": "LLM evaluation not available - API key may be invalid or quota exceeded.",
                "confidence": 0.0,
                "search_method": "faiss_vector_search",
                "reranking_method": self.reranker_type,
                "model_used": "none",
                "evaluation_method": method,
                "llm_available": False
            }
        
        # Prepare comprehensive context with adjacent chunks (same as query_processor.py)
        context = self._create_comprehensive_context(chunks)
        
        # LLM prompt (updated to match query_processor.py format)
        prompt = f"""
        You are an insurance policy expert. Based on the comprehensive context from policy documents, provide a clear and concise answer in 2-3 sentences.

        QUERY: {query}

        CONTEXT FROM POLICY DOCUMENTS:
        {context}

        Instructions:
        - These 5 vectors were reranked using BGE Reranker-v2-m3 for maximum relevance to your query
        - Consider information from all 5 vector sections when forming your answer
        - Look for complementary information across different sections
        - If multiple sections discuss the same topic, synthesize the information
        - For coverage questions, check waiting periods, exclusions, and conditions across all sections
        - For amount/limit questions, look for specific numbers in any of the sections
        - Don't look for exact phrases; instead, focus on the overall meaning and context.

        CRITICAL: YOU MUST RETURN ONLY VALID JSON FORMAT:
        {{
          "answer": "Your detailed answer here "
        }}

        RULES:
        - For yes/no questions, format your answer as "Yes, [brief reason]" or "No, [brief reason]"
        - Even if something is not explicitly mentioned, infer from the comprehensive context provided
        - Use information from multiple sections to provide a complete answer
        - Be careful to consider all relevant information before making a decision
        - RETURN ONLY THE JSON OBJECT - NO OTHER TEXT

        RETURN ONLY THE JSON OBJECT:
        """

        try:
            # Generate response using Gemini (simplified like query_processor.py)
            response = self.model.generate_content(prompt)
            
            # Check response validity more carefully
            if response and response.candidates:
                candidate = response.candidates[0]
                
                # Check if response was blocked
                if candidate.finish_reason.name != "STOP":
                    return {
                        "answer": f"Response blocked by safety filter: {candidate.finish_reason.name}",
                        "confidence": 0.0,
                        "search_method": "faiss_vector_search",
                        "reranking_method": self.reranker_type,
                        "model_used": self.model_name,
                        "evaluation_method": method,
                        "llm_available": True,
                        "safety_block": True,
                        "finish_reason": candidate.finish_reason.name
                    }
                
                # Check if we have valid content
                if candidate.content and candidate.content.parts:
                    raw_answer = candidate.content.parts[0].text.strip()
                    
                    # Try to extract JSON from the response
                    parsed_answer = self._extract_json_from_response(raw_answer)
                    
                    if parsed_answer and "answer" in parsed_answer:
                        answer = parsed_answer["answer"]
                    else:
                        # Fallback to raw answer if JSON parsing fails
                        answer = raw_answer
                    
                    return {
                        "answer": answer,
                        "confidence": 0.8,  # Static confidence value
                        "search_method": "faiss_vector_search",
                        "reranking_method": self.reranker_type,
                        "model_used": self.model_name,
                        "evaluation_method": method,
                        "llm_available": True,
                        "context_length": len(context),
                        "num_sources": len(chunks),
                        "json_parsed": parsed_answer is not None,
                        "llm_context": context,  # Full context used by LLM
                        "source_vectors": chunks  # Vectors used to build context
                    }
                else:
                    return {
                        "answer": "LLM response contains no valid text content.",
                        "confidence": 0.0,
                        "search_method": "faiss_vector_search",
                        "reranking_method": self.reranker_type,
                        "model_used": self.model_name,
                        "evaluation_method": method,
                        "llm_available": True,
                        "error": "No content parts in response",
                        "llm_context": context,  # Include context even for errors
                        "source_vectors": chunks
                    }
            else:
                return {
                    "answer": "LLM did not generate any response candidates.",
                    "confidence": 0.0,
                    "search_method": "faiss_vector_search",
                    "reranking_method": self.reranker_type,
                    "model_used": self.model_name,
                    "evaluation_method": method,
                    "llm_available": True,
                    "error": "No response candidates",
                    "llm_context": context,  # Include context even for errors
                    "source_vectors": chunks
                }
                
        except Exception as e:
            print(f"❌ LLM evaluation error: {e}")
            return {
                "answer": f"LLM evaluation failed: {str(e)}",
                "confidence": 0.0,
                "search_method": "faiss_vector_search",
                "reranking_method": self.reranker_type,
                "model_used": self.model_name if self.model else "none",
                "evaluation_method": method,
                "llm_available": False,
                "error": str(e),
                "llm_context": context if 'context' in locals() else "",
                "source_vectors": chunks
            }
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Helper method to robustly extract JSON from LLM responses (copied from query_processor.py)."""
        import json
        import re
        
        if not response_text or not response_text.strip():
            print(f"🔍 Empty response, skipping JSON extraction")
            return None
        
        response_text = response_text.strip()
        
        # Try parsing the whole response first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON block in the response
        if '{' in response_text and '}' in response_text:
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"🔍 JSON extraction failed: {str(e)[:50]}...")
                print(f"🔍 Raw response preview: '{response_text[:200]}...'")
        
        # Try to find JSON-like content with regex
        json_pattern = r'\{[^{}]*"answer"[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # If all else fails, try to extract just the answer content
        answer_pattern = r'"answer":\s*"([^"]*)"'
        answer_match = re.search(answer_pattern, response_text)
        if answer_match:
            return {"answer": answer_match.group(1)}
        
        return None
    
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
        """Create comprehensive context from top 5 vectors with their adjacent chunks."""
        context_sections = []
        
        for i, vector in enumerate(top_vectors, 1):
            doc_name = vector.get("document_name", "unknown")
            chunk_index = vector.get("chunk_index", 0)
            # Get text content from multiple possible fields
            main_text = vector.get("text", "") or vector.get("content", "") or vector.get("metadata", {}).get("content", "")
            similarity_score = vector.get("score", 0.0)
            
            print(f"📄 Processing Vector {i}: Getting adjacent context for chunk {chunk_index} from {doc_name}")
            print(f"🔍 Main text preview: {main_text[:100]}..." if len(main_text) > 100 else f"🔍 Main text: '{main_text}'")
            if not main_text.strip():
                print(f"⚠️ WARNING: Main text is empty for vector {i}! Vector data: {vector}")
            
            # Get 25 chunks before and 25 chunks after
            adjacent_chunks = self._get_adjacent_chunks_extended(doc_name, chunk_index, 25, 25)
            
            # Organize chunks
            before_chunks = [c for c in adjacent_chunks if c["position"] == "before"]
            after_chunks = [c for c in adjacent_chunks if c["position"] == "after"]
            
            # Sort by distance from main chunk
            before_chunks.sort(key=lambda x: x["distance"], reverse=True)  # Closest first
            after_chunks.sort(key=lambda x: x["distance"])  # Closest first
            
            # Build the section
            section_parts = []
            
            # Add header for this vector
            section_parts.append(f"=== VECTOR {i} (Similarity: {similarity_score:.3f}) ===")
            section_parts.append(f"Document: {doc_name}")
            section_parts.append(f"Main Chunk Index: {chunk_index}")
            section_parts.append("")
            
            # Add before context
            if before_chunks:
                section_parts.append(f"--- CONTEXT BEFORE (25 chunks) ---")
                for chunk in before_chunks:
                    section_parts.append(f"[Chunk {chunk['chunk_index']}] {chunk['text']}")
                section_parts.append("")
            
            # Add main chunk
            section_parts.append(f"--- MAIN CHUNK (Most Relevant) ---")
            section_parts.append(f"[Chunk {chunk_index}] {main_text}")
            section_parts.append("")
            
            # Add after context
            if after_chunks:
                section_parts.append(f"--- CONTEXT AFTER (25 chunks) ---")
                for chunk in after_chunks:
                    section_parts.append(f"[Chunk {chunk['chunk_index']}] {chunk['text']}")
                section_parts.append("")
            
            # Add summary for this vector
            total_context = len(before_chunks) + 1 + len(after_chunks)
            section_parts.append(f"--- END VECTOR {i} (Total chunks: {total_context}) ---")
            section_parts.append("")
            
            context_sections.append("\n".join(section_parts))
        
        # Combine all sections
        full_context = "\n".join(context_sections)
        
        print(f"📊 Created comprehensive context with {len(top_vectors)} vectors and their adjacent chunks")
        print(f"📊 Total context length: {len(full_context)} characters")
        
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
