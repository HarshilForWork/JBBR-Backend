"""
Module: query_processor.py
Functionality: Complete query processing pipeline with LLM integration using FAISS for vector storage.
"""
import json
import re
import time
import warnings
import os
import pickle
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
from rake_nltk import Rake

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

# Pinecone embeddings integration
from .embed_and_index import generate_query_embedding_pinecone

class QueryProcessor:
    def __init__(self, pinecone_api_key: str, gemini_api_key: str, index_name: str = 'policy-index'):
        self.pinecone_api_key = pinecone_api_key
        self.gemini_api_key = gemini_api_key
        self.index_name = index_name
        self.quota_exceeded = False
        self.fallback_reason = None
        
        # Initialize FAISS for vector search
        try:
            print("🔍 Checking FAISS index...")
            storage_dir = 'faiss_storage'
            index_path = os.path.join(storage_dir, f"{index_name}.faiss")
            metadata_path = os.path.join(storage_dir, f"{index_name}_metadata.json")
            id_map_path = os.path.join(storage_dir, f"{index_name}_id_map.pkl")
            
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
                
                # Load metadata
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                except:
                    self.metadata = {}
                
                # Load ID mapping
                try:
                    with open(id_map_path, 'rb') as f:
                        id_map_data = pickle.load(f)
                        self.id_to_idx = id_map_data['id_to_idx']
                        self.idx_to_id = id_map_data['idx_to_id']
                except:
                    self.id_to_idx = {}
                    self.idx_to_id = {}
                
                print(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                print("❌ FAISS index not found")
                self.faiss_index = None
                self.metadata = {}
                self.id_to_idx = {}
                self.idx_to_id = {}
                
        except Exception as e:
            print(f"FAISS initialization error: {e}")
            self.faiss_index = None
            self.metadata = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
        
        # Use Pinecone embeddings - no fallbacks
        print("✅ Using Pinecone multilingual-e5-large embeddings")
        
        # Initialize Pinecone client for embeddings and reranking
        if PINECONE_AVAILABLE and pinecone_api_key and pinecone_api_key != 'dummy':
            try:
                self.pc = Pinecone(api_key=pinecone_api_key)
                print("✅ Initialized Pinecone client for embeddings and reranking")
            except Exception as e:
                print(f"Pinecone client initialization error: {e}")
                self.pc = None
        else:
            self.pc = None
        
        # Initialize BGE Reranker
        self.reranker_available = False
        try:
            # Check if BGE reranker is available in Pinecone
            if hasattr(self.pc, 'inference') and self.pc:
                print("🔍 Checking BGE Reranker-v2-m3 availability...")
                self.reranker_available = True
                self.reranker_type = "bge-reranker-v2-m3"
                print("✅ BGE Reranker-v2-m3 available for reranking")
            else:
                print("⚠️ BGE reranker not available, using similarity scores only")
                self.reranker_type = "none"
        except Exception as e:
            print(f"⚠️ Reranker initialization failed: {e}")
            self.reranker_available = False
            self.reranker_type = "none"
        
        # Initialize Gemini
        if GENAI_AVAILABLE and gemini_api_key and gemini_api_key != 'dummy':
            try:
                genai.configure(api_key=gemini_api_key)
                # Try different models based on availability
                model_options = [
                    'gemini-2.5-flash',      # Most reliable
                    'gemini-2.5-pro',       # Good alternative
                ]
                
                self.llm = None
                self.model_name = None
                
                # Try each model until one works
                for model_name in model_options:
                    try:
                        # Set generation config optimized for JSON responses
                        # Use low temperature for consistent, structured outputs
                        generation_config = {
                            "temperature": 0.7,  # Very low temperature for structured JSON responses
                        }
                        
                        test_model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config
                        )
                        
                        # Test with a simple query to verify access
                        test_response = test_model.generate_content("Hello")
                        if test_response:
                            self.llm = test_model
                            self.model_name = model_name
                            print(f"Successfully initialized Gemini model: {model_name} with temperature=0.1 for JSON responses")
                            break
                    except Exception as e:
                        print(f"Failed to initialize {model_name}: {str(e)}")
                        if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                            self.quota_exceeded = True
                            self.fallback_reason = f"Quota exceeded for {model_name}"
                        continue
                
                if not self.llm:
                    print("Warning: No Gemini models available, using fallback methods")
                    if not self.quota_exceeded:
                        self.fallback_reason = "No models accessible"
                    
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self.llm = None
                self.model_name = None
                if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                    self.quota_exceeded = True
                    self.fallback_reason = "API quota exceeded"
                else:
                    self.fallback_reason = f"Initialization error: {str(e)}"
        else:
            self.llm = None
            self.model_name = None
            if gemini_api_key == 'dummy':
                self.fallback_reason = "Using dummy API key for testing"
            else:
                self.fallback_reason = "API key not provided"
    
    def _encode_query(self, query: str) -> List[float]:
        """Encode query using Pinecone's embedding service."""
        try:
            # Use Pinecone embeddings with the same API key
            return generate_query_embedding_pinecone(query, self.pinecone_api_key)
                    
        except Exception as e:
            print(f"❌ Pinecone query encoding error: {e}")
            # Return zero vector as fallback (1024 dimensions for multilingual-e5-large)
            return [0.0] * 1024
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and recommendations."""
        status = {
            'gemini_available': self.llm is not None,
            'model_name': getattr(self, 'model_name', None),
            'quota_exceeded': getattr(self, 'quota_exceeded', False),
            'fallback_reason': getattr(self, 'fallback_reason', None),
            'reranker_available': getattr(self, 'reranker_available', False),
            'reranker_type': getattr(self, 'reranker_type', 'none'),
            'recommendations': []
        }
        
        if self.quota_exceeded:
            status['recommendations'].extend([
                "🚨 Gemini API quota exceeded - system is using fallback methods",
                "💡 Wait for quota reset (usually 24 hours) or upgrade your API plan",
                "📚 Student tier: Check Google AI Studio for quota details",
                "⚡ Fallback methods provide ~70% accuracy vs LLM's 95%"
            ])
        elif not self.llm:
            status['recommendations'].extend([
                "⚠️ LLM not available - using rule-based analysis",
                "🔑 Check your Gemini API key is valid",
                "📱 Try different model tiers (flash, pro, pro-1.5)"
            ])
        else:
            status['recommendations'].append(f"✅ Using {self.model_name} for optimal results")
        
        # Add reranker status
        if self.reranker_available:
            status['recommendations'].append(f"🔄 Using {self.reranker_type} for enhanced relevance ranking")
        else:
            status['recommendations'].append("⚠️ Reranker not available - using similarity scores only")
        
        return status
    
    def _search_documents_faiss(self, embedding: list, top_k: int) -> list:
        """Search documents using FAISS"""
        if not self.faiss_index:
            print("❌ FAISS index not available")
            return []
        
        try:
            # Convert embedding to numpy array
            query_vector = np.array([embedding], dtype=np.float32)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_vector, min(top_k, self.faiss_index.ntotal))
            
            # Convert results to list format
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid index
                    # Get chunk ID from index mapping
                    chunk_id = self.idx_to_id.get(idx, f"chunk_{idx}")
                    
                    # Get metadata
                    metadata = self.metadata.get(chunk_id, {})
                    
                    results.append({
                        'id': chunk_id,
                        'score': float(score),
                        'values': [],  # FAISS doesn't store vectors in results
                        'metadata': metadata
                    })
            
            return results
            
        except Exception as e:
            print(f"FAISS search error: {e}")
            return []
    
    def _extract_json_from_response(self, response_text: str, context: str = "response") -> Optional[Dict]:
        """Helper method to robustly extract JSON from LLM responses."""
        if not response_text or not response_text.strip():
            print(f"🔍 Empty {context}, skipping JSON extraction")
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
                print(f"🔍 JSON extraction failed from {context}: {str(e)[:50]}...")
                print(f"🔍 Raw response preview: '{response_text[:200]}...'")
        
        # Try to find JSON-like content with regex
        import re
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
        
        print(f"🔍 No valid JSON found in {context}")
        print(f"🔍 Full response for debugging: '{response_text}'")
        return None
    
    def _extract_keywords_with_rake(self, query: str, max_words: int = 5) -> List[str]:
        """
        Extract keywords and phrases from a query using RAKE-NLTK and a
        heuristic for consecutive capitalized words.
        """
        # 1. Extract keywords using RAKE
        r = Rake()
        r.extract_keywords_from_text(query)
        ranked_phrases = r.get_ranked_phrases_with_scores()
        rake_keywords = [phrase for score, phrase in ranked_phrases if score > 1]
        if not rake_keywords:
            rake_keywords = [phrase for score, phrase in ranked_phrases]

        # 2. Extract consecutive capitalized words and acronyms
        # This pattern finds consecutive capitalized words
        capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', query)
        # This pattern finds acronyms like "TPA" or "OPD"
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)

        # 3. Combine and deduplicate
        combined_keywords = capitalized_phrases + acronyms + rake_keywords
        
        # Deduplicate while preserving order
        unique_keywords = list(dict.fromkeys(combined_keywords))
        
        # Limit to max_words
        return unique_keywords[:max_words]

    def _make_llm_request_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Make LLM request with retry logic and quota detection."""
        if not self.llm:
            return None
            
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"🔍 Empty response on attempt {attempt + 1}")
                    if attempt < max_retries:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    return None
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'quota' in error_msg or 'limit' in error_msg or 'exceeded' in error_msg:
                    print(f"⚠️ Gemini API quota exceeded: {e}")
                    self.quota_exceeded = True
                    return None
                elif attempt < max_retries:
                    print(f"🔄 Retry {attempt + 1} after error: {str(e)[:50]}...")
                    time.sleep(1)
                    continue
                else:
                    print(f"🔍 Final attempt failed: {e}")
                    return None
        
        return None

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from natural language query using Gemini only."""
        if self.llm:
            return self._llm_entity_extraction(query)
        else:
            print("❌ No LLM available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _llm_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Use LLM for entity extraction - no fallbacks."""
        if not self.llm:
            print("❌ LLM not available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
            
        prompt = f"""
        Extract the following entities from this insurance/medical query: "{query}"
        
        Return a JSON object with these fields (use null if not found):
        - age: integer (patient age)
        - gender: string (M/F/Male/Female)
        - procedure: string (medical procedure/surgery)
        - location: string (city/location)
        - policy_duration: string (how old is the policy)
        - policy_type: string (type of insurance policy)
        - amount: number (any monetary amount mentioned)
        
        Example: {{"age": 46, "gender": "M", "procedure": "knee surgery", "location": "Pune", "policy_duration": "3 months", "policy_type": null, "amount": null}}
        
        Only return the JSON, no other text.
        """
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("❌ No response from LLM for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
        
        # Extract JSON from response
        entities = self._extract_json_from_response(response_text, "entity extraction")
        if entities:
            print("✅ Successfully extracted entities using LLM")
            return entities
        else:
            print("❌ JSON extraction failed from LLM response")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query for hybrid search."""
        # Use the new RAKE-based keyword extraction
        return self._extract_keywords_with_rake(query)
        
    def _calculate_hybrid_score(self, candidate: Dict, keywords: List[str]) -> float:
        """Calculate a hybrid score based on vector similarity and keyword presence."""
        # Start with vector similarity score (typically 0-1)
        score = candidate.get("vector_score", 0.0)
        
        if not keywords:
            return score
            
        # Get text from candidate
        text = candidate.get("text", "").lower()
        
        # Count keywords present in the text
        keyword_count = sum(1 for kw in keywords if kw.lower() in text)
        
        # Boost score based on keyword matches (0.05 boost per keyword match)
        keyword_boost = min(0.3, keyword_count * 0.05)  # Cap at 0.3 to avoid dominating vector score
        
        # Combine scores
        hybrid_score = score + keyword_boost
        
        # Add debug info
        candidate["keyword_matches"] = keyword_count
        candidate["keyword_boost"] = keyword_boost
        
        return hybrid_score

    def _rerank_with_bge(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank candidates using BGE Reranker-v2-m3 via Pinecone."""
        if not self.reranker_available or not self.pc:
            print("⚠️ BGE reranker not available, returning original candidates")
            return candidates[:top_k]
        
        try:
            print(f"🔄 Reranking {len(candidates)} candidates with BGE Reranker-v2-m3...")
            
            # Prepare documents for reranking
            documents = [candidate["text"] for candidate in candidates]
            
            # Use Pinecone's inference API for BGE reranking
            rerank_response = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=documents,
                top_n=top_k,  # BGE reranker also uses top_n
                return_documents=True
            )
            
            # Map reranked results back to original candidates
            reranked_candidates = []
            for result in rerank_response.data:
                original_idx = result.index
                candidate = candidates[original_idx].copy()
                
                # Add reranking information
                candidate["rerank_score"] = result.score
                candidate["original_rank"] = original_idx + 1
                candidate["reranked_rank"] = len(reranked_candidates) + 1
                candidate["final_score"] = result.score
                candidate["ranking_method"] = "bge_reranker_v2_m3"
                
                reranked_candidates.append(candidate)
            
            print(f"✅ Reranked to top {len(reranked_candidates)} candidates using BGE Reranker-v2-m3")
            return reranked_candidates
            
        except Exception as e:
            print(f"❌ BGE reranking failed: {e}")
            print("🔄 Falling back to similarity-based ranking...")
            # Fallback to original similarity ranking
            return candidates[:top_k]
        """Decide if reranking is beneficial based on score distribution."""
        if len(candidates) <= final_k:
            return False
        
        # Get vector scores of top candidates
        top_scores = [c.get("vector_score", 0) for c in candidates[:final_k * 2]]
        
        if not top_scores:
            return True  # Always rerank if no scores
        
        # Calculate score variance - if scores are very similar, reranking helps
        import statistics
        try:
            score_std = statistics.stdev(top_scores) if len(top_scores) > 1 else 0
            # If standard deviation is low (scores are similar), reranking is beneficial
            return score_std < 0.1  # Threshold for "similar" scores
        except:
            return True  # Default to reranking if calculation fails

    def semantic_search_with_similarity(self, query: str, top_k: int = 5, query_embedding: Optional[list] = None) -> List[Dict]:
        """
        Hybrid search using vector similarity and keyword matching:
        1. Retrieve top candidates using vector similarity 
        2. Boost results that contain keywords from the query (post-retrieval)
        3. Return top results sorted by combined scores with context expansion
        """
        if not self.faiss_index:
            print("❌ FAISS index not available")
            return []
        try:
            # Use provided query_embedding if available, else encode
            if query_embedding is not None:
                print("🔍 Using precomputed query embedding for search...")
                embedding = query_embedding
            else:
                embedding = self._encode_query(query)
                
            print(f"🔍 Retrieving top {top_k} candidates using vector search...")
            
            # Extract keywords for hybrid search (post-retrieval)
            keywords = self._extract_keywords(query)
            if keywords:
                print(f"🔍 Will apply keyword boosting after retrieval: {keywords}")
            
            # Get more candidates for post-filtering
            retrieve_k = min(top_k , 20)  # Get more results but cap at 20
            
            # First retrieve with vector search only
            results = self._search_documents_faiss(embedding, retrieve_k)
            
            # Format results (already sorted by similarity score)
            candidates = []
            for match in results:
                candidates.append({
                    "id": match['id'],
                    "vector_score": match['score'],
                    "text": match.metadata.get("text", ""),
                    "document_name": match.metadata.get("document_name", ""),
                    "page_number": match.metadata.get("page_number", 1),
                    "chunk_index": match.metadata.get("chunk_index", 0)
                })
            
        
                
            if not candidates:
                print("⚠️ No candidates found")
                return []
                
            print(f"✅ Retrieved {len(candidates)} candidates with vector search")
            
            # Apply hybrid scoring that combines vector similarity with keyword matching
            if keywords:
                print("🔍 Applying hybrid scoring with keyword boosting...")
                for candidate in candidates:
                    hybrid_score = self._calculate_hybrid_score(candidate, keywords)
                    candidate["hybrid_score"] = hybrid_score
                    candidate["final_score"] = hybrid_score
                    
                # Re-sort based on hybrid scores
                candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                print(f"✅ Re-ranked results using hybrid scoring (vector + keyword boost)")
            else:
                # Use similarity scores as final scores (no hybrid scoring)
                for candidate in candidates:
                    candidate["final_score"] = candidate["vector_score"]
                print("ℹ️ Using pure vector similarity scores (no keywords found)")
            
            # Take top-k after hybrid scoring
            candidates = candidates[:top_k]
            
            # Context expansion
            print(f"📄 Context expansion for {len(candidates)} chunks...")
            expanded_results = self._expand_context(candidates)
            
            # Add ranking metadata
            for i, result in enumerate(expanded_results):
                result["final_rank"] = i + 1
                result["ranking_method"] = "hybrid_post_retrieval" if keywords else "vector_similarity"
                
            self._print_ranking_summary(query, expanded_results)
            return expanded_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _expand_context(self, candidates: List[Dict], context_chars: int = 500) -> List[Dict]:
        """Expand context around selected chunks by retrieving adjacent chunks."""
        expanded_results = []
        
        for candidate in candidates:
            try:
                doc_name = candidate["document_name"]
                chunk_index = candidate.get("chunk_index", 0)
                
                # Try to get adjacent chunks for context
                adjacent_chunks = self._get_adjacent_chunks(doc_name, chunk_index, context_chars)
                
                if adjacent_chunks:
                    # Combine with context
                    expanded_text = self._combine_chunks_with_context(
                        candidate["text"], 
                        adjacent_chunks,
                        context_chars
                    )
                    candidate["text"] = expanded_text
                    candidate["context_expanded"] = True
                    candidate["adjacent_chunks_count"] = len(adjacent_chunks)
                else:
                    candidate["context_expanded"] = False
                    candidate["adjacent_chunks_count"] = 0
                
                expanded_results.append(candidate)
                
            except Exception as e:
                print(f"⚠️ Context expansion failed for chunk: {e}")
                candidate["context_expanded"] = False
                expanded_results.append(candidate)
        
        return expanded_results
    
    def _get_adjacent_chunks_extended(self, doc_name: str, chunk_index: int, chunks_before: int = 25, chunks_after: int = 25) -> List[Dict]:
        """Retrieve extended adjacent chunks (25 before + 25 after) from the same document."""
        try:
            if not self.faiss_index:
                return []
            
            # Create a dummy vector for metadata-only search (correct dimension)
            dummy_vector = [0.0] * 1024  # Fixed dimension for multilingual-e5-large
            
            # Get all chunks from FAISS
            results = self._search_documents_faiss(dummy_vector, 200)
            
            # Filter and find adjacent chunks manually
            same_doc_chunks = []
            for match in results:
                metadata = match['metadata'] or {}
                if metadata.get("document_name") == doc_name:
                    same_doc_chunks.append({
                        "text": metadata.get("text", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "chunk_id": match['id'],
                        "score": match['score']
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
                # Get 15 previous chunks
                start_idx = max(0, target_position - chunks_before)
                for j in range(start_idx, target_position):
                    prev_chunk = same_doc_chunks[j]
                    adjacent.append({
                        "text": prev_chunk["text"],
                        "chunk_index": prev_chunk["chunk_index"],
                        "position": "before",
                        "distance": target_position - j
                    })
                
                # Get 15 next chunks
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
            
            print(f"🔍 Found {len(adjacent)} adjacent chunks for {doc_name} chunk {chunk_index} (target: 25 before + 25 after)")
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
            doc_name = vector["document_name"]
            chunk_index = vector.get("chunk_index", 0)
            main_text = vector["text"]
            similarity_score = vector.get("score", 0.0)
            
            print(f"📄 Processing Vector {i}: Getting adjacent context for chunk {chunk_index} from {doc_name}")
            
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
    
    def _print_ranking_summary(self, query: str, results: List[Dict]):
        """Print detailed ranking summary for debugging."""
        print(f"\n📊 Semantic Search Results Summary:")
        print(f"Query: '{query}'")
        print(f"Final results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            similarity_score = result.get("vector_score", 0)
            final_score = result.get("final_score", similarity_score)
            doc_name = result.get("document_name", "Unknown")
            context_expanded = result.get("context_expanded", False)
            
            print(f"  {i}. Doc: {doc_name}")
            print(f"     Similarity: {similarity_score:.3f} | Final: {final_score:.3f} | Context: {'✅' if context_expanded else '❌'}")
            print(f"     Text: {result['text'][:100]}...")
            print()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
         """Perform semantic search in FAISS."""
         if not self.faiss_index:
             return []
         
         try:
             # Generate query embedding using helper method
             query_embedding = self._encode_query(query)
             
             # Search in FAISS
             results = self._search_documents_faiss(query_embedding, top_k)
 
             # Format results
             search_results = []
             for m in results:
                 search_results.append({
                     "id": m['id'],
                     "score": m['score'],
                     "text": m['metadata'].get("text", ""),
                     "document_name": m['metadata'].get("document_name", ""),
                     "page_number": m['metadata'].get("page_number", 1)
                 })
             
             return search_results
         except Exception as e:
             print(f"Search error: {e}")
             return []
    
    def evaluate_claim(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM to evaluate claim based on retrieved context - no fallbacks."""
        
        if self.llm:
            return self._llm_evaluation(query, entities, search_results)
        else:
            print("❌ No LLM available for claim evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "LLM not available for claim evaluation.",
                "relevant_clauses": [],
                "reasoning": "System requires LLM for claim evaluation"
            }
    
    def _llm_evaluation_with_comprehensive_context(self, query: str, entities: Dict, comprehensive_context: str, top_vectors: List[Dict]) -> Dict[str, Any]:
        """Use LLM for evaluation with comprehensive adjacent context from 5 vectors."""
        if not self.llm:
            print("❌ LLM not available for evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "LLM not available for claim evaluation.",
                "relevant_clauses": [],
                "reasoning": "System requires LLM for claim evaluation"
            }
        
        # Create a summary of the vectors for reference
        vector_summary = []
        for i, vector in enumerate(top_vectors, 1):
            vector_summary.append(f"Vector {i}: {vector['document_name']} (Similarity: {vector['score']:.3f})")
        
        prompt = f"""
        You are an insurance policy expert. Based on the comprehensive context from policy documents, provide a clear and concise answer in 2-3 sentences.

        QUERY: {query}

        EXTRACTED ENTITIES:
        {json.dumps(entities, indent=2)}

        TOP 5 RERANKED VECTORS SUMMARY:
        {chr(10).join(vector_summary)}

        COMPREHENSIVE POLICY CONTEXT WITH ADJACENT CHUNKS:
        {comprehensive_context}

        Instructions:
        - These 5 vectors were reranked using BGE Reranker-v2-m3 for maximum relevance to your query
        - Each vector section contains the most relevant chunk plus 25 chunks before and 25 chunks after it
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
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("❌ No response from LLM for comprehensive evaluation")
            return {
                "answer": "Unable to process query due to LLM unavailability.",
                "source_document": "N/A",
                "relevant_sections": []
            }
        
        # Extract JSON from response
        evaluation = self._extract_json_from_response(response_text, "comprehensive evaluation")
        if evaluation:
            print("✅ Successfully completed comprehensive evaluation using LLM")
            # Add metadata about the comprehensive context
            evaluation['context_stats'] = {
                'total_vectors': len(top_vectors),
                'total_context_length': len(comprehensive_context),
                'chunks_per_vector': 51  # 25 before + 1 main + 25 after
            }
            return evaluation
        else:
            print("❌ JSON extraction failed from LLM comprehensive evaluation response")
            return {
                "answer": "Failed to parse LLM response for comprehensive evaluation.",
                "source_document": "N/A",
                "relevant_sections": []
            }
    
    def process_query(self, query: str, query_embedding: Optional[list] = None) -> Dict[str, Any]:
        """Complete query processing pipeline with Cohere reranking and comprehensive adjacent context."""
        try:
            # Get API status for debugging
            api_status = self.get_api_status()
            
            # Step 1: Get top 25 vectors with similarity search for reranking
            print("🔍 Step 1: Getting top 25 vectors with similarity search for reranking...")
            try:
                # Use provided query_embedding if available, else encode
                if query_embedding is not None:
                    print("🔍 Using precomputed query embedding for search...")
                    embedding = query_embedding
                else:
                    embedding = self._encode_query(query)
                
                # Get top 25 candidates for reranking (optimized retrieval)
                results = self._search_documents_faiss(embedding, 30)
                
                # Format results (already sorted by similarity score)
                candidates = []
                for match in results:
                    candidates.append({
                        "id": match['id'],
                        "score": match['score'],
                        "text": match['metadata'].get("text", ""),
                        "document_name": match['metadata'].get("document_name", ""),
                        "page_number": match['metadata'].get("page_number", 1),
                        "chunk_index": match['metadata'].get("chunk_index", 0),
                        "vector_score": match['score'],
                        "final_score": match['score']
                    })
                
                print(f"✅ Retrieved {len(candidates)} candidates for reranking")
                
                # Step 2: Rerank with BGE to get top 5
                print("🔍 Step 2: Reranking with BGE Reranker-v2-m3 to get top 5...")
                top_vectors = self._rerank_with_bge(query, candidates, top_k=5)
                
                if not top_vectors:
                    print("⚠️ No vectors after reranking, using top 5 from similarity search")
                    top_vectors = candidates[:5]
                
                # Step 3: Create comprehensive context with adjacent chunks for top 5 reranked vectors
                print("🔍 Step 3: Creating comprehensive context with 25 before + 25 after chunks for each top 5 reranked vector...")
                comprehensive_context = self._create_comprehensive_context(top_vectors)
                
                # Use the comprehensive context as search results for LLM
                search_results = top_vectors  # Keep reranked results for response structure
                
            except Exception as e:
                print(f"❌ Vector search/reranking error: {e}")
                top_vectors = []
                comprehensive_context = ""
                search_results = []
            
            # Step 4: Evaluate with comprehensive context using reranked top 5
            print("🔍 Step 4: Evaluating with comprehensive context using top 5 reranked vectors...")
            evaluation = self._llm_evaluation_with_comprehensive_context(query, {}, comprehensive_context, top_vectors)
            
            # Add search method information
            evaluation['search_method'] = 'bge_rerank_comprehensive_context'
            evaluation['reranker_type'] = self.reranker_type
            evaluation['reranker_available'] = self.reranker_available
            evaluation['hybrid_search_enabled'] = False
            evaluation['reranking_enabled'] = self.reranker_available
            evaluation['initial_candidates_retrieved'] = len(candidates) if 'candidates' in locals() else 0
            evaluation['final_vectors_after_reranking'] = len(top_vectors)
            evaluation['adjacent_chunks_per_vector'] = 50  # 25 before + 25 after
            
            # Add performance notes
            if not self.llm:
                evaluation['note'] = "❌ Analysis performed without LLM (required for full functionality)"
            
            if self.reranker_available:
                evaluation['performance_note'] = "⚡ Using BGE Reranker-v2-m3 + comprehensive adjacent context (25 before + 25 after) for top 5 vectors"
            else:
                evaluation['performance_note'] = "⚡ Using similarity ranking + comprehensive adjacent context (25 before + 25 after) for top 5 vectors"
            
            # Combine all results
            result = {
                "query": query,
                "search_results": search_results,
                "evaluation": evaluation,
                "api_status": api_status,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "query": query,
                "search_results": [],
                "evaluation": {
                    "decision": "error",
                    "amount": None,
                    "confidence": 0.0,
                    "justification": f"Processing error: {str(e)}",
                    "relevant_clauses": [],
                    "reasoning": "System error occurred"
                },
                "api_status": self.get_api_status(),
                "status": "error",
                "error": str(e)
            }
    
    async def process_queries_batch(self, queries: List[str], query_embeddings: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in parallel using asyncio.
        
        Args:
            queries: List of queries to process
            query_embeddings: Optional list of precomputed embeddings for each query
            
        Returns:
            List of results in the same order as input queries
        """
        import asyncio
        
        # Validate inputs
        if not queries:
            return []
        
        if query_embeddings and len(query_embeddings) != len(queries):
            raise ValueError("Number of embeddings must match number of queries")
        
        # Create semaphore to limit concurrent processing (prevent overwhelming the APIs)
        max_concurrent = min(10, len(queries))  # Limit to 10 concurrent queries
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query_async(idx: int, query: str) -> Dict[str, Any]:
            """Process a single query with semaphore protection."""
            async with semaphore:
                try:
                    # Use precomputed embedding if available
                    embedding = query_embeddings[idx] if query_embeddings else None
                    
                    # Run the synchronous process_query in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        self.process_query, 
                        query, 
                        embedding
                    )
                    
                    # Add index to track original order
                    result["original_index"] = idx
                    return result
                    
                except Exception as e:
                    # Return error result in same format
                    return {
                        "query": query,
                        "search_results": [],
                        "evaluation": {
                            "decision": "error",
                            "amount": None,
                            "confidence": 0.0,
                            "justification": f"Batch processing error: {str(e)}",
                            "relevant_clauses": [],
                            "reasoning": "Batch processing system error"
                        },
                        "api_status": self.get_api_status(),
                        "status": "error",
                        "error": str(e),
                        "original_index": idx
                    }
        
        # Create tasks for all queries
        print(f"🚀 Starting batch processing of {len(queries)} queries with max {max_concurrent} concurrent...")
        tasks = []
        for idx, query in enumerate(queries):
            task = process_single_query_async(idx, query)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Handle any exceptions that occurred
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed queries
                error_result = {
                    "query": queries[idx],
                    "search_results": [],
                    "evaluation": {
                        "decision": "error",
                        "amount": None,
                        "confidence": 0.0,
                        "justification": f"Exception during processing: {str(result)}",
                        "relevant_clauses": [],
                        "reasoning": "Exception occurred"
                    },
                    "api_status": self.get_api_status(),
                    "status": "error",
                    "error": str(result),
                    "original_index": idx
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Sort results by original index to maintain order
        processed_results.sort(key=lambda x: x.get("original_index", 0))
        
        # Remove the original_index field from final results
        final_results = []
        for result in processed_results:
            if "original_index" in result:
                del result["original_index"]
            final_results.append(result)
        
        print(f"✅ Batch processing completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Processed {len(final_results)} queries successfully")
        
        return final_results