# enhanced_backend.py
"""
Enhanced backend with improved FAISS implementation and better error handling.
"""
from fastapi import FastAPI, File, UploadFile, Form, Request, Header, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import requests
from dotenv import load_dotenv
from src.pipeline import process_all_documents_pipeline, query_documents_batch_sync
from src.embed_and_index import generate_query_embedding_pinecone
from pydantic import BaseModel
from typing import List, Optional
import time
import asyncio
import datetime
import json
import shutil
import threading
import concurrent.futures

load_dotenv()
app = FastAPI(title="Enhanced HackRx Insurance API", description="Enhanced API for querying insurance PDFs with FAISS")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()
API_TOKEN = "552a90e441d8b2a0c195b5425dd982e0e71292568a08d2facf1ebc9434c1bcd0"

class QueryPDFRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]  # List of questions to answer

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the token provided in the authorization header."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.post("/hackrx/run")
async def enhanced_query_pdf(input: QueryPDFRequest, token: str = Depends(verify_token)):
    """Enhanced endpoint with improved FAISS implementation and multithreading."""
    total_start_time = time.time()
    
    pdf_url = input.documents
    queries = input.questions
    
    if not pdf_url or not queries or not isinstance(queries, list):
        return JSONResponse({
            "error": "documents URL and questions (list) are required",
            "answers": ["Error: Invalid input"] * len(queries) if queries else [],
            "success": False
        }, status_code=400)

    # Performance tracking
    performance_stats = {
        "pdf_download": 0,
        "pdf_processing": 0,
        "embedding_time": 0,
        "query_processing_time": 0,
        "cleanup_time": 0
    }
    
    # Multithreading performance tracking
    multithreading_stats = {
        "embedding_workers": min(len(queries), 10),
        "query_workers": min(len(queries), 10),
        "individual_embedding_times": [],
        "individual_query_times": [],
        "bottleneck_analysis": {}
    }

    # Create directories
    tmpdir = tempfile.mkdtemp()
    pdf_storage_dir = "stored_pdfs"
    os.makedirs(pdf_storage_dir, exist_ok=True)
    
    print(f"📁 Created temporary directory: {tmpdir}")

    # Generate unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"input_{timestamp}.pdf"
    pdf_path = os.path.join(pdf_storage_dir, pdf_filename)

    # Step 1: Download PDF
    try:
        download_start = time.time()
        print(f"📥 Downloading PDF from: {pdf_url}")
        
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
        
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        
        # Copy to temp directory
        tmpdir_pdf_path = os.path.join(tmpdir, pdf_filename)
        shutil.copy2(pdf_path, tmpdir_pdf_path)
        
        performance_stats["pdf_download"] = time.time() - download_start
        print(f"📄 Downloaded PDF to: {pdf_path}")
        print(f"📄 Copied PDF to temp directory: {tmpdir_pdf_path}")
        
    except Exception as e:
        error_msg = f"Failed to download PDF: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse({
            "error": error_msg,
            "answers": ["Error: PDF download failed"] * len(queries),
            "success": False,
            "total_time_taken": time.time() - total_start_time
        }, status_code=400)

    # Step 2: Multithreaded Query Embedding
    embedding_start = time.time()
    
    print(f"🧵 Starting multithreaded embedding for {len(queries)} queries...")
    
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        # Multithreaded embedding generation
        all_embeddings = []
        embedding_times = []
        embedding_workers = min(len(queries), 10)
        
        def embed_single_query(query_data):
            query_idx, query = query_data
            thread_id = threading.get_ident()
            
            embed_start = time.time()
            print(f"🧵 Thread {thread_id}: Starting embedding for query {query_idx + 1}")
            
            try:
                embedding = generate_query_embedding_pinecone(query, pinecone_key)
                embed_time = time.time() - embed_start
                
                print(f"✅ Thread {thread_id}: Completed embedding for query {query_idx + 1} in {embed_time:.2f}s")
                return (query_idx, embedding, embed_time)
                
            except Exception as e:
                embed_time = time.time() - embed_start
                print(f"❌ Thread {thread_id}: Error embedding query {query_idx + 1}: {e}")
                # Return dummy embedding
                return (query_idx, [0.0] * 1024, embed_time)
        
        print(f"🚀 Starting multithreaded embedding for {len(queries)} queries with {embedding_workers} workers")
        
        # Process embeddings with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=embedding_workers) as executor:
            query_data = list(enumerate(queries))
            embedding_results = list(executor.map(embed_single_query, query_data))
        
        # Sort results by original index and extract data
        embedding_results.sort(key=lambda x: x[0])
        all_embeddings = [result[1] for result in embedding_results]
        embedding_times = [result[2] for result in embedding_results]
        
        multithreading_stats["individual_embedding_times"] = embedding_times
        
        total_embedding_time = time.time() - embedding_start
        performance_stats["embedding_time"] = total_embedding_time
        
        # Calculate speedup
        sequential_time = sum(embedding_times)
        speedup = sequential_time / total_embedding_time if total_embedding_time > 0 else 1
        multithreading_stats["embedding_speedup"] = speedup
        
        print(f"✅ Completed multithreaded embedding in {total_embedding_time:.2f}s")
        print(f"📊 Thread assignments: {[f'Query {i+1}: Thread {threading.get_ident()}' for i in range(len(queries))]}")
        print(f"📊 Individual embedding times: {[f'{t:.2f}s' for t in embedding_times]}")
        
    except Exception as e:
        error_msg = f"Failed to generate embeddings: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse({
            "error": error_msg,
            "answers": ["Error: Embedding generation failed"] * len(queries),
            "success": False,
            "total_time_taken": time.time() - total_start_time
        }, status_code=500)

    # Step 3: Process PDF
    try:
        processing_start = time.time()
        print(f"🚀 Starting hybrid parsing for {pdf_filename}...")
        
        result = await process_all_documents_pipeline(
            docs_dir=tmpdir,
            pinecone_api_key=pinecone_key,
            force_reprocess=True
        )
        
        performance_stats["pdf_processing"] = time.time() - processing_start
        
        if not result.get("success"):
            error_msg = result.get("error", "Pipeline failed")
            print(f"❌ PDF processing failed: {error_msg}")
            return JSONResponse({
                "error": f"PDF processing failed: {error_msg}",
                "answers": ["Error: PDF processing failed"] * len(queries),
                "success": False,
                "total_time_taken": time.time() - total_start_time
            }, status_code=500)
        
        print("✅ PDF processing and indexing complete!")
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse({
            "error": error_msg,
            "answers": ["Error: PDF processing error"] * len(queries),
            "success": False,
            "total_time_taken": time.time() - total_start_time
        }, status_code=500)

    # Step 4: Multithreaded Query Processing
    query_processing_start = time.time()
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print(f"🧵 Starting multithreaded query processing for {len(queries)} queries...")
    
    try:
        query_results = []
        query_times = []
        query_workers = min(len(queries), 10)
        
        def process_single_query(query_data):
            query_idx, query, embedding = query_data
            thread_id = threading.get_ident()
            
            query_start = time.time()
            print(f"🧵 Thread {thread_id}: Starting query processing for query {query_idx + 1}")
            
            try:
                from src.pipeline import query_documents_sync
                
                result = query_documents_sync(
                    query=query,
                    pinecone_api_key=pinecone_key,
                    gemini_api_key=gemini_key,
                    index_name="policy-index",
                    query_embedding=embedding
                )
                
                query_time = time.time() - query_start
                print(f"✅ Thread {thread_id}: Completed query processing for query {query_idx + 1} in {query_time:.2f}s")
                
                return (query_idx, result, query_time)
                
            except Exception as e:
                query_time = time.time() - query_start
                print(f"❌ Thread {thread_id}: Error processing query {query_idx + 1}: {e}")
                
                error_result = {
                    "evaluation": {
                        "answer": f"Query processing error: {str(e)}",
                        "confidence": 0.0
                    },
                    "success": False
                }
                return (query_idx, error_result, query_time)
        
        print(f"🚀 Starting multithreaded query processing for {len(queries)} queries with {query_workers} workers")
        
        # Process queries with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=query_workers) as executor:
            query_data = list(zip(range(len(queries)), queries, all_embeddings))
            query_processing_results = list(executor.map(process_single_query, query_data))
        
        # Sort results by original index
        query_processing_results.sort(key=lambda x: x[0])
        query_results = [result[1] for result in query_processing_results]
        query_times = [result[2] for result in query_processing_results]
        
        multithreading_stats["individual_query_times"] = query_times
        
        total_query_processing_time = time.time() - query_processing_start
        performance_stats["query_processing_time"] = total_query_processing_time
        
        # Calculate speedup
        sequential_query_time = sum(query_times)
        query_speedup = sequential_query_time / total_query_processing_time if total_query_processing_time > 0 else 1
        multithreading_stats["query_speedup"] = query_speedup
        
        # Bottleneck analysis
        max_query_time = max(query_times) if query_times else 0
        min_query_time = min(query_times) if query_times else 0
        time_variance = max_query_time - min_query_time
        parallel_efficiency = query_speedup / query_workers if query_workers > 0 else 0
        
        multithreading_stats["bottleneck_analysis"] = {
            "slowest_query_time": max_query_time,
            "fastest_query_time": min_query_time,
            "time_variance": time_variance,
            "parallel_efficiency": parallel_efficiency
        }
        
        print(f"✅ Completed multithreaded query processing in {total_query_processing_time:.2f}s")
        print(f"📊 Thread assignments: {[f'Query {i+1}: Thread {threading.get_ident()}' for i in range(len(queries))]}")
        print(f"📊 Individual query times: {[f'{t:.2f}s' for t in query_times]}")
        print(f"📊 Max query time: {max_query_time:.2f}s, Parallel execution time: {total_query_processing_time:.2f}s")
        
    except Exception as e:
        error_msg = f"Error in query processing: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse({
            "error": error_msg,
            "answers": ["Error: Query processing failed"] * len(queries),
            "success": False,
            "total_time_taken": time.time() - total_start_time
        }, status_code=500)

    # Step 5: Cleanup
    cleanup_start = time.time()
    try:
        print("✅ FAISS cleanup skipped - using local file storage")
        performance_stats["cleanup_time"] = time.time() - cleanup_start
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        performance_stats["cleanup_time"] = time.time() - cleanup_start

    # Extract answers and prepare response
    answers = []
    similarity_vectors = []
    llm_contexts = []  # New: Store LLM contexts
    source_vectors_info = []  # New: Store source vectors info
    
    for result in query_results:
        if result and "evaluation" in result:
            answer = result["evaluation"].get("answer", "No answer found")
            answers.append(answer)
            
            # Extract LLM context and source vectors
            llm_context = result["evaluation"].get("llm_context", "")
            source_vectors = result["evaluation"].get("source_vectors", [])
            
            llm_contexts.append(llm_context)
            source_vectors_info.append(source_vectors)
            
            # Extract similarity vectors if available
            search_results = result.get("search_results", [])
            if search_results:
                # Get similarity scores from search results
                scores = [r.get("similarity_score", 0.0) for r in search_results[:5]]
                similarity_vectors.append(scores)
            else:
                similarity_vectors.append([])
        else:
            answers.append("No answer found")
            similarity_vectors.append([])
            llm_contexts.append("")  # Empty context for failed queries
            source_vectors_info.append([])  # Empty source vectors for failed queries

    # Calculate total time
    total_time = time.time() - total_start_time

    # Prepare comprehensive response
    response_data = {
        "answers": answers,
        "similarity_vectors": similarity_vectors,
        "llm_contexts": llm_contexts,  # New: Full context used by LLM
        "source_vectors": source_vectors_info,  # New: Vectors used to build context
        "total_time_taken": total_time,
        "timing_breakdown": performance_stats,
        "multithreading_performance": multithreading_stats,
        "vector_info": {
            "total_queries": len(queries),
            "query_embedding_dimension": 1024,
            "embedding_model": "multilingual-e5-large",
            "total_similarity_results": sum(len(sv) for sv in similarity_vectors),
            "context_info": {
                "total_contexts_provided": len(llm_contexts),
                "average_context_length": sum(len(ctx) for ctx in llm_contexts) / len(llm_contexts) if llm_contexts else 0,
                "total_source_vectors": sum(len(sv) for sv in source_vectors_info)
            }
        }
    }

    # Log comprehensive data
    try:
        logs_dir = "request_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request": {
                "pdf_url": pdf_url,
                "questions": queries,
                "pdf_filename": pdf_filename
            },
            "response": response_data,
            "processing_summary": {
                "success": True,
                "total_questions": len(queries),
                "total_time_seconds": total_time,
                "pdf_processing_time": performance_stats["pdf_processing"],
                "embedding_time": performance_stats["embedding_time"],
                "query_processing_time": performance_stats["query_processing_time"]
            }
        }
        
        log_filename = f"multithread_log_{timestamp}.json"
        log_path = os.path.join(logs_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Complete request and response logged to: {log_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save request log: {e}")

    # Print performance summary
    print(f"🧵 Multithreading Performance:")
    print(f"   📊 Embedding: {len(queries)} queries in {performance_stats['embedding_time']:.2f}s with {embedding_workers} threads")
    print(f"   📊 Query Processing: {len(queries)} queries in {performance_stats['query_processing_time']:.2f}s with {query_workers} threads") 
    print(f"   📊 Embedding Speedup: {multithreading_stats.get('embedding_speedup', 1.0):.2f}x")
    print(f"   📊 Query Speedup: {multithreading_stats.get('query_speedup', 1.0):.2f}x")

    # Return only the list of answers as the final response to the client
    return JSONResponse(content={"answers": answers})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/stats")
async def get_stats():
    """Get FAISS index statistics."""
    try:
        from src.faiss_storage import FAISSVectorStore
        
        vector_store = FAISSVectorStore("policy-index")
        stats = vector_store.get_stats()
        
        return JSONResponse({
            "success": True,
            "faiss_stats": stats,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enhanced_backend:app", host="0.0.0.0", port=8001, reload=True)
