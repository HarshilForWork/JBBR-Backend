# 🏥 Insurance Policy RAG System

This project is an advanced **retrieval-augmented generation (RAG)** system designed to process, index, and query insurance policy documents. It allows users to ask natural language questions about coverage, claims, and exclusions and receive accurate, citation-backed answers.

---

## 🏗️ Architecture Overview

The system is built on a modular pipeline architecture that decouples document ingestion from query processing.

### High-Level Flow
1. **Ingestion Phase**: PDFs ➡ Parsing ➡ Chunking ➡ Embedding ➡ Vector Storage (Pinecone/FAISS).
2. **Query Phase**: User Query ➡ Vector Search ➡ Context Retrieval ➡ LLM (Gemini) ➡ Answer Generation.
3. **LLMOps Phase**: Observability Hooks ➡ Background Evaluation ➡ Experiment Logging ➡ Alerting.

---

## 🛠️ Implementation Details

This section dives deep into how the core components are implemented in the `src/` directory.

### 1. Advanced Document Parsing (`src/parse_documents.py`)
Handling PDF documents, especially those with complex layouts like insurance policies, requires robust parsing.
*   **Hybrid Extraction Strategy**: We combine the strengths of two libraries:
    *   **`pdfplumber`**: Used specifically for high-fidelity **table detection and extraction**.
    *   **`PyMuPDF` (fitz)**: Used for fast and accurate **text extraction** and layout analysis.
*   **Intelligent Table Processing**:
    *   **Detection & Validation**: The system uses heuristics to validate if a detected structure is actually a table (checking row consistency, cell density, and ruling out list-like structures).
    *   **Contextual Headers**: For tables without explicit headers (common in policies), the system analyzes nearby text blocks to accurately infer and attach the correct header.
    *   **Markdown Conversion**: Extracted tables are converted into clean **Markdown format**. This preserves the structural relationship of the data (rows/cols) in a way that is token-efficient and understandable for the LLM.

### 2. Context-Aware Chunking (`src/chunk_documents_optimized.py`)
Standard fixed-size chunking often breaks the context. We implement a custom "Optimized Text Chunker" that respects document structure:
*   **Dedicated Table Chunking**: Tables are detected and **chunked separately** from the narrative text. This ensures that a table is never split in the middle, preserving the integrity of rows and columns for accurate retrieval.
*   **Hierarchical Strategy**: The specific chunking logic follows a priority hierarchy:
    1.  **Paragraph-based**: Tries to split by double newlines (`\n\n`) to keep paragraphs intact.
    2.  **Sentence-based**: Fallback to splitting by sentence boundaries if paragraphs are too long.
    3.  **Character-based**: Final fallback with overlap for unstructured text.
*   **Overlap**: We maintain a `chunk_overlap` (default: 150 tokens) to ensure that context isn't lost between adjacent chunks.

### 3. Embeddings & Indexing (`src/embed_and_index.py`)
The system employs a high-performance vectorization pipeline:
*   **Embedding Model**: We use **`multilingual-e5-large`** (via Pinecone Inference API) to generate high-quality 1024-dimensional vectors. This model is chosen for its superior semantic understanding across languages and document types.
*   **Batch Processing**: Embeddings are generated in batches (size: 96) to optimize throughput and respect API rate limits.
*   **Dual Vector Store Strategy**:
    *   **Pinecone (Cloud)**: The primary store for scalable, low-latency production use.
    *   **FAISS (Local)**: A fully functional local fallback used when offline or for lower latency in specific deployment scenarios. It uses `IndexFlatIP` (Inner Product) for efficient similarity matching.
*   **Duplicate Management**: The system calculates content hashes for every chunk to detect and prevent duplicate vectors, ensuring index hygiene.

### 4. Advanced Query Processing & Retrieval (`src/faiss_query_processor.py`)
The retrieval logic goes beyond simple similarity search to ensure accurate grounded answers.
*   **Search Pipeline**:
    1.  **Vector Search**: Retrieves the top $K$ (approx. 20) most similar chunks from the vector store using the query embedding.
    2.  **Reranking**: We apply **`bge-reranker-v2-m3`** to re-score the initial candidates. This step significantly improves relevance by assessing the exact match quality between query and document text, which vector similarity might miss.
*   **Extended Context Window**:
    *   To prevent "keyhole" issues (where a chunk misses surrounding context), the system effectively retrieves the **whole document section**.
    *   For the top reranked chunks, we fetch **25 chunks before and 25 chunks after** the target match. This provides the LLM with a massive, continuous context window (potentially covering entire policy sections) to answer comprehensive questions like "What are *all* the exclusions?".
*   **Batch Querying**: The system supports multithreaded batch processing for handling multiple queries simultaneously.

### 5. Answer Generation (LLM Integration)
We utilize **Google Gemini 2.5** as the reasoning engine to synthesize answers from the retrieved context.
*   **Models**: The system prioritizes **`gemini-2.5-flash`** for speed and reliability, with fallback support for **`gemini-2.5-pro`**.
*   **Prompt Engineering**:
    *   **Persona**: Acts as an "insurance policy expert".
    *   **Strict JSON Output**: The prompt enforces a strict JSON structure `{"answer": "..."}` to ensure deterministic and machine-readable responses.
    *   **Context usage**: Explicit instructions to synthesize information across multiple vector sections and check for exclusions/waiting periods.
*   **Robust Parsing**: A dedicated JSON extractor (`_extract_json_from_response`) handles potential LLM formatting errors (e.g., Markdown code blocks, relaxed syntax) to ensure the application never crashes on malformed model output.
*   **Safety Handling**: The system detects and gracefully handles safety/content policy blocks from the Gemini API.

---

### 6. High-Performance Concurrency Strategy
Performance is critical when handling large PDFs and complex RAG queries. We implement a multi-layered concurrency model:
*   **Async I/O (`asyncio`)**:
    *   Used for the primary PDF processing pipeline (`backend.py`).
    *   Allows non-blocking operations, such as handling file uploads and network requests while the core pipeline initializes.
*   **Parallel Query Processing (`enhanced_backend.py`)**:
    *   We utilize **`concurrent.futures.ThreadPoolExecutor`** to handle multiple user questions simultaneously.
    *   Instead of processing 5 queries sequentially (which would take Sum(T1...T5)), we run them in parallel threads, reducing total latency to approximately Max(T1...T5).
    *   The system includes a **speedup calculator** that logs the efficiency gain (e.g., "5.00s parallel vs 20.00s sequential").
* **Multithreaded Embedding Generation**:
  * Vector embedding generation is offloaded to a thread pool (max 10 workers) to maximize throughput against the Pinecone Inference API.
  * Prevents network bottlenecks during the initial vectorization phase.

### 7. Production LLMOps & MLOps (`src/ops/`)

The system is instrumented for production-grade monitoring and continuous improvement:

* **Deep Observability**: A dedicated Prometheus suite tracks system health (latencies, token costs, model confidence).
* **Experiment Tracking**: Every request is logged to **MLflow (via DagsHub)**, capturing every parameter and prompt version for full reproducibility.
* **Async Quality Evaluation**: Implements a non-blocking **RAGAS-style evaluator** that computes *Context Relevance* and *Faithfulness* heuristics in real-time without adding request latency.
* **Cost Management**: Granular token counting via `tiktoken` predicts costs per request and cumulative session spend.

---

## 💻 Tech Stack

* **Frontend**: Streamlit (Processing & Chat UI).
* **Core Logic**: Python 3.9+.
* **PDF Processing**:
  * `pdfplumber`: Advanced table extraction.
  * `PyMuPDF` (fitz): Layout and text extraction.
* **Vector Store**:
  * `Pinecone`: Cloud vector database.
  * `FAISS`: Efficient local vector search.
* **AI & ML**:
  * **LLM**: Google Gemini 2.0 (via `google-genai`).
  * **Embeddings**: `multilingual-e5-large` (via Pinecone Inference).
  * **Reranking**: `bge-reranker-v2-m3` (via Pinecone Inference).
  * **Guardrails**: Similarity-based abstain logic.
* **LLMOps & Monitoring**:
  * **MLflow + DagsHub**: Experiment tracking and artifact storage.
  * **Prometheus**: Real-time metric collection and `/metrics` endpoint.
  * **Tiktoken**: Precision token counting for cost estimation.
  * **YAML Config**: Centralized threshold management.
* **Utilities**: `pandas`, `numpy`, `PyYAML`, `python-dotenv`.
* **Orchestration**: Custom `DocumentPipeline` implementation.

---

## 📂 Project Structure

```bash
JBBR-Backend/
├── final_backend.py            # Main high-performance FastAPI server
├── config.yaml                 # Central system configuration
├── requirements.txt            # Project dependencies
├── src/                        # Core source code
│   ├── ops/                    # [NEW] LLMOps layer (Metrics, Eval, Tracking)
│   ├── retrieval/              # Smart vector retrieval & reranking
│   ├── inference/              # LLM providers & prompt logic
│   ├── indexing/               # Document versioning & registry
│   ├── parse_documents.py      # PDF parsing logic
│   ├── pipeline.py             # Orchestration logic
│   └── ...
├── logs/                       # System alerts and evaluation results
├── request_logs/               # Detailed JSON logs of every API call
└── faiss_storage/              # Local FAISS index storage
```

---

## 🚀 Setup & Usage

### 1. Prerequisites
*   Python 3.9 or higher.
*   API Keys for **Pinecone** and **Google Gemini**.

### 2. Installation
```bash
# Clone the repository
git clone <repository_url>

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Set up your secrets. You can create a `.env` file or use Streamlit secrets (`.streamlit/secrets.toml`) with the following keys:
```ini
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

### 4. Running the App
```bash
streamlit run app.py
```

### 5. Using the System
1.  **Upload**: Provide a PDF URL or upload via the API.
2.  **Process**: The system downloads, parses, and indexes the document into an isolated FAISS session.
3.  **Query**: Submit natural language questions (e.g., "Is dental surgery covered?").
4.  **Review**: See the answer, confidence score, source citations, and check the MLflow dashboard for eval metrics.

---

## 📊 Monitoring & Reliability

### Prometheus Metrics

Exposed at `/metrics`. Tracks:

* `rag_pipeline_duration_seconds`: Latency breakdown by stage.
* `rag_token_usage_total`: Exact consumption by model.
* `rag_active_pipelines`: Real-time concurrency gauge.
* `rag_empty_retrieval_total`: Signal for indexing issues.

### Guardrails

* **Similarity Threshold**: If top search results are below $0.15$, the system abstains from answering to prevent hallucinations.
* **Similarity Threshold**: If top search results are below $0.15$, the system abstains from answering to prevent hallucinations.
* **AlertManager**: Automatically triggers alerts (`HIGH_LATENCY`, `LOW_CONFIDENCE`, `METRIC_DRIFT`) into `logs/alerts.jsonl`.
* **Background Evaluation**: Detailed RAG quality metrics (faithfulness, relevance) logged to `logs/eval_results.jsonl`. No external Ragas server is required; evaluation runs optimally in the backend daemon threads.

---

## 📈 Running the Observability Stack (Local)

To view real-time metrics in Grafana without installing software directly on your machine, you can run Prometheus and Grafana via Docker.

### 1. Requirements

* [Docker Desktop](https://docs.docker.com/desktop/) installed and running on your machine.
* The JBBR-Backend must be running (e.g., `uv run python final_backend.py` on port 8085).

### 2. Start the Stack

Run the following command in the `JBBR-Backend` directory where the `docker-compose.yml` file is located:

```bash
docker compose up -d
```

### 3. Access the Dashboards

* **Prometheus**: Go to [http://localhost:9090](http://localhost:9090)
  * Go to **Status > Targets** to ensure it successfully connected to `host.docker.internal:8085`.
* **Grafana**: Go to [http://localhost:3000](http://localhost:3000)
  * **Login**: Username `admin`, Password `admin`.
  * **Add Data Source**: Go to Data Sources, add Prometheus, and set the URL to `http://prometheus:9090`.
  * **Create Dashboard**: You can now create visual dashboards for metrics like `rag_pipeline_duration_seconds` and `rag_token_usage_total`.

To stop the monitoring stack, run:
```bash
docker compose down
```
