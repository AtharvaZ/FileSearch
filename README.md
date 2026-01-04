# FileSearch

Ever spent 20 minutes hunting for that PDF you _know_ you saved somewhere? Yeah, me too.

FileSearch uses AI-powered semantic search to find your documents by _what they mean_, not just what they're named. Built with FAISS and sentence transformers, it's like having a photographic memory for all your files - minus the superhero origin story.

## Features

- **Semantic Search**: FAISS and sentence transformers for intelligent content-based search
- **Two-Stage Reranking**: Cross-encoder reranking for improved search accuracy (20-30% fewer false positives)
- **Product Quantization**: IVF-PQ compression for 8-16x less storage with minimal accuracy loss
- **Background Indexing**: Search immediately while slow files continue indexing in the background
- **Fast PDF Reading**: pdfplumber for 2-3x faster PDF text extraction
- **GPU Acceleration**: Automatic GPU support (CUDA/MPS) for 10-30x faster encoding
- **Parallel File Reading**: Multiprocessing for fast file reading across CPU cores
- **Batch Encoding**: Efficiently encodes large document collections (1024 chunks per batch)
- **Smart Directory Skipping**: Automatically skips 30+ common bloat directories (node_modules, Downloads, Library, etc.)
- **Code File Filtering**: Optional exclusion of code files from search results
- **Lazy Loading**: Instant GUI startup - models load only when needed
- **Incremental Updates**: Only re-indexes modified files with hash-based change detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Index your files:**

```bash
python main.py
```

**Search with GUI:**

```bash
python main.py --gui
```

**Nuclear option (reset everything and re-index):**

```bash
./reset_and_reindex.sh
```

**Found files with 0 chunks? Fix them:**

```bash
python fix_empty_files.py
```

**Debug search results:**

```bash
python diagnose.py "your search query"
```

See exactly why you're getting weird results - PQ compression? Reranker? Embedding model? Find out!

## Configuration

Edit these variables in [main.py](main.py) to customize:

- `search_directory`: Directory to index (default: `/Users/atharvazaveri/`)
- `MAX_FILE_SIZE_MB`: Skip files larger than this size (default: 10MB)
- `SKIP_DIRS`: Directories to ignore during indexing

**Note on Index Training:**
The IVF-PQ index requires training before use. The system automatically:

1. Collects embeddings from first few batches
2. Trains the index when enough data is collected (typically first 1-2 batches)
3. Adds all vectors to the trained index
4. Subsequent runs use the pre-trained index for fast indexing

## Project Structure

- `main.py` - Main indexing and search logic
- `vector_search.py` - FAISS vector search implementation
- `database.py` - SQLite database operations
- `file_reader.py` - PDF/DOCX file parsing
- `app.py` - Tkinter GUI
- `dry_run.py` - File preview utility

## Roadmap

### Planned Features

1. **UI Enhancement**

   - Modern, clean user interface
   - Better result visualization
   - Search history

2. **Performance**
   - Watchdog integration for real-time file monitoring
   - Automatic re-indexing on file changes
   - Faster search response times

## Technical Details

- **Embedding Model**: paraphrase-MiniLM-L3-v2 (384 dimensions, 2x faster than L6)
- **Reranker Model**: cross-encoder/ms-marco-MiniLM-L-6-v2 (ensures high accuracy)
- **FAISS Index**: IVF-PQ (Inverted File with Product Quantization)
  - **Compression**: 8 subquantizers × 8 bits = 8x compression ratio
  - **Clusters**: Auto-adjusted based on data size (√n to 4√n)
  - **Storage**: ~8-16x less than flat index with 95-98% accuracy
- **Database**: SQLite with SQLAlchemy ORM
- **Chunking**: 500 characters with 50 character overlap (~125 tokens, respects model limit)
- **Batch Size**: 512 chunks per encoding batch (auto-reduces on OOM)
- **Search Pipeline**: IVF-PQ retrieval (20 candidates) → Cross-encoder reranking → Top 5 results
