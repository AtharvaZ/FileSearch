# FileSearch

Ever spent 20 minutes hunting for that PDF you *know* you saved somewhere? Yeah, me too.

FileSearch uses AI-powered semantic search to find your documents by *what they mean*, not just what they're named. Built with FAISS and sentence transformers, it's like having a photographic memory for all your files - minus the superhero origin story.

## Features

- **Semantic Search**: Uses FAISS and sentence transformers for intelligent content-based search
- **Two-Stage Reranking**: Cross-encoder reranking for improved search accuracy (20-30% fewer false positives)
- **Product Quantization**: IVF-PQ compression for 8-16x less storage with minimal accuracy loss
- **GPU Acceleration**: Automatic GPU support (CUDA/MPS) for 10-30x faster encoding
- **Parallel Processing**: Multiprocessing for fast file reading
- **Batch Indexing**: Efficiently indexes large document collections (1000 files per batch)
- **Incremental Updates**: Only re-indexes modified files with hash-based change detection
- **Smart Directory Skipping**: Automatically skips 30+ common bloat directories (node_modules, Downloads, Library, etc.)
- **File Size Filtering**: Skips files larger than 10MB to prevent slowdowns
- **OOM Protection**: Automatic batch size reduction on GPU memory errors
- **GUI Interface**: Simple Tkinter-based interface for easy searching
- **Dual Mode**: CLI for indexing, GUI for searching

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Index files:**

```bash
python main.py
```

**Search with GUI:**

```bash
python main.py --gui
```

**Preview files to be indexed:**

```bash
python dry_run.py
```

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

## Current Status

⚠️ **Work in Progress** - This project is actively being developed.

## Project Structure

- `main.py` - Main indexing and search logic
- `vector_search.py` - FAISS vector search implementation
- `database.py` - SQLite database operations
- `file_reader.py` - PDF/DOCX file parsing
- `app.py` - Tkinter GUI
- `dry_run.py` - File preview utility

## Roadmap

**Target Completion: December 26th, 2024**

### Planned Features

1. **Additional File Format Support**

   - Code files (.py, .js, .java, etc.)
   - Plain text files (.txt)
   - PowerPoint presentations (.pptx)
   - Markdown files (.md)

2. **UI Enhancement**

   - Modern, clean user interface
   - Better result visualization
   - Search history

3. **Performance**
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
