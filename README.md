# FileSearch

Ever spent 20 minutes hunting for that PDF you *know* you saved somewhere? Yeah, me too.

FileSearch uses AI-powered semantic search to find your documents by *what they mean*, not just what they're named. Built with FAISS and sentence transformers, it's like having a photographic memory for all your files - minus the superhero origin story.

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

Edit these in `main.py` if you're feeling adventurous:
- `search_directory`: What to index (default: your entire home directory, you brave soul)
- `FILE_READ_TIMEOUT`: How long to wait before giving up (default: 5s)
- `MAX_FILE_SIZE_MB`: Skip chonky files (default: 10MB, your GPU will thank you)
- `SKIP_DIRS`: Directories to pretend don't exist (node_modules, we're looking at you)

## Project Structure

- `main.py` - The conductor. Orchestrates indexing, manages timeouts, queues background tasks
- `vector_search.py` - Where the magic happens. FAISS IVF-PQ + cross-encoder reranking
- `database.py` - SQLite keeping track of what goes where
- `file_reader.py` - Reads PDFs, DOCX, PPTX, code files, plain text - basically everything
- `app.py` - The pretty face (Tkinter GUI)
- `background_indexer.py` - Patient worker bee indexing slow files while you search
- `exceptions.py` - Timeout handlers (because some PDFs just don't want to cooperate)
- `diagnose.py` - Search CSI - investigate why your results look weird
- `fix_empty_files.py` - Rescue mission for files that failed to index

## Technical Details (For the Nerds)

**Models:**
- Embedding: `paraphrase-MiniLM-L3-v2` (384 dimensions, 2x faster than L6, good enough)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (because sometimes first impressions are wrong)

**Index Architecture:**
- Type: IVF-PQ (Inverted File with Product Quantization)
- Translation: Smart clustering + aggressive compression
- Compression: 8 subquantizers × 8 bits = your index just lost 87.5% of its weight
- Clusters: Auto-adjusted (4√n, maxes out at 1000 because we have standards)
- Search strategy: Check 50 clusters, grab 100 candidates, let the reranker pick the winners

**Pipeline Flow:**
1. Chunk text: 500 chars with 50 char overlap (plays nice with the 512 token limit)
2. Encode: 1024 chunks per batch (GPU goes brrrr)
3. Search: FAISS retrieves 100 candidates → Reranker scores them → You get top k
4. Deduplication: Multiple chunks from same file? Collapse them.

**The Timeout Dance:**
- Files reading in <5s: Indexed immediately, you don't even notice
- Files reading in >5s: "Hold on, I'll get back to you" → Background queue with 60s timeout
- Result: You search now, slow files catch up later

**Performance Numbers:**
- Apple Silicon (MPS): ~921 chunks/second
- Your mileage may vary (but probably won't be worse unless you're on a potato)

## Supported File Types

- **PDF** - Using pdfplumber (the fast one)
- **DOCX** - Microsoft Word documents
- **PPTX** - PowerPoint presentations
- **Code** - .py, .js, .java, .cpp, .c, .ts, .jsx, .tsx, .go, .rs, .rb (we see you, polyglots)
- **Text** - .txt, .md (the classics)

## Pro Tips

1. First search is slow (loading 2GB of models). Subsequent searches? Lightning fast.
2. Check "Exclude code files" if you're tired of seeing Python files when searching for documents
3. The diagnostic tool is your friend when results look sus
4. Index takes time proportional to how much random stuff you've accumulated. Blame past you.

## The Boring Legal Stuff

This is a personal project. Use at your own risk. If it breaks, you get to keep both pieces.
