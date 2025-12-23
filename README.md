# FileSearch

A semantic file search application using vector embeddings to search through PDF and DOCX documents.

## Features

- **Semantic Search**: Uses FAISS and sentence transformers for intelligent content-based search
- **Batch Indexing**: Efficiently indexes large document collections (350 files per batch)
- **Incremental Updates**: Only re-indexes modified files
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

Edit `search_directory` in `main.py` to change the directory being indexed.

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

2. **Search Improvements**

   - Re-ranking for more accurate results
   - Filename keyword boosting
   - Distance threshold filtering

3. **UI Enhancement**

   - Modern, clean user interface
   - Better result visualization
   - Search history

4. **Performance**
   - Watchdog integration for real-time file monitoring
   - Automatic re-indexing on file changes
   - Faster search response times

## Technical Details

- **Vector Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Database**: SQLite with SQLAlchemy ORM
- **Chunking**: 400 characters with 50 character overlap
- **Batch Size**: 512 chunks per encoding batch
  > > > > > > > feat/semantic_search_faiss
