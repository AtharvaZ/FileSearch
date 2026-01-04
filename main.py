import os
from database import *
from file_reader import CODE_FILE_EXTENSIONS, TEXT_FILE_EXTENSIONS

# Lazy imports - only import when needed to avoid loading ML models on startup
def _lazy_import_vector_search():
    global load_index, save_index, remove_file_from_index
    global add_file_to_index, search_similar_with_reranking
    from vector_search import load_index, save_index, remove_file_from_index
    from vector_search import add_file_to_index, search_similar_with_reranking

def _lazy_import_background_indexer():
    global add_to_background_queue, start_background_indexer
    from background_indexer import add_to_background_queue, start_background_indexer

search_directory = "/Users/atharvazaveri/"
SUPPORTED_EXTENSIONS = tuple(['.pdf', '.docx', '.pptx'] + list(TEXT_FILE_EXTENSIONS) + list(CODE_FILE_EXTENSIONS))
MAX_FILE_SIZE_MB = 10  # Skip files larger than 10MB
FILE_READ_TIMEOUT = 5  # Timeout in seconds for file reading

def get_file_content(path: str) -> list[str]:
    all_paths_content = []
    print(f"Scanning directory: {path}")

    # Skip directories that typically contain many files but aren't useful for search
    SKIP_DIRS = {
        # System directories
        'Library', 'Applications', 'System',
        # Development bloat
        'node_modules', 'venv', 'env', '__pycache__',
        '.git', '.vscode', '.idea', 'cache', 'Cache',
        'build', 'dist', '.next', '.nuxt',
        # Media and downloads (usually not searchable text)
        'Downloads', 'Movies', 'Music', 'Pictures', 'Photos',
        # Package managers and build artifacts
        '.cargo', '.rustup', '.npm', '.gradle', '.m2',
        'target', 'bin', 'obj', 'out', 'vendor',
        # Logs and temp files
        'Logs', 'logs', 'tmp', 'temp', 'Temp',
        # iOS/Android development
        'Pods', 'DerivedData', 'Android', 'Gradle',
    }

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and bloat directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]

        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.abspath(os.path.join(root, file))
                # Skip files larger than MAX_FILE_SIZE_MB
                try:
                    if os.path.getsize(file_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
                        continue
                except OSError:
                    continue  # Skip if we can't get file size
                all_paths_content.append(file_path)

    print(f"Found {len(all_paths_content)} files")
    return all_paths_content
    

def index_files():
    import time

    # Import vector search and background indexer modules
    _lazy_import_vector_search()
    _lazy_import_background_indexer()

    print("Loading existing FAISS index...")
    load_index()

    # Get all file paths from directory
    all_file_paths = get_file_content(search_directory)
    print(f"Found {len(all_file_paths)} files to process")

    # Check for deleted files and remove from index
    print("\nChecking for deleted files...")
    db_files = get_all_files()
    for db_file in db_files:
        if not os.path.exists(db_file.file_path):
            faiss_idx = delete_file_by_path(db_file.file_path)
            if faiss_idx is not None:
                remove_file_from_index(faiss_idx)
                print(f"Removed deleted file: {db_file.file_name}")

    # Get next available FAISS index
    existing_indices = [f.faiss_index for f in db_files if os.path.exists(f.file_path)]
    next_faiss_index = max(existing_indices) + 1 if existing_indices else 1

    # Categorize files: new vs existing (for incremental updates)
    print("\nCategorizing files...")
    new_files = []
    modified_files = []
    unchanged_count = 0

    for file_path in all_file_paths:
        existing_file = get_file_by_path(file_path)
        if existing_file:
            # File exists in DB - will check hash during streaming
            modified_files.append((existing_file.faiss_index, file_path, existing_file))
        else:
            # New file
            new_files.append((next_faiss_index, file_path))
            next_faiss_index += 1

    print(f"Files to process: {len(new_files)} new, {len(modified_files)} existing")

    # Process all files using streaming pipeline
    try:
        all_files_to_process = new_files + modified_files

        if not all_files_to_process:
            print("\nNo files to process (all files up to date)")
            return

        print(f"\nProcessing {len(all_files_to_process)} files using streaming pipeline...")
        print(f"  {len(new_files)} new, {len(modified_files)} to check for modifications")

        # Stream processing: read → chunk → encode → index (all in parallel)
        from streaming_pipeline import StreamingPipeline, StreamingIndexWriter
        from vector_search import encode_chunks

        pipeline = StreamingPipeline(
            encode_function=encode_chunks,
            chunk_size=500,
            overlap=50,
            micro_batch_size=1024,
            queue_maxsize=4
        )

        writer = StreamingIndexWriter()

        # Prepare streaming data
        streaming_data = []
        for item in all_files_to_process:
            if len(item) == 2:  # New file: (faiss_index, file_path)
                streaming_data.append(item)
            else:  # Modified file: (faiss_index, file_path, existing_file_record)
                streaming_data.append((item[0], item[1]))

        total_chunks = 0
        batch_count = 0
        files_processed = 0
        stream_start = time.time()

        print(f"\n  Starting streaming pipeline (read → chunk → encode → index)...")

        # Process encoded batches as they arrive
        for encoded_batch in pipeline.process_files(streaming_data, timeout_seconds=FILE_READ_TIMEOUT):
            batch_count += 1

            # Add to FAISS and update database
            result = writer.write_batch(
                encoded_batch,
                new_files,
                modified_files,
                all_files_to_process
            )

            total_chunks += result['chunks_added']
            files_processed += result['files_processed']

            # Progress update every 10 batches
            if batch_count % 10 == 0:
                elapsed = time.time() - stream_start
                rate = total_chunks / elapsed if elapsed > 0 else 0
                print(f"    Progress: {files_processed} files, {total_chunks} chunks ({rate:.0f} chunks/s)")

        stream_time = time.time() - stream_start
        final_rate = total_chunks / stream_time if stream_time > 0 else 0
        print(f"\n  ✓ Completed: {files_processed} files, {total_chunks} chunks in {stream_time:.1f}s ({final_rate:.0f} chunks/s)")
        print(f"  Summary: {writer.new_count} new, {writer.modified_count} modified, {writer.unchanged_count} unchanged")

        # Handle timeout files
        if pipeline.timeout_files:
            print(f"\n⏱️  {len(pipeline.timeout_files)} files timed out - queuing for background indexing...")
            start_background_indexer()
            for file_path in pipeline.timeout_files:
                add_to_background_queue(file_path, timeout_seconds=60)
            print(f"   Background indexer started")

        # Report failed files
        if pipeline.failed_files:
            print(f"\n⚠️  {len(pipeline.failed_files)} files failed:")
            for path, error in pipeline.failed_files[:5]:
                print(f"  - {os.path.basename(path)}: {error}")
            if len(pipeline.failed_files) > 5:
                print(f"  ... and {len(pipeline.failed_files) - 5} more")

    finally:
        # Always save the index, even if there were errors during processing
        print("\nSaving FAISS index...")
        save_index()
        print("FAISS index saved!")

def search_files(query: str, k: int = 5, exclude_code_files: bool = True):
    """Search files using vector semantic search

    Args:
        query: Search query string
        k: Number of results to return
        exclude_code_files: If True, filters out code files (.py, .js, etc.) from results
    """
    # Import vector search module
    _lazy_import_vector_search()

    load_index()
    # Request more chunks to ensure we get k unique files after deduplication and filtering
    # Use 5x multiplier to account for both deduplication and code file filtering
    search_results = search_similar_with_reranking(query, k=k*5)

    if not search_results:
        print(f"No results found for '{query}'")
        return []

    # Get unique file indices and retrieve from database
    file_indices = list(set([r['file_faiss_index'] for r in search_results]))
    files = get_files_by_faiss_indices(file_indices)

    results = []
    print(f"Found {len(files)} file(s) with {len(search_results)} matching chunks\n")

    for file_record in files:
        file_chunks = [r for r in search_results if r['file_faiss_index'] == file_record.faiss_index]
        best_match = min(file_chunks, key=lambda x: x['distance'])

        results.append({
            'file_name': file_record.file_name,
            'file_path': file_record.file_path,
            'distance': best_match['distance'],
            'chunk_text': best_match['chunk_text'],
            'rerank_score': best_match.get('rerank_score', 0)
        })

    # Sort by rerank score (higher is better)
    results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

    # Filter out code files if requested
    if exclude_code_files:
        original_count = len(results)
        results = [r for r in results if not r['file_path'].endswith(tuple(CODE_FILE_EXTENSIONS))]
        filtered_count = original_count - len(results)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} code file(s)\n")

    return results[:k]

if __name__ == "__main__":
    import sys

    Base.metadata.create_all(engine)

    # Check if GUI mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        import tkinter as tk
        from app import FileSearchApp
        root = tk.Tk()
        app = FileSearchApp(root)
        root.mainloop()
    else:
        # CLI mode
        print("Indexing files...")
        index_files()
        print("Indexing complete!\n")