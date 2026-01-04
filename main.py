import os
from multiprocessing import Pool, cpu_count
import signal
from database import *
from database import create_files_bulk
from file_reader import read_file, CODE_FILE_EXTENSIONS, TEXT_FILE_EXTENSIONS
from exceptions import TimeoutException, timeout_handler

# Lazy imports - only import when needed to avoid loading ML models on startup
def _lazy_import_vector_search():
    global load_index, save_index, remove_file_from_index
    global add_file_to_index, add_files_to_index_batch, search_similar_with_reranking
    from vector_search import load_index, save_index, remove_file_from_index
    from vector_search import add_file_to_index, add_files_to_index_batch, search_similar_with_reranking

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
    
def check_modified_file(file_path: str, hashed_file: str) -> bool:
    existing_file = get_file_by_path(file_path)
    if not existing_file:
        return True
    return existing_file.file_hash != hashed_file

def _read_file_safe(file_path: str):
    """Wrapper for multiprocessing - catches exceptions per file with timeout"""
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(FILE_READ_TIMEOUT)

        try:
            result = read_file(file_path)
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutException:
            signal.alarm(0)
            return None, (file_path, "Timeout")
    except PermissionError:
        signal.alarm(0)
        return None, (file_path, "Permission denied")
    except Exception as e:
        signal.alarm(0)
        return None, (file_path, str(e))

def index_files():
    import time

    # Import vector search and background indexer modules
    _lazy_import_vector_search()
    _lazy_import_background_indexer()

    print("Loading existing FAISS index...")
    load_index()

    # Get all files from directory
    file_contents = get_file_content(search_directory)
    file_data = []
    failed_files = []

    # Use multiprocessing to read files in parallel (use 50% of cores to be conservative)
    num_workers = max(2, cpu_count() // 2)
    print(f"\nReading file contents using {num_workers} workers...")

    read_start = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(_read_file_safe, file_contents)

    # Separate successful reads from failures
    timeout_files = []
    for result in results:
        if result is None:
            continue
        if isinstance(result, tuple) and len(result) == 2 and result[0] is None:
            path, error = result[1]
            if error == "Timeout":
                timeout_files.append(path)
            else:
                failed_files.append(result[1])
        else:
            file_data.append(result)

    read_time = time.time() - read_start

    if failed_files:
        print(f"\n⚠️  Failed to read {len(failed_files)} files:")
        for path, error in failed_files[:5]:  # Show first 5 errors
            print(f"  - {os.path.basename(path)}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    if timeout_files:
        print(f"\n⏱️  {len(timeout_files)} files timed out (will retry after main indexing)")

    print(f"\nSuccessfully read {len(file_data)} files in {read_time:.1f}s")

    # Check for deleted files and remove from index
    db_files = get_all_files()
    for db_file in db_files:
        if not os.path.exists(db_file.file_path):
            faiss_idx = delete_file_by_path(db_file.file_path)
            if faiss_idx is not None:
                remove_file_from_index(faiss_idx)
                print(f"Removed deleted file from database and FAISS: {db_file.file_name}")

    existing_indices = [f.faiss_index for f in db_files if os.path.exists(f.file_path)]
    next_faiss_index = max(existing_indices) + 1 if existing_indices else 1

    # Separate files into modified and new
    print("\nCategorizing files...")
    new_files_batch = []
    modified_count = 0
    unchanged_count = 0

    for data in file_data:
        existing_file = get_file_by_path(data[0])
        if existing_file:
            if check_modified_file(data[0], data[2]):
                modified_count += 1
                print(f"Re-indexing modified file: {data[1]}")
                try:
                    remove_file_from_index(existing_file.faiss_index)
                    add_file_to_index(existing_file.faiss_index, data[4])
                    update_file(file_id=existing_file.file_id,
                                file_path=data[0],
                                file_name=data[1],
                                file_hash=data[2],
                                modified_time=data[3])
                except Exception as e:
                    print(f"  ⚠️  Error re-indexing: {e}")
            else:
                unchanged_count += 1
        else:
            # Collect new files for batch processing
            new_files_batch.append((next_faiss_index, data))
            next_faiss_index += 1

    print(f"\nSummary: {unchanged_count} unchanged, {modified_count} modified, {len(new_files_batch)} new files")

    # Process new files in larger batches for better performance
    try:
        if new_files_batch:
            batch_size = 1000  # Increased from 350 to reduce number of batches
            total_batches = (len(new_files_batch) + batch_size - 1) // batch_size
            print(f"\nProcessing {len(new_files_batch)} new files in {total_batches} batch(es)...")

            for i in range(0, len(new_files_batch), batch_size):
                batch = new_files_batch[i:i + batch_size]
                batch_num = i//batch_size + 1
                batch_start = time.time()
                print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} files...")

                # Create database records using bulk insert (much faster)
                db_start = time.time()
                print(f"  Creating database records...")
                try:
                    # Prepare bulk insert data
                    bulk_data = [(data[0], data[1], data[2], data[3], faiss_idx)
                                for faiss_idx, data in batch]
                    count = create_files_bulk(bulk_data)
                    db_time = time.time() - db_start
                    print(f"  ✓ Database: {count} records created ({db_time:.1f}s)")
                except Exception as e:
                    db_time = time.time() - db_start
                    print(f"  ⚠️  Database bulk insert failed ({db_time:.1f}s): {e}")
                    # Fallback to individual inserts if bulk fails
                    print(f"  Falling back to individual inserts...")
                    db_errors = 0
                    for faiss_idx, data in batch:
                        try:
                            create_file(file_path=data[0],
                                    file_name=data[1],
                                    file_hash=data[2],
                                    modified_time=data[3],
                                    faiss_index=faiss_idx)
                        except Exception as e:
                            db_errors += 1
                    if db_errors > 0:
                        print(f"  ⚠️  {db_errors} errors during fallback")

                # Batch encode all files in this batch
                encode_start = time.time()
                try:
                    batch_data = [(faiss_idx, data[4]) for faiss_idx, data in batch]
                    add_files_to_index_batch(batch_data, batch_size=1024)
                    encode_time = time.time() - encode_start
                    print(f"  ✓ Encoding completed ({encode_time:.1f}s)")
                except Exception as e:
                    print(f"  ⚠️  Error during batch encoding: {e}")

                batch_time = time.time() - batch_start
                print(f"  Batch {batch_num} total time: {batch_time:.1f}s")

        # Queue timeout files for background indexing
        if timeout_files:
            print(f"\n⏱️  {len(timeout_files)} files timed out - queuing for background indexing...")
            print(f"   You can start searching now while these files index in the background.")

            # Start background indexer
            start_background_indexer()

            # Add all timeout files to background queue with 60s timeout
            for file_path in timeout_files:
                add_to_background_queue(file_path, timeout_seconds=60)

            print(f"   Background indexer started. Check status with get_background_status()")

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