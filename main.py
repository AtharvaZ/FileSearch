import os
from database import *
from file_reader import read_file
from vector_search import load_index, save_index, remove_file_from_index, add_file_to_index, add_files_to_index_batch, search_similar



search_directory = "/Users/atharvazaveri/"
file_paths = list()

def get_file_content(path: str) -> list[str]:
    all_paths_content = []
    print(f"Scanning directory: {path}")

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common system/app directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['Library', 'Applications', 'System']]

        for file in files:
            if file.endswith(('.pdf', '.docx')):
                all_paths_content.append(os.path.abspath(os.path.join(root, file)))

    print(f"Found {len(all_paths_content)} PDF/DOCX files")
    return all_paths_content
    
def check_modified_file(file_path: str, modified_time: str) -> bool:
    existing_file = get_file_by_path(file_path)
    if existing_file:
        existing_modified_time = existing_file.modified_time.strftime("%a %b %d %H:%M:%S %Y")
        return existing_modified_time != modified_time
    return True

def index_files():
    print("Loading existing FAISS index...")
    load_index()

    # Get all files from directory
    file_contents = get_file_content(search_directory)
    file_data = []
    failed_files = []

    print(f"\nReading file contents...")
    for i, content in enumerate(file_contents, 1):
        try:
            if i % 50 == 0:
                print(f"  Processed {i}/{len(file_contents)} files...")
            file_data.append(read_file(content))
        except PermissionError:
            failed_files.append((content, "Permission denied"))
        except Exception as e:
            failed_files.append((content, str(e)))

    if failed_files:
        print(f"\n⚠️  Failed to read {len(failed_files)} files:")
        for path, error in failed_files[:5]:  # Show first 5 errors
            print(f"  - {os.path.basename(path)}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    print(f"\nSuccessfully read {len(file_data)} files")

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
            if check_modified_file(data[0], data[3]):
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

    # Process new files in batches of 350
    if new_files_batch:
        batch_size = 350
        total_batches = (len(new_files_batch) + batch_size - 1) // batch_size
        print(f"\nProcessing {len(new_files_batch)} new files in {total_batches} batch(es)...")

        for i in range(0, len(new_files_batch), batch_size):
            batch = new_files_batch[i:i + batch_size]
            batch_num = i//batch_size + 1
            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} files...")

            # Create database records first
            print(f"  Creating database records...")
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
                    print(f"    ⚠️  DB error for {data[1]}: {e}")

            if db_errors == 0:
                print(f"  ✓ Database records created")
            else:
                print(f"  ⚠️  Database records created with {db_errors} errors")

            # Batch encode all files in this batch
            try:
                batch_data = [(faiss_idx, data[4]) for faiss_idx, data in batch]
                add_files_to_index_batch(batch_data, batch_size=512)
            except Exception as e:
                print(f"  ⚠️  Error during batch encoding: {e}")

    print("\nSaving FAISS index...")
    save_index()
    print("FAISS index saved!")

def search_files(query: str, k: int = 5):
    """Search files using vector semantic search"""
    load_index()
    search_results = search_similar(query, k=k)

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
            'chunk_text': best_match['chunk_text']
        })

    return results

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