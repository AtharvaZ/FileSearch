#!/usr/bin/env python3
"""
Fix files that are in database but have no indexed chunks

This script finds all files in the DB with 0 chunks and queues them
for background indexing.
"""

import pickle
from database import Base, engine, get_all_files
from background_indexer import add_to_background_queue, start_background_indexer, get_background_status
from vector_search import load_index
import time

def fix_empty_files():
    Base.metadata.create_all(engine)

    print("Loading FAISS index...")
    load_index()

    # Load chunk metadata
    with open('chunk_metadata.pkl', 'rb') as f:
        data = pickle.load(f)
        chunks = data['chunk_metadata']

    # Get file indices that have chunks
    file_indices_with_chunks = set(c['file_faiss_index'] for c in chunks)

    # Find files without chunks
    all_files = get_all_files()
    files_without_chunks = [f for f in all_files if f.faiss_index not in file_indices_with_chunks]

    print(f"\nTotal files in database: {len(all_files)}")
    print(f"Files with indexed content: {len(all_files) - len(files_without_chunks)}")
    print(f"Files WITHOUT indexed content: {len(files_without_chunks)}")

    if not files_without_chunks:
        print("\n✓ All files are properly indexed!")
        return

    print(f"\nSample files without content:")
    for f in files_without_chunks[:10]:
        print(f"  - {f.file_name}")
    if len(files_without_chunks) > 10:
        print(f"  ... and {len(files_without_chunks) - 10} more")

    print(f"\nQueuing {len(files_without_chunks)} files for background indexing...")

    # Start background indexer
    start_background_indexer()

    # Queue all files
    for file in files_without_chunks:
        add_to_background_queue(file.file_path, timeout_seconds=60)

    print(f"\n✓ Background indexer started!")
    print(f"  Files queued: {len(files_without_chunks)}")
    print(f"\nYou can:")
    print(f"  - Start searching immediately (files will index in background)")
    print(f"  - Wait for completion (this script will monitor progress)")
    print()

    choice = input("Wait for background indexing to complete? (y/n): ")

    if choice.lower() == 'y':
        print("\nWaiting for background indexing...")
        print("(This may take a while for large files)")
        print()

        while True:
            status = get_background_status()
            if not status['is_active'] or status['stats']['pending'] == 0:
                break

            pending = status['stats']['pending']
            completed = status['stats']['completed']
            failed = status['stats']['failed']
            current = status['stats']['current_file']

            if current:
                print(f"\r⏱️  Progress: {completed} done, {pending} pending, {failed} failed | Current: {current[:50]}...", end='', flush=True)
            else:
                print(f"\r⏱️  Progress: {completed} done, {pending} pending, {failed} failed", end='', flush=True)

            time.sleep(1)

        final_status = get_background_status()
        print(f"\n\n✓ Background indexing complete!")
        print(f"  Successfully indexed: {final_status['stats']['completed']}")
        print(f"  Failed: {final_status['stats']['failed']}")
    else:
        print("\nBackground indexing will continue in the background.")
        print("You can start the GUI and search while files are being indexed.")

if __name__ == "__main__":
    fix_empty_files()
