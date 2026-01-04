"""
Background indexer for files that timeout during initial indexing

This module handles asynchronous indexing of slow files so users can
search immediately while problematic files continue indexing in background.
"""

import threading
import queue
import time
from database import get_file_by_path, create_file, update_file, get_all_files
from file_reader import read_file
from vector_search import load_index, save_index, add_file_to_index, remove_file_from_index
import signal
from exceptions import TimeoutException, timeout_handler

# Global queue for background indexing tasks
background_queue = queue.Queue()
background_thread = None
is_indexing = False
indexing_stats = {
    'pending': 0,
    'completed': 0,
    'failed': 0,
    'current_file': None
}

def background_indexer_worker():
    """Worker thread that processes background indexing tasks"""
    global is_indexing, indexing_stats

    while True:
        try:
            # Get task from queue (blocks until available)
            task = background_queue.get(timeout=1.0)

            if task is None:  # Poison pill to stop thread
                break

            file_path, timeout_seconds = task
            indexing_stats['current_file'] = file_path

            try:
                # Set up timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

                # Read file
                result = read_file(file_path)
                signal.alarm(0)  # Cancel alarm

                if not result:
                    indexing_stats['failed'] += 1
                    indexing_stats['pending'] -= 1
                    background_queue.task_done()
                    continue

                file_path_result, file_name, file_hash, modified_time, content = result

                # Check if file exists in DB
                existing_file = get_file_by_path(file_path_result)

                if existing_file:
                    # Update existing file
                    remove_file_from_index(existing_file.faiss_index)
                    chunks_added = add_file_to_index(existing_file.faiss_index, content)
                    update_file(
                        file_id=existing_file.file_id,
                        file_path=file_path_result,
                        file_name=file_name,
                        file_hash=file_hash,
                        modified_time=modified_time
                    )
                else:
                    # Add new file
                    faiss_idx = max([f.faiss_index for f in get_all_files()] + [0]) + 1
                    create_file(
                        file_path=file_path_result,
                        file_name=file_name,
                        file_hash=file_hash,
                        modified_time=modified_time,
                        faiss_index=faiss_idx
                    )
                    chunks_added = add_file_to_index(faiss_idx, content)

                # Save index after each file
                save_index()

                indexing_stats['completed'] += 1
                indexing_stats['pending'] -= 1

            except TimeoutException:
                signal.alarm(0)
                indexing_stats['failed'] += 1
                indexing_stats['pending'] -= 1
            except Exception as e:
                signal.alarm(0)
                indexing_stats['failed'] += 1
                indexing_stats['pending'] -= 1

            indexing_stats['current_file'] = None
            background_queue.task_done()

        except queue.Empty:
            # No tasks available, continue waiting
            continue
        except Exception as e:
            # Log error but keep thread alive
            print(f"Background indexer error: {e}")
            continue

def start_background_indexer():
    """Start the background indexing thread"""
    global background_thread, is_indexing

    if background_thread is not None and background_thread.is_alive():
        return  # Already running

    is_indexing = True
    background_thread = threading.Thread(target=background_indexer_worker, daemon=True)
    background_thread.start()

def add_to_background_queue(file_path: str, timeout_seconds: int = 30):
    """Add a file to the background indexing queue"""
    global indexing_stats

    # Start thread if not running
    start_background_indexer()

    # Add task to queue
    background_queue.put((file_path, timeout_seconds))
    indexing_stats['pending'] += 1

def stop_background_indexer():
    """Stop the background indexing thread"""
    global background_thread, is_indexing

    if background_thread is None or not background_thread.is_alive():
        return

    # Send poison pill
    background_queue.put(None)
    background_thread.join(timeout=5.0)
    is_indexing = False

def get_background_status():
    """Get current status of background indexing"""
    return {
        'is_active': background_thread is not None and background_thread.is_alive(),
        'queue_size': background_queue.qsize(),
        'stats': indexing_stats.copy()
    }

def wait_for_background_indexing(timeout: float = None):
    """Wait for all background indexing tasks to complete"""
    start_time = time.time()

    while True:
        if background_queue.empty() and indexing_stats['current_file'] is None:
            break

        if timeout and (time.time() - start_time) > timeout:
            return False

        time.sleep(0.1)

    return True
