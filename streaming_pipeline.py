

import threading
import queue
from typing import Iterator
from dataclasses import dataclass
import numpy as np
import os
from file_reader import read_file, hash_file
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


@dataclass
class ChunkBatch:
    """A batch of chunks ready for encoding"""
    chunks: list[str]
    file_indices: list[int]
    chunk_numbers: list[int]


@dataclass
class EncodedBatch:
    """A batch of encoded chunks ready for FAISS insertion"""
    embeddings: np.ndarray
    file_indices: list[int]
    chunk_numbers: list[int]
    chunk_texts: list[str]


class StreamingFileReader:
    """
    Producer: Reads files and yields chunks in micro-batches

    Runs in separate thread to overlap file I/O with encoding
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, micro_batch_size: int = 1024):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.micro_batch_size = micro_batch_size

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks (same logic as vector_search.py)"""
        if not text or len(text) == 0:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end < text_length:
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            new_start = end - self.overlap if end < text_length else text_length

            if new_start <= start and new_start < text_length:
                new_start = start + self.chunk_size

            start = new_start

        return chunks

    def read_file_with_timeout(self, file_path: str, timeout_seconds: int):
        """
        Read a single file with timeout using ThreadPoolExecutor

        Returns:
            tuple: (success, result_or_error)
                - If success: (True, file_data)
                - If timeout: (False, "timeout")
                - If error: (False, error_message)
        """
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(read_file, file_path)

        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
            return (True, result)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            return (False, "timeout")
        except Exception as e:
            executor.shutdown(wait=False)
            return (False, str(e))

    def read_files_streaming(
        self,
        files_data: list[tuple[int, str]],  # (faiss_index, file_path)
        output_queue: queue.Queue,
        timeout_seconds: int = 5
    ):
        """
        Read files and produce chunk batches

        Args:
            files_data: List of (faiss_index, file_path) tuples
            output_queue: Queue to put ChunkBatch objects
            timeout_seconds: Timeout for file reading
        """
        batch_chunks = []
        batch_file_indices = []
        batch_chunk_numbers = []

        failed_files = []
        timeout_files = []

        for faiss_index, file_path in files_data:
            success, result = self.read_file_with_timeout(file_path, timeout_seconds)

            if not success:
                if result == "timeout":
                    timeout_files.append(file_path)
                else:
                    failed_files.append((file_path, result))
                continue

            if not result:
                continue

            try:
                _, _, _, _, text = result
                chunks = self.chunk_text(text)

                if not chunks:
                    continue

                # Add chunks to current batch
                for chunk_num, chunk_text in enumerate(chunks):
                    batch_chunks.append(chunk_text)
                    batch_file_indices.append(faiss_index)
                    batch_chunk_numbers.append(chunk_num)

                    # Send micro-batch when it reaches target size
                    if len(batch_chunks) >= self.micro_batch_size:
                        output_queue.put(ChunkBatch(
                            chunks=batch_chunks,
                            file_indices=batch_file_indices,
                            chunk_numbers=batch_chunk_numbers
                        ))
                        batch_chunks = []
                        batch_file_indices = []
                        batch_chunk_numbers = []

            except Exception as e:
                failed_files.append((file_path, str(e)))
                continue

        # Send remaining chunks as final batch
        if batch_chunks:
            output_queue.put(ChunkBatch(
                chunks=batch_chunks,
                file_indices=batch_file_indices,
                chunk_numbers=batch_chunk_numbers
            ))

        # Send sentinel to signal completion
        output_queue.put(None)

        return failed_files, timeout_files


class StreamingEncoder:
    """
    Consumer: Encodes chunk batches using GPU

    Runs in separate thread to overlap encoding with file reading
    """

    def __init__(self, encode_function):
        """
        Args:
            encode_function: Function that takes (chunks, batch_size) and returns embeddings
        """
        self.encode_function = encode_function

    def encode_batches(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        batch_size: int = 1024
    ):
        """
        Consume ChunkBatch objects and produce EncodedBatch objects

        Args:
            input_queue: Queue of ChunkBatch objects
            output_queue: Queue to put EncodedBatch objects
            batch_size: Batch size for encoding
        """
        while True:
            batch = input_queue.get()

            # Sentinel value signals completion
            if batch is None:
                output_queue.put(None)
                break

            # Encode the batch
            embeddings = self.encode_function(batch.chunks, batch_size=batch_size)

            # Send encoded batch
            output_queue.put(EncodedBatch(
                embeddings=embeddings,
                file_indices=batch.file_indices,
                chunk_numbers=batch.chunk_numbers,
                chunk_texts=batch.chunks
            ))

            input_queue.task_done()


class StreamingPipeline:
    """
    Orchestrates the streaming pipeline for parallel file indexing

    Pipeline stages:
    1. File reader thread: Reads files, chunks text, produces ChunkBatch objects
    2. Encoder thread: Encodes ChunkBatch objects, produces EncodedBatch objects
    3. Main thread: Adds EncodedBatch objects to FAISS and database
    """

    def __init__(
        self,
        encode_function,
        chunk_size: int = 500,
        overlap: int = 50,
        micro_batch_size: int = 1024,
        queue_maxsize: int = 4
    ):
        """
        Args:
            encode_function: Function that encodes chunks
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            micro_batch_size: Number of chunks per micro-batch
            queue_maxsize: Maximum queue size (bounds memory usage)
        """
        self.reader = StreamingFileReader(chunk_size, overlap, micro_batch_size)
        self.encoder = StreamingEncoder(encode_function)

        # Queues with bounded size to prevent memory explosion
        self.chunk_queue = queue.Queue(maxsize=queue_maxsize)
        self.encoded_queue = queue.Queue(maxsize=queue_maxsize)

        self.micro_batch_size = micro_batch_size
        self.failed_files = []
        self.timeout_files = []

    def process_files(
        self,
        files_data: list[tuple[int, str]],
        timeout_seconds: int = 5,
        encoding_batch_size: int = 1024
    ) -> Iterator[EncodedBatch]:
        """
        Process files through the streaming pipeline

        Args:
            files_data: List of (faiss_index, file_path) tuples
            timeout_seconds: Timeout for file reading
            encoding_batch_size: Batch size for encoding

        Yields:
            EncodedBatch objects ready for FAISS insertion
        """
        # Wrapper to capture failed/timeout files
        def reader_wrapper():
            failed, timeout = self.reader.read_files_streaming(
                files_data, self.chunk_queue, timeout_seconds
            )
            self.failed_files = failed
            self.timeout_files = timeout

        # Start reader thread
        reader_thread = threading.Thread(
            target=reader_wrapper,
            daemon=True
        )

        # Start encoder thread
        encoder_thread = threading.Thread(
            target=self.encoder.encode_batches,
            args=(self.chunk_queue, self.encoded_queue, encoding_batch_size),
            daemon=True
        )

        reader_thread.start()
        encoder_thread.start()

        # Main thread consumes encoded batches
        while True:
            encoded_batch = self.encoded_queue.get()

            # Sentinel signals completion
            if encoded_batch is None:
                break

            yield encoded_batch

        # Wait for threads to finish
        reader_thread.join()
        encoder_thread.join()


class StreamingIndexWriter:
    """
    Handles writing encoded batches to FAISS and database with incremental update logic
    """

    def __init__(self):
        self.new_count = 0
        self.modified_count = 0
        self.unchanged_count = 0
        self.file_hash_cache = {}  # Cache hashes from encoded batches

    def write_batch(self, encoded_batch: EncodedBatch, new_files: list,
                    modified_files: list, all_files: list) -> dict:
        """
        Write an encoded batch to FAISS and database

        Args:
            encoded_batch: Batch of encoded chunks
            new_files: List of new file tuples
            modified_files: List of modified file tuples
            all_files: Combined list of all files being processed

        Returns:
            dict with 'chunks_added' and 'files_processed'
        """
        from vector_search import add_encoded_batch_to_index, remove_file_from_index
        from database import create_file, update_file, get_file_by_path
        import time

        # Group chunks by file
        file_chunks = {}
        for i, file_idx in enumerate(encoded_batch.file_indices):
            if file_idx not in file_chunks:
                file_chunks[file_idx] = {
                    'embeddings_indices': [],
                    'chunk_numbers': [],
                    'chunk_texts': [],
                    'text': ''
                }
            file_chunks[file_idx]['embeddings_indices'].append(i)
            file_chunks[file_idx]['chunk_numbers'].append(encoded_batch.chunk_numbers[i])
            file_chunks[file_idx]['chunk_texts'].append(encoded_batch.chunk_texts[i])
            file_chunks[file_idx]['text'] += encoded_batch.chunk_texts[i] + ' '

        chunks_added = 0
        files_processed = 0

        # Process each file in the batch
        for file_idx, file_data in file_chunks.items():
            # Find file info
            file_info = None
            for item in all_files:
                if item[0] == file_idx:
                    file_info = item
                    break

            if not file_info:
                continue

            is_new = len(file_info) == 2
            file_path = file_info[1]

            # Extract embeddings for this file
            indices = file_data['embeddings_indices']
            file_embeddings = encoded_batch.embeddings[indices]

            # Compute hash from text
            combined_text = file_data['text']
            file_hash = hash_file(combined_text)

            # Get file metadata
            file_name = os.path.basename(file_path)
            modified_time = time.ctime(os.path.getmtime(file_path))

            if is_new:
                # New file - add to database and index
                try:
                    create_file(
                        file_path=file_path,
                        file_name=file_name,
                        file_hash=file_hash,
                        modified_time=modified_time,
                        faiss_index=file_idx
                    )

                    add_encoded_batch_to_index(
                        file_embeddings,
                        [file_idx] * len(file_embeddings),
                        file_data['chunk_numbers'],
                        file_data['chunk_texts']
                    )

                    chunks_added += len(file_embeddings)
                    files_processed += 1
                    self.new_count += 1
                except Exception:
                    pass
            else:
                # Existing file - check if modified
                existing_file = file_info[2]

                if existing_file.file_hash != file_hash:
                    # File modified - re-index
                    try:
                        remove_file_from_index(file_idx)

                        add_encoded_batch_to_index(
                            file_embeddings,
                            [file_idx] * len(file_embeddings),
                            file_data['chunk_numbers'],
                            file_data['chunk_texts']
                        )

                        update_file(
                            file_id=existing_file.file_id,
                            file_path=file_path,
                            file_name=file_name,
                            file_hash=file_hash,
                            modified_time=modified_time
                        )

                        chunks_added += len(file_embeddings)
                        files_processed += 1
                        self.modified_count += 1
                    except Exception:
                        pass
                else:
                    # Unchanged
                    self.unchanged_count += 1

        return {
            'chunks_added': chunks_added,
            'files_processed': files_processed
        }
