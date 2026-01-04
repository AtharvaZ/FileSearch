#!/usr/bin/env python3
"""
Test script for streaming pipeline

Creates a few test files and verifies the streaming pipeline works correctly
"""

import os
import tempfile
import shutil
from streaming_pipeline import StreamingPipeline
from vector_search import encode_chunks, load_index, save_index
from database import Base, engine, create_file, get_all_files
import time


def create_test_files(test_dir, num_files=10):
    """Create test text files"""
    files = []
    for i in range(num_files):
        file_path = os.path.join(test_dir, f"test_file_{i}.txt")
        with open(file_path, 'w') as f:
            # Write varying amounts of text
            content = f"This is test file number {i}. " * (100 * (i + 1))
            content += f"\n\nThis file contains information about topic {i}."
            f.write(content)
        files.append(file_path)
    return files


def test_streaming_pipeline():
    """Test the streaming pipeline"""
    print("=" * 80)
    print("STREAMING PIPELINE TEST")
    print("=" * 80)

    # Create temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix="filesearch_test_")
    print(f"\nCreated test directory: {test_dir}")

    try:
        # Create test files
        print("\n1. Creating test files...")
        test_files = create_test_files(test_dir, num_files=10)
        print(f"   Created {len(test_files)} test files")

        # Initialize database
        print("\n2. Initializing database...")
        Base.metadata.create_all(engine)

        # Load FAISS index
        print("\n3. Loading FAISS index...")
        load_index()

        # Prepare data for streaming (faiss_index, file_path)
        streaming_data = [(i + 1000, path) for i, path in enumerate(test_files)]

        # Create pipeline
        print("\n4. Creating streaming pipeline...")
        pipeline = StreamingPipeline(
            encode_function=encode_chunks,
            chunk_size=500,
            overlap=50,
            micro_batch_size=1024,
            queue_maxsize=4
        )

        # Process files
        print("\n5. Processing files through pipeline...")
        total_chunks = 0
        batch_count = 0
        start_time = time.time()

        for encoded_batch in pipeline.process_files(streaming_data, timeout_seconds=5):
            batch_count += 1
            chunks_in_batch = len(encoded_batch.embeddings)
            total_chunks += chunks_in_batch

            print(f"   Batch {batch_count}: {chunks_in_batch} chunks")

            # Verify batch structure
            assert len(encoded_batch.file_indices) == chunks_in_batch
            assert len(encoded_batch.chunk_numbers) == chunks_in_batch
            assert len(encoded_batch.chunk_texts) == chunks_in_batch
            assert encoded_batch.embeddings.shape[0] == chunks_in_batch
            assert encoded_batch.embeddings.shape[1] == 384  # Embedding dimension

        elapsed = time.time() - start_time
        rate = total_chunks / elapsed if elapsed > 0 else 0

        print(f"\n6. Results:")
        print(f"   Total batches: {batch_count}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Rate: {rate:.0f} chunks/s")
        print(f"   Timeout files: {len(pipeline.timeout_files)}")
        print(f"   Failed files: {len(pipeline.failed_files)}")

        # Show failures if any
        if pipeline.failed_files:
            print(f"\n   Failed files details:")
            for path, error in pipeline.failed_files:
                print(f"     - {os.path.basename(path)}: {error}")

        # Verify no timeouts or failures
        assert len(pipeline.timeout_files) == 0, "No files should timeout"
        assert len(pipeline.failed_files) == 0, f"No files should fail, but {len(pipeline.failed_files)} did"
        assert total_chunks > 0, "Should have processed some chunks"
        assert batch_count > 0, "Should have produced at least one batch"

        print(f"\n7. Verification:")
        print(f"   ✓ All {len(test_files)} files processed")
        print(f"   ✓ No timeouts or failures")
        print(f"   ✓ Batch structure validated")
        print(f"   ✓ Embedding dimensions correct (384)")

        print("\n" + "=" * 80)
        print("TEST PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\nCleaning up test directory...")
        shutil.rmtree(test_dir)

    return True


if __name__ == "__main__":
    success = test_streaming_pipeline()
    exit(0 if success else 1)
