import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import os
import pickle
import torch

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use GPU if available for massive speedup (10-50x faster encoding)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

transformer_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)
dimension = 384

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# Product Quantization settings for 8-16x compression
nlist = 100  # Number of clusters (will be adjusted based on data size)
m = 8  # Number of subquantizers (dimension / m must be divisible)
nbits = 8  # Bits per subquantizer (8 = 256 centroids per subquantizer)

# Start with flat index, will upgrade to IVF-PQ after training
quantizer = faiss.IndexFlatL2(dimension)
base_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
index = faiss.IndexIDMap(base_index)
is_trained = False  # Track if index has been trained

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNK_METADATA_PATH = "chunk_metadata.pkl"
TRAINING_DATA_PATH = "training_embeddings.npy"  # Cache for training data

# Store metadata about chunks (which chunks belong to which file)
chunk_metadata = []
training_embeddings = []  # Collect embeddings for training

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of specified size with overlap (max ~125 tokens for model)"""
    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

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

        new_start = end - overlap if end < text_length else text_length

        # Ensure we're making progress (handle cases with no spaces/sentence breaks)
        if new_start <= start and new_start < text_length:
            new_start = start + chunk_size  # Skip forward by chunk_size to avoid infinite loop

        start = new_start

    return chunks


def encode_chunks(chunks: list[str], batch_size: int = 1024) -> np.ndarray:
    """Encode text chunks into embeddings using SentenceTransformer with batching

    Implements automatic batch size reduction on OOM to prevent crashes
    Default batch_size=1024 provides optimal throughput on MPS (921 chunks/sec)
    """
    # Conservative batch size - don't multiply on GPU to avoid OOM
    effective_batch_size = batch_size

    # Try encoding with retry logic for OOM errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            embeddings = transformer_model.encode(
                chunks,
                convert_to_numpy=True,
                batch_size=effective_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better similarity comparison
            )
            return embeddings.astype('float32')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and attempt < max_retries - 1:
                # Reduce batch size and retry
                effective_batch_size = effective_batch_size // 2
                print(f"  ⚠️  GPU OOM detected, reducing batch size to {effective_batch_size} and retrying...")

                # Clear GPU cache if using CUDA or MPS
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()
            else:
                raise  # Re-raise if not OOM or out of retries

def train_index_if_needed():
    """Train the IVF-PQ index using collected training embeddings"""
    global is_trained, training_embeddings, base_index, index, nlist

    if is_trained or len(training_embeddings) == 0:
        return

    # Convert list to numpy array
    train_data = np.vstack(training_embeddings)
    n_vectors = train_data.shape[0]

    # FIXED: Use a fixed target nlist (1000) to prevent infinite loop
    # where the requirement (39*nlist) grows faster than data collection
    target_nlist = 1000
    min_train_vectors = 39 * target_nlist  # = 39,000 vectors

    if n_vectors >= min_train_vectors:
        print(f"\nTraining IVF-PQ index with {n_vectors} vectors (nlist={target_nlist})...")

        # Create index with target nlist
        quantizer = faiss.IndexFlatL2(dimension)
        base_index = faiss.IndexIVFPQ(quantizer, dimension, target_nlist, m, nbits)
        index = faiss.IndexIDMap(base_index)

        base_index.train(train_data)
        is_trained = True
        print(f"✓ Index trained successfully!\n")

        # Clear training data to free memory
        training_embeddings.clear()
    else:
        # Reduce spam - only print every 5000 vectors
        if n_vectors % 5000 < 1024:
            pct = n_vectors * 100 // min_train_vectors
            print(f"  Collecting training data: {n_vectors:,}/{min_train_vectors:,} vectors ({pct}%)")

def add_file_to_index(file_faiss_index: int, text: str):
    global chunk_metadata

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = encode_chunks(chunks, batch_size=1024)
    chunk_ids = np.array([file_faiss_index * 10000 + i for i in range(len(chunks))], dtype=np.int64)
    index.add_with_ids(embeddings, chunk_ids)

    # Store metadata
    for chunk_id, chunk_content in zip(chunk_ids, chunks):
        chunk_metadata.append({
            'faiss_id': int(chunk_id),
            'file_faiss_index': file_faiss_index,
            'chunk_text': chunk_content
        })

    return len(chunks)


def add_encoded_batch_to_index(embeddings: np.ndarray, file_indices: list[int],
                                chunk_numbers: list[int], chunk_texts: list[str]):
    """
    Add pre-encoded batch directly to index (for streaming pipeline)

    Args:
        embeddings: Pre-encoded embeddings array
        file_indices: List of file FAISS indices
        chunk_numbers: List of chunk numbers within each file
        chunk_texts: List of chunk text content

    Returns:
        Number of chunks added
    """
    global chunk_metadata, training_embeddings, is_trained

    if len(embeddings) == 0:
        return 0

    # If not trained, collect embeddings for training
    if not is_trained:
        training_embeddings.append(embeddings)
        train_index_if_needed()

    # Create chunk IDs and metadata
    chunk_ids = []
    for file_idx, chunk_num, chunk_text in zip(file_indices, chunk_numbers, chunk_texts):
        chunk_id = file_idx * 10000 + chunk_num
        chunk_ids.append(chunk_id)

        chunk_metadata.append({
            'faiss_id': int(chunk_id),
            'file_faiss_index': file_idx,
            'chunk_text': chunk_text
        })

    # Add to FAISS if trained
    if is_trained:
        chunk_ids_array = np.array(chunk_ids, dtype=np.int64)
        index.add_with_ids(embeddings, chunk_ids_array)

    return len(embeddings)


def search_similar_with_reranking(query: str, k: int = 5, initial_k: int = 100):
    """
    Two-stage search: retrieve candidates with FAISS, then rerank with cross-encoder

    Retrieves more candidates (100 vs 20) to compensate for PQ compression accuracy loss.
    The cross-encoder reranker then picks the best k results from these candidates.
    """
    # Set nprobe for IVF index (how many clusters to search)
    if is_trained:
        base_index.nprobe = min(50, base_index.nlist)  # Search 50 clusters (or all if less)

    query_embedding = transformer_model.encode([query], convert_to_numpy=True).astype('float32')
    distance, indices = index.search(query_embedding, initial_k)

    candidates = []
    for dist, idx in zip(distance[0], indices[0]):
        if idx == -1:
            continue

        metadata = next((d for d in chunk_metadata if d['faiss_id'] == int(idx)), None)
        if metadata:
            candidates.append({
                "file_faiss_index": metadata['file_faiss_index'],
                "distance": float(dist),
                "chunk_text": metadata['chunk_text'],
                "idx": int(idx)
            })

    if not candidates:
        return []
    query_doc_pairs = [[query, candidate["chunk_text"]] for candidate in candidates]

    #enhance search with reranker
    rerank_scores = reranker_model.predict(query_doc_pairs)
    for candidate, score in zip(candidates, rerank_scores):
        candidate['rerank_score'] = float(score)

    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    return candidates[:k]


def diagnose_search(query: str, k: int = 15):
    """
    Diagnostic function to identify search accuracy issues

    Compares:
    1. PQ-compressed index results vs theoretical flat index
    2. Raw FAISS distances vs reranker scores
    3. File type distribution in results

    Returns detailed breakdown showing which component may be causing issues
    """
    print(f"\n{'='*80}")
    print(f"SEARCH DIAGNOSTICS FOR: '{query}'")
    print(f"{'='*80}\n")

    # Stage 1: FAISS retrieval with PQ compression
    print("[Stage 1] FAISS Retrieval (with PQ compression)")
    print("-" * 80)

    if is_trained:
        base_index.nprobe = min(50, base_index.nlist)
        print(f"Index type: IVF-PQ (nlist={base_index.nlist}, nprobe={base_index.nprobe})")
        print(f"Compression: {m} subquantizers × {nbits} bits = {dimension/(m*nbits/8):.1f}x compression")
    else:
        print("Index type: Flat (uncompressed)")

    query_embedding = transformer_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, 100)  # Get 100 candidates

    # Collect candidates with metadata
    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        metadata = next((d for d in chunk_metadata if d['faiss_id'] == int(idx)), None)
        if metadata:
            candidates.append({
                "file_faiss_index": metadata['file_faiss_index'],
                "faiss_distance": float(dist),
                "chunk_text": metadata['chunk_text'][:200] + "..." if len(metadata['chunk_text']) > 200 else metadata['chunk_text']
            })

    print(f"\nRetrieved {len(candidates)} candidates from FAISS")
    print(f"\nTop 10 by FAISS distance:")
    for i, c in enumerate(candidates[:10], 1):
        print(f"  {i}. Distance: {c['faiss_distance']:.4f}")
        print(f"     Chunk: {c['chunk_text']}")
        print()

    # Stage 2: Reranking
    print(f"\n[Stage 2] Cross-Encoder Reranking")
    print("-" * 80)

    query_doc_pairs = [[query, c["chunk_text"]] for c in candidates]
    rerank_scores = reranker_model.predict(query_doc_pairs)

    for candidate, score in zip(candidates, rerank_scores):
        candidate['rerank_score'] = float(score)

    candidates_reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    print(f"\nTop 10 after reranking:")
    for i, c in enumerate(candidates_reranked[:10], 1):
        rank_change = ""
        original_rank = next((j for j, orig in enumerate(candidates) if orig['file_faiss_index'] == c['file_faiss_index']), -1)
        if original_rank != -1:
            rank_change = f" (was #{original_rank+1}, moved {original_rank - i + 1:+d})"

        print(f"  {i}. Rerank score: {c['rerank_score']:.4f}, FAISS dist: {c['faiss_distance']:.4f}{rank_change}")
        print(f"     Chunk: {c['chunk_text']}")
        print()

    # Stage 3: File-level deduplication
    print(f"\n[Stage 3] File Deduplication (top {k} unique files)")
    print("-" * 80)

    from database import get_files_by_faiss_indices
    file_indices = list(set([c['file_faiss_index'] for c in candidates_reranked]))
    files = get_files_by_faiss_indices(file_indices)

    file_results = []
    for file_record in files:
        file_chunks = [c for c in candidates_reranked if c['file_faiss_index'] == file_record.faiss_index]
        best_match = min(file_chunks, key=lambda x: x['faiss_distance'])

        file_ext = os.path.splitext(file_record.file_path)[1]

        file_results.append({
            'file_name': file_record.file_name,
            'file_path': file_record.file_path,
            'file_ext': file_ext,
            'faiss_distance': best_match['faiss_distance'],
            'rerank_score': best_match['rerank_score'],
            'chunk_text': best_match['chunk_text']
        })

    # Sort by rerank score
    file_results = sorted(file_results, key=lambda x: x['rerank_score'], reverse=True)[:k]

    print(f"\nTop {k} files (deduplicated):")
    for i, f in enumerate(file_results, 1):
        print(f"  {i}. {f['file_name']} ({f['file_ext']})")
        print(f"     Path: {f['file_path']}")
        print(f"     Rerank: {f['rerank_score']:.4f}, FAISS: {f['faiss_distance']:.4f}")
        print(f"     Chunk: {f['chunk_text']}")
        print()

    # Analysis
    print(f"\n[Analysis]")
    print("-" * 80)

    # File type distribution
    ext_counts = {}
    for f in file_results:
        ext = f['file_ext']
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"\nFile type distribution in top {k} results:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or 'no extension'}: {count} files")

    # Score analysis
    faiss_range = max([f['faiss_distance'] for f in file_results]) - min([f['faiss_distance'] for f in file_results])
    rerank_range = max([f['rerank_score'] for f in file_results]) - min([f['rerank_score'] for f in file_results])

    print(f"\nScore ranges:")
    print(f"  FAISS distance: {min([f['faiss_distance'] for f in file_results]):.4f} - {max([f['faiss_distance'] for f in file_results]):.4f} (range: {faiss_range:.4f})")
    print(f"  Rerank score: {min([f['rerank_score'] for f in file_results]):.4f} - {max([f['rerank_score'] for f in file_results]):.4f} (range: {rerank_range:.4f})")

    # Reranking impact
    big_changes = []
    for i, c in enumerate(candidates_reranked[:20]):
        original_rank = next((j for j, orig in enumerate(candidates) if orig['file_faiss_index'] == c['file_faiss_index']), -1)
        if original_rank != -1:
            rank_change = original_rank - i
            if abs(rank_change) > 10:
                big_changes.append((c, original_rank, i, rank_change))

    if big_changes:
        print(f"\nLarge ranking changes by reranker (moved >10 positions):")
        for c, old_rank, new_rank, change in big_changes[:5]:
            print(f"  Moved from #{old_rank+1} to #{new_rank+1} ({change:+d} positions)")
            print(f"    FAISS: {c['faiss_distance']:.4f}, Rerank: {c['rerank_score']:.4f}")
            print(f"    Chunk: {c['chunk_text']}")

    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")

    return file_results


def save_index():
    """Save FAISS index, metadata, and training state to disk"""
    if is_trained:
        faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNK_METADATA_PATH, 'wb') as f:
        pickle.dump({
            'chunk_metadata': chunk_metadata,
            'is_trained': is_trained
        }, f)


def load_index():
    """Load FAISS index, metadata, and training state from disk"""
    global index, chunk_metadata, is_trained, base_index

    if os.path.exists(CHUNK_METADATA_PATH):
        with open(CHUNK_METADATA_PATH, 'rb') as f:
            data = pickle.load(f)
            # Handle both old and new format
            if isinstance(data, dict) and 'chunk_metadata' in data:
                chunk_metadata = data['chunk_metadata']
                is_trained = data.get('is_trained', False)
            else:
                # Old format (just list)
                chunk_metadata = data
                is_trained = False

    if os.path.exists(FAISS_INDEX_PATH) and is_trained:
        index = faiss.read_index(FAISS_INDEX_PATH)
        # Extract base_index from IDMap
        base_index = faiss.downcast_index(index.index)
        print(f"Loaded trained IVF-PQ index with {index.ntotal} vectors")
        return True
    elif not os.path.exists(FAISS_INDEX_PATH):
        print("No existing index found, will create new one")
    return False


def remove_file_from_index(file_faiss_index: int):
   
    global chunk_metadata

    # Find all chunk IDs for this file
    chunk_ids_to_remove = [
        m['faiss_id'] for m in chunk_metadata
        if m['file_faiss_index'] == file_faiss_index
    ]

    if chunk_ids_to_remove:
        # Remove from FAISS index
        index.remove_ids(np.array(chunk_ids_to_remove, dtype=np.int64))

        # Remove from metadata
        chunk_metadata = [
            m for m in chunk_metadata
            if m['file_faiss_index'] != file_faiss_index
        ]

    return len(chunk_ids_to_remove)


def clear_index():
    """Clear the entire FAISS index and metadata"""
    global index, chunk_metadata

    base_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(base_index)
    chunk_metadata = []

    # Remove saved files if they exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(CHUNK_METADATA_PATH):
        os.remove(CHUNK_METADATA_PATH)