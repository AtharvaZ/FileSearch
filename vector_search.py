import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import os
import pickle

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

base_index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(base_index)

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNK_METADATA_PATH = "chunk_metadata.pkl"

# Store metadata about chunks (which chunks belong to which file)
chunk_metadata = []

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into chunks of specified size with overlap"""
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


def encode_chunks(chunks: list[str], batch_size: int = 512) -> np.ndarray:
    """Encode text chunks into embeddings using SentenceTransformer with batching"""
    embeddings = transformer_model.encode(chunks, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    return embeddings.astype('float32')

def add_file_to_index(file_faiss_index: int, text: str):
    global chunk_metadata

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = encode_chunks(chunks, batch_size=512)
    chunk_ids = np.array([file_faiss_index * 5000 + i for i in range(len(chunks))], dtype=np.int64)
    index.add_with_ids(embeddings, chunk_ids)

    # Store metadata
    for chunk_id, chunk_content in zip(chunk_ids, chunks):
        chunk_metadata.append({
            'faiss_id': int(chunk_id),
            'file_faiss_index': file_faiss_index,
            'chunk_text': chunk_content
        })

    return len(chunks)


def add_files_to_index_batch(files_data: list[tuple[int, str]], batch_size: int = 512):
    """
    Add multiple files to index with batched encoding across all files

    Args:
        files_data: List of tuples (file_faiss_index, text)
        batch_size: Batch size for encoding (default 512)

    Returns:
        Total number of chunks added
    """
    global chunk_metadata

    # Collect all chunks from all files with their metadata
    all_chunks = []
    chunk_info = []

    for file_faiss_index, text in files_data:
        chunks = chunk_text(text)
        if not chunks:
            continue

        for chunk_num, chunk_content in enumerate(chunks):
            all_chunks.append(chunk_content)
            chunk_info.append((file_faiss_index, chunk_num, chunk_content))

    if not all_chunks:
        return 0

    # Encode all chunks at once with batching
    print(f"Encoding {len(all_chunks)} chunks from {len(files_data)} files...")
    all_embeddings = encode_chunks(all_chunks, batch_size=batch_size)

    # Add to FAISS and metadata
    chunk_ids = []
    for file_faiss_index, chunk_num, chunk_content in chunk_info:
        chunk_id = file_faiss_index * 10000 + chunk_num
        chunk_ids.append(chunk_id)

        chunk_metadata.append({
            'faiss_id': int(chunk_id),
            'file_faiss_index': file_faiss_index,
            'chunk_text': chunk_content
        })

    # Add all embeddings to FAISS at once
    chunk_ids_array = np.array(chunk_ids, dtype=np.int64)
    index.add_with_ids(all_embeddings, chunk_ids_array)

    print(f"Added {len(all_chunks)} chunks to FAISS index")
    return len(all_chunks)


def search_similar_with_reranking(query: str, k: int = 5, initial_k: int = 20):
    """
    Two-stage search: retrieve candidates with FAISS, then rerank with cross-encoder
    """
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


def save_index():
    """Save FAISS index and metadata to disk"""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNK_METADATA_PATH, 'wb') as f:
        pickle.dump(chunk_metadata, f)


def load_index():
    """Load FAISS index and metadata from disk"""
    global index, chunk_metadata

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNK_METADATA_PATH, 'rb') as f:
            chunk_metadata = pickle.load(f)
        return True
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