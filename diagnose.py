#!/usr/bin/env python3
"""
Diagnostic tool to identify search accuracy issues

Usage:
    python diagnose.py "your search query"

This will show you detailed information about each stage of the search pipeline:
1. FAISS retrieval with PQ compression
2. Cross-encoder reranking
3. File deduplication

It helps identify whether issues come from:
- PQ compression losing relevant vectors
- Embedding model creating poor semantic representations
- Reranker failing to correct initial retrieval errors
"""

import sys
from database import Base, engine
from vector_search import load_index, diagnose_search

if __name__ == "__main__":
    # Initialize database
    Base.metadata.create_all(engine)

    # Load index
    print("Loading FAISS index...")
    load_index()

    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "semester grades"
        print(f"No query provided, using default: '{query}'")

    # Run diagnostic
    results = diagnose_search(query, k=15)

    print("\nDiagnostic complete! Review the output above to identify issues.")
    print("\nWhat to look for:")
    print("  - If code files appear in Stage 1 (FAISS): The embedding model is semantically confused")
    print("  - If code files appear only after Stage 2 (Reranking): The reranker is boosting wrong results")
    print("  - If FAISS distances are similar but content is irrelevant: PQ compression may be losing detail")
    print("  - Check 'Large ranking changes' to see if reranker is helping or hurting")
