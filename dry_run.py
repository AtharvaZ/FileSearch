#!/usr/bin/env python3
"""
Dry run script to preview how many files will be indexed
without actually indexing them.
"""
import os

def count_files(path: str):
    """Count PDF and DOCX files in directory"""
    pdf_files = []
    docx_files = []

    print(f"Scanning: {path}\n")

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common system/app directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['Library', 'Applications', 'System']]

        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            if file.endswith('.pdf'):
                pdf_files.append(full_path)
            elif file.endswith('.docx'):
                docx_files.append(full_path)

    return pdf_files, docx_files

if __name__ == "__main__":
    # Change this to the directory you want to scan
    search_dir = "/Users/atharvazaveri"

    print("=" * 60)
    print("DRY RUN - File Indexing Preview")
    print("=" * 60)

    pdf_files, docx_files = count_files(search_dir)

    print(f"\nResults:")
    print(f"  PDF files:  {len(pdf_files)}")
    print(f"  DOCX files: {len(docx_files)}")
    print(f"  TOTAL:      {len(pdf_files) + len(docx_files)}")

    # Estimate disk space (rough approximation)
    total_files = len(pdf_files) + len(docx_files)
    # Assume average 30 chunks per file, 384 dimensions, 4 bytes per float
    estimated_index_mb = (total_files * 30 * 384 * 4) / (1024 * 1024)
    estimated_metadata_mb = estimated_index_mb * 0.3  # Metadata is smaller

    print(f"\nEstimated disk space:")
    print(f"  FAISS index: ~{estimated_index_mb:.1f} MB")
    print(f"  Metadata:    ~{estimated_metadata_mb:.1f} MB")
    print(f"  TOTAL:       ~{estimated_index_mb + estimated_metadata_mb:.1f} MB")

    print(f"\nSample files (first 10):")
    for i, path in enumerate(list(pdf_files + docx_files)[:10], 1):
        print(f"  {i}. {path}")

    if total_files > 10:
        print(f"  ... and {total_files - 10} more files")

    print("\n" + "=" * 60)
