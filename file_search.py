from database import get_all_files
import os

def search_by_content(query: str, case_sensitive: bool = False) -> list[dict]:

    results = []
    all_files = get_all_files()

    for file_record in all_files:
        try:
            file_path = file_record.file_path
            file_ext = os.path.splitext(file_path)[1].lower()

            # Read and extract text based on file type
            if file_ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif file_ext == ".docx":
                text = extract_text_from_docx(file_path)
            else:
                continue

            # Perform search
            if case_sensitive:
                found = query in text
            else:
                found = query.lower() in text.lower()

            if found:
                results.append({
                    'file_path': file_record.file_path,
                    'file_name': file_record.file_name,
                    'file_hash': file_record.file_hash,
                    'modified_time': file_record.modified_time
                })

        except Exception as e:
            print(f"Error reading {file_record.file_path}: {e}")
            continue

    return results


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from PDF file"""
    import PyPDF2
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from DOCX file"""
    from docx import Document
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text
