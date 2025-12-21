import os
import PyPDF2
from docx import Document
import hashlib
import time

def read_file(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == ".pdf":
        return return_pdf_info(file_path)
    elif file_ext == ".docx":
        return return_docx_info(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def return_pdf_info(file_path: str) -> list[str]:
    file_name = os.path.basename(file_path)
    modified_time = os.path.getmtime(file_path)
    modified_time = time.ctime(modified_time)

    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

    hashed_content = hash_file(text)

    return [file_path, file_name, hashed_content, modified_time]

def return_docx_info(file_path: str) -> list[str]:
    file_name = os.path.basename(file_path)
    modified_time = os.path.getmtime(file_path)
    modified_time = time.ctime(modified_time)

    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    hashed_content = hash_file(text)
    
    return [file_path, file_name, hashed_content, modified_time]


def hash_file(text: str) -> str:
    hash = hashlib.new('sha256')
    hash.update(text.encode())
    return hash.hexdigest()