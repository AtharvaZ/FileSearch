import os
from database import *
from file_reader import read_file
from file_search import search_by_content

search_directory = "/Users/atharvazaveri/Desktop/fileSearch_test"
file_paths = list()

def get_file_content(path: str) -> list[str]:
    all_paths_content = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.pdf', '.docx')):
                all_paths_content.append(os.path.abspath(os.path.join(root, file)))
    return all_paths_content
    
def check_modified_file(file_path: str, modified_time: str) -> bool:
    existing_file = get_file_by_path(file_path)
    if existing_file:
        existing_modified_time = existing_file.modified_time.strftime("%a %b %d %H:%M:%S %Y")
        return existing_modified_time != modified_time
    return True

def index_files():
    # Get all files from directory
    file_contents = get_file_content(search_directory)
    file_data = []
    for content in file_contents:
         file_data.append(read_file(content))

    db_files = get_all_files()

    # Remove files from database that no longer exist on disk
    for db_file in db_files:
        if not os.path.exists(db_file.file_path):
            delete_file_by_path(db_file.file_path)
            print(f"Removed deleted file from database: {db_file.file_name}")

    # Add or update files
    for data in file_data:
        existing_file = get_file_by_path(data[0])
        if existing_file:
            if check_modified_file(data[0], data[3]):
                update_file(file_id=existing_file.file_id,
                            file_path=data[0],
                            file_name=data[1],
                            file_hash=data[2],
                            modified_time=data[3])
        else:
            create_file(file_path=data[0],
                    file_name=data[1],
                    file_hash=data[2],
                    modified_time=data[3])

def search_files(query: str, case_sensitive: bool = False):
    results = search_by_content(query, case_sensitive)

    if not results:
        print(f"No results found for '{query}'")
    else:
        print(f"Found {len(results)} result(s)\n")
        for result in results:
            print(f"File: {result['file_name']}")
            print("-" * 50)

    return results

if __name__ == "__main__":
    import sys

    Base.metadata.create_all(engine)

    # Check if GUI mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        import tkinter as tk
        from app import FileSearchApp
        root = tk.Tk()
        app = FileSearchApp(root)
        root.mainloop()
    else:
        # CLI mode
        print("Indexing files...")
        index_files()
        print("Indexing complete!\n")

        # Example search (you can comment this out or modify)
        search_files("module", case_sensitive=False)