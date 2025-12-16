import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


search_directory = "/Users/atharvazaveri/"


for root, dirs, files in os.walk(search_directory):
    app_dirs = [d for d in dirs if d.endswith('.app')]
    
    dirs[:] = [d for d in dirs if not d.startswith('.') and not d.endswith('.app')]
    
    has_subdirs = len(dirs) > 0
   
    if not has_subdirs:
        for app_dir in app_dirs:
            print(f"  📱 {app_dir}")
        for file in files:
            if not file.startswith('.'):
                print(f"  📄 {file}")
    else:
        print("=" * 60)
        print(f"FOLDER: {root}")
        print("=" * 60)
    
        for app_dir in app_dirs:
            print(f"  📱 {app_dir}")
        
        if files:
            for file in files:
                if not file.startswith('.'):
                    print(f"  📄 {file}")
        elif not app_dirs:
            print("  (empty folder)")
        
        print()