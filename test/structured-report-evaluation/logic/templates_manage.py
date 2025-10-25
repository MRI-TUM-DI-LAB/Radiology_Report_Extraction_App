import os
from pathlib import Path
from shared.chroma_interface import upload_files, get_all_reports

class UploadedFile:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.size = path.stat().st_size

    def read(self):
        return self.path.read_bytes()

if __name__ == "__main__":
    # template_dir = Path("data/templates")  

    '''if not template_dir.exists():
        raise FileNotFoundError(f"Template folder '{template_dir}' not found.")

    supported_extensions = {".json", ".html", ".htm"}
    files_to_upload = [
        UploadedFile(path)
        for path in template_dir.iterdir()
        if path.suffix.lower() in supported_extensions
    ]

    print(f"\n📤 Uploading {len(files_to_upload)} files to ChromaDB...")
    upload_files(files_to_upload)
    print("✅ Upload completed.\n")
    '''

    print("Current templates in ChromaDB:")
    reports = get_all_reports()

    for idx, (doc, meta) in enumerate(zip(reports["documents"], reports["metadatas"])):
        print(f"{idx+1:3d}. Filename: {meta.get('filename')}, Category: {meta.get('category')}, Language: {meta.get('language')}")
