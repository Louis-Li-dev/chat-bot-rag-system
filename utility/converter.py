from pypdf import PdfReader
def pdf_to_string(path):
    reader = PdfReader(path)
    files = []
    for page in reader.pages:
        files.append(page.extract_text())
    files = '\n'.join(files)
    return files
