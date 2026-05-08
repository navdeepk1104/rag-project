from PyPDF2 import PdfReader
from utils.helpers import extract_text_with_ocr, is_scanned_pdf

def load_pdf_with_metadata(file_path: str) -> list[dict]:
    """
    Same as above but also keeps track of which page each text came from.
    Useful later for showing sources in answers.
    """
    reader = PdfReader(file_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page_number": page_num + 1,
                "content": text
            })

# 🔄 Auto fallback to OCR if text extraction failed
    if is_scanned_pdf(pages):
        print(" Scanned PDF detected - switching to OCR...")        
        pages = extract_text_with_ocr(file_path)

    return pages        


def load_pdf(file_path: str) -> str:
    """
    Reads a PDF file and extracts all text from it.
    Returns a single string with all the text.
    """
    pages = load_pdf_with_metadata(file_path)
    return " ".join([p["content"] for p in pages])


def get_total_pages(file_path: str) -> int:
    """Returns total number of pages in the PDF"""
    reader = PdfReader(file_path)
    return len(reader.pages)

def get_first_page_text(file_path: str) -> str:
    """Returns text from first page - useful for title extraction"""
    reader = PdfReader(file_path)
    if len(reader.pages) > 0:
        return reader.pages[0].extract_text() or ""
    return ""