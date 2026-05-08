import pytesseract
from pdf2image import convert_from_path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_with_ocr(file_path: str) -> list[dict]:
    """
    Converts each PDF page to an image and runs OCR on it.
    Returns list of dicts with page_number and content.
    Used as fallback when normal PDF text extraction fails.
    """
    print("Running OCR on scanned PDF...")
    pages = convert_from_path(file_path, dpi=300)

    results = []
    for page_num, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image)
        if text.strip():
            results.append({
                "page_number": page_num + 1,
                "content": text
            })
        else:
            print(f"No text found on page {page_num + 1} even with OCR")

    return results


def is_scanned_pdf(pages: list[dict]) -> bool:
    """
    Checks if a PDF is scanned by seeing if normal extraction
    returned very little text.
    """
    total_text = " ".join([p["content"] for p in pages])
    return len(total_text.strip()) < 100
