from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE,CHUNK_OVERLAP

def split_text(text: str) -> list[str]:
    """
    Splits raw text into chunks.
    Returns a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks

def split_documents(pages: list[dict]) -> list[dict]:
    """
    Splits page-wise text into chunks but keeps page metadata.
    Useful for showing sources later.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks_with_metadata = []

    for page in pages:
        chunks = splitter.split_text(page["content"])
        for chunk in chunks:
            chunks_with_metadata.append({
                "page_number": page["page_number"],
                "content": chunk
            })

    return chunks_with_metadata    
    


if __name__ == "__main__":
    from pipeline.loader import load_pdf, load_pdf_with_metadata

    text = load_pdf("sample.pdf")
    chunks = split_text(text)
    print(f"✅ Total chunks created: {len(chunks)}")
    print(f"📄 First chunk preview:\n{chunks[0]}")
    print(f"\n📄 Last chunk preview:\n{chunks[-1]}")

    pages = load_pdf_with_metadata("sample.pdf")
    chunks_with_meta = split_documents(pages)
    print(f"\n✅ Total chunks with metadata: {len(chunks_with_meta)}")
    print(f"📄 Sample chunk: {chunks_with_meta[0]}")