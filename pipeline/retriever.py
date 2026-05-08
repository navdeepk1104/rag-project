from pipeline.embeddings import load_vectorstore
from config.settings import TOP_K_RESULTS

def get_relevant_chunks(question: str) -> list[str]:
    """
    Takes a question,searches the vectorstore,
    returns the most relevant chunks with page number metadata.
    """
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        query=question,
        k=TOP_K_RESULTS
    )

    chunks=[]
    for doc in results:
        chunks.append({
            "content": doc.page_content,
            "page": doc.metadata.get("page_number", "N/A")
        })
    return chunks


if __name__ =="__main__":
    question = "what is this document about?"

    print(f"Searching for: {question}\n")
    chunks=get_relevant_chunks(question)

    print(f"Found {len(chunks)} relevant chunks:\n")
    for i,chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk[:300])
        print()