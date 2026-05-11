import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import GEMINI_API_KEY, VECTORSTORE_PATH
from langchain_core.documents import Document


def create_vectorstore(chunks: list[dict]):
    """
    Converts text chunks into embeddings and stores them in FAISS.
    Handles rate limits automatically with exponential backoff.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    docs = [
        Document(
            page_content=chunk["content"],
            metadata={"page_number": chunk["page_number"]}
        )
        for chunk in chunks
    ]

    batch_size = 25  # stay well under 100/min limit
    vectorstore = None
    total_batches = (len(docs) // batch_size) + 1

    print("⏳ Waiting 10 seconds before starting...")
    time.sleep(10)
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"📦 Embedding batch {batch_num}/{total_batches}...")

        retries = 0
        max_retries = 5

        while retries < max_retries:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vectorstore.add_documents(batch)
                break  # success, move to next batch

            except Exception as e:
                retries += 1
                wait_time = 15 * retries  # 15s, 30s, 45s...
                print(f"⏳ Rate limit hit on batch {batch_num}. Retry {retries}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)

                if retries == max_retries:
                    raise Exception(f"❌ Failed after {max_retries} retries on batch {batch_num}") from e

        # small polite delay between every batch
        time.sleep(5)

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ Vectorstore saved to {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore():
    """
    Loads an already saved FAISS index from disk.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Vectorstore loaded from {VECTORSTORE_PATH}")
    return vectorstore