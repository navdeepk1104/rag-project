from dotenv import load_dotenv
import os

# explicitly point to .env file location
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast and very capable

MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
MAX_TOKENS = 1000
TEMPERATURE = 0.2   #keeps answers factual, not creative. Perfect for Q&A

CHUNK_SIZE = 3000    #too large = noisy answers, too small = loses context.
CHUNK_OVERLAP = 400   #ensures a sentence cut at the edge of a chunk isn't lost

VECTORSTORE_PATH = "vectorstore/faiss_index"

TOP_K_RESULTS = 8  #fetches top 4 most relevant chunks for every question