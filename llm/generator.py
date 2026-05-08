from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE

client = Groq(api_key=GROQ_API_KEY)

def genrate_answer(question: str, chunks: list[str], chat_history: list[dict] = [], total_pages: int = 0, first_page: str = "") -> str:
    context = "\n\n".join(chunks)

    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    # inject document metadata
    metadata_text = ""
    if total_pages > 0:
        metadata_text = f"Document metadata: This PDF has {total_pages} pages total.\n"
    if first_page:
        metadata_text += f"First page content: {first_page[:500]}\n"

    system_prompt = """You are a helpful assistant answering questions about a book or document.

Instructions:
- For summary questions, synthesize all available context into a coherent answer
- For chapter questions, describe events, characters, and themes you can identify
- Be detailed and specific — don't just list what you can't find
- If context is partial, give the best answer possible from what's available
- Never say you can't find something without first giving what you DO know
- Use the previous conversation to understand follow up questions
- Use document metadata to answer questions about page count or document structure"""

    user_prompt = f"""{metadata_text}
{history_text}
Context from document:
{context}

Current Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=1000
    )

    return response.choices[0].message.content