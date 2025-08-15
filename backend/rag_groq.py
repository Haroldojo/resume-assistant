import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# Load .env variables
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# System instructions for Groq
SYSTEM_PROMPT = """
You are a professional resume assistant. Follow these rules:
1. Answer ONLY based on the provided resume context.
2. If the question is ambiguous or not in the resume, respond politely that the info is not available.
3. Handle reworded or paraphrased questions consistently.
4. Keep answers concise and clear.
5. Do not hallucinate any details not in the resume.
"""

# Refined RAG query function
def rag_query_refined(user_query, chunks, top_k=3):
    # Step 1: Score relevance of each chunk
    scored_chunks = []
    for chunk in chunks:
        prompt = f"""
Chunk:
{chunk}

Question: {user_query}

Rate relevance from 0 to 100. Output: Score: <number>
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            score = int(response.choices[0].message.content.strip().split(":")[1].strip())
        except:
            score = 0
        scored_chunks.append({"chunk": chunk, "score": score})

    # Step 2: Merge top-k chunks
    top_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
    merged_context = "\n\n".join([c["chunk"] for c in top_chunks])

    # Step 3: Ask Groq with merged context + system instructions
    final_prompt = f"""
Context from resume:
{merged_context}

Question:
{user_query}

Provide a concise and professional answer based ONLY on the context.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt}
        ]
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# Run as standalone for testing
# -----------------------------
if __name__ == "__main__":
    # Example test chunks (replace with your parsed resume chunks)
    test_chunks = [
        "Parul Verma is a Backend Developer Intern at TunicaTech, skilled in Python, FastAPI, and Azure.",
        "She has experience building scalable REST APIs and deploying cloud workloads."
    ]
    test_query = "Whose resume is this?"
    answer = rag_query_refined(test_query, test_chunks)
    print("Test Query:", test_query)
    print("Answer:", answer)
