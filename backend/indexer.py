import os
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# 1️⃣ Load resume text
TXT_PATH = "data/resume.txt"
with open(TXT_PATH, "r", encoding="utf-8") as f:
    resume_text = f.read()

# 2️⃣ Split text into ~300-token chunks with 40-token overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40,
    length_function=len
)
chunks = text_splitter.split_text(resume_text)
print(f"[INFO] Total chunks created: {len(chunks)}")

# 3️⃣ Function to query Groq for a chunk and return relevance & answer
def query_chunk(chunk, user_query):
    prompt = f"""
Resume Chunk:
{chunk}

Question: {user_query}

1) Give a relevance score from 0 to 100 for how well this chunk answers the question.
2) Provide a concise answer to the question based on this chunk.
Format:
Score: <number>
Answer: <text>
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()

    # Parse score and answer
    score = 0
    answer = ""
    try:
        lines = content.splitlines()
        for line in lines:
            if line.lower().startswith("score:"):
                score = int(line.split(":")[1].strip())
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
    except Exception as e:
        answer = content
    return score, answer

# 4️⃣ Query all chunks
query = "Tell me about my backend experience"
results = []

for i, chunk in enumerate(chunks):
    score, answer = query_chunk(chunk, query)
    results.append({"chunk_id": i+1, "score": score, "answer": answer})

# 5️⃣ Sort by score descending and pick top-k
top_k = 3
results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

print("\n[TOP-K RESULTS]")
for r in results_sorted:
    print(f"\nChunk {r['chunk_id']} | Score: {r['score']}\nAnswer: {r['answer']}")

# 6️⃣ Optional: Combine top-k answers into a single concise summary
combined_summary = " ".join([r['answer'] for r in results_sorted])
print("\n[COMBINED SUMMARY]")
print(combined_summary)
