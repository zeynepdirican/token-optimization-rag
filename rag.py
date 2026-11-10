
import requests
import numpy as np
import re
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


# 1. EMBEDDINGS
def embed(texts):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": texts}
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


# 2. ASK LLM
def ask_llm(prompt):
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    r.raise_for_status()
    data = r.json()

    print("Input tokens:", data["prompt_eval_count"])
    print("Output tokens:", data["eval_count"])

    return data["message"]["content"]



# 3. LOAD DOCUMENTS — SECTION-AWARE CHUNKING
chunks = []
sources = []

for file in Path("docs").glob("*.md"):
    text = file.read_text(encoding="utf-8")

    # Each Markdown subsection becomes one chunk
    sections = re.split(r"(?=^###\s)", text, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        chunks.append(section.strip())
        sources.append(file.name)

chunk_vectors = embed(chunks)

print("Total chunks:", len(chunks))
print("Average words:", sum(len(c.split()) for c in chunks) / len(chunks))


# 4. RETRIEVE TOP-K CHUNKS
def retrieve(question, k=5):
    q = embed([question])[0]

    scores = chunk_vectors @ q / (
        np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    )

    indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], sources[i]) for i in indices]


# 5. COMPRESS CONTEXT
def compress_context(question, retrieved, max_tokens=200):
    units = []

    for chunk, source in retrieved:
        # Split text into paragraphs, keeping numbered lists together
        parts = re.split(r"\n\s*\n", chunk)

        for part in parts:
            if part.strip():
                units.append((part.strip(), source))

    if not units:
        return ""

    # Rank each unit by relevance to the question
    vectors = embed([question] + [text for text, _ in units])
    q = vectors[0]
    s = vectors[1:]

    scores = s @ q / (
        np.linalg.norm(s, axis=1) * np.linalg.norm(q)
    )

    selected = []
    used = 0

    for i in np.argsort(scores)[::-1]:
        text, source = units[i]
        item = f"[{source}] {text}"
        cost = int(len(item.split()) * 1.5)

        if used + cost <= max_tokens:
            selected.append(item)
            used += cost

    return "\n\n".join(selected)


# 6. COMPARE BASELINE VS OPTIMIZED
if __name__ == "__main__":
    while True:
        question = input("\nAsk (or 'exit'): ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        retrieved = retrieve(question)

        baseline = "\n\n".join(
            f"[{source}] {chunk}" for chunk, source in retrieved
        )

        optimized = compress_context(question, retrieved, max_tokens=200)

        # Check whether compression actually reduces context
        print("\nBASELINE CONTEXT:\n", baseline)
        print("\nOPTIMIZED CONTEXT:\n", optimized)

        print("\nBaseline characters:", len(baseline))
        print("Optimized characters:", len(optimized))

        # Compare both methods
        for name, context in [
            ("BASELINE", baseline),
            ("OPTIMIZED", optimized)
        ]:
            print(f"\n--- {name} ---")

            prompt = f"""Answer using ONLY the context below.
    If the answer is missing, say "I don't know".

    Context:
    {context}

    Question: {question}"""

            answer = ask_llm(prompt)
            print("Answer:", answer)
