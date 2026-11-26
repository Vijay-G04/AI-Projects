# Chatbot Over Your Own Documents (Simple RAG)

**Level:** Beginner | **Estimated time:** 5 to 7 days | **Domain:** NLP / LLM applications

## What this project shows
Retrieval-augmented generation: chunking, embeddings, similarity search, and prompting an LLM with retrieved context. Very current and valuable on a resume.

## Dataset
Your own PDFs: lecture notes, a textbook chapter, or a research paper. Keep it small (10 to 50 pages) to start.

## Tech stack
Python, `pypdf`, `sentence-transformers`, NumPy, an LLM API (this example uses the Anthropic SDK), Streamlit or Gradio.

## Suggested folder structure
```
doc-chatbot/
├── docs/                # your PDFs (do not commit private files)
├── src/
│   ├── ingest.py        # read PDFs, chunk, embed
│   ├── retrieve.py      # similarity search
│   └── chat.py          # prompt + LLM call
├── app.py
├── requirements.txt
├── .env.example         # ANTHROPIC_API_KEY=your_key_here
└── README.md
```

## Step-by-step plan
1. Extract text from each PDF page with `pypdf`.
2. Split text into overlapping chunks (around 800 characters with about 100 overlap).
3. Embed every chunk with a sentence-transformers model and store the vectors in a NumPy array (no vector database needed at first).
4. For a question, embed it and retrieve the top 4 chunks by cosine similarity.
5. Send the retrieved chunks plus the question to the LLM, instructing it to answer only from the context and to say when the answer is not there.
6. Show the source chunk or page number next to each answer so users can verify it.
7. Evaluate with 10 to 15 questions you wrote yourself. Record how often the answer is correct and grounded.
8. Wrap it in a Streamlit or Gradio chat interface.

## Starter code (`src/chat.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import anthropic

# --- Ingest ---------------------------------------------------------------
def load_chunks(pdf_path, size=800, overlap=100):
    text = " ".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunks = load_chunks("docs/notes.pdf")
embeddings = embedder.encode(chunks, normalize_embeddings=True)

# --- Retrieve -------------------------------------------------------------
def retrieve(question, k=4):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    top = np.argsort(embeddings @ q)[::-1][:k]
    return [chunks[i] for i in top]

# --- Generate -------------------------------------------------------------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

def ask(question):
    context = "\n\n---\n\n".join(retrieve(question))
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.messages.create(
        model="claude-sonnet-5-5",  # check the docs for the current model name
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

if __name__ == "__main__":
    print(ask("What are the main topics covered in this document?"))
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Your evaluation set and the accuracy you measured.
- Two or three example conversations with source citations.
- A screenshot or GIF of the chat interface.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Support multiple PDFs and show which file each answer came from.
- Use a vector store such as FAISS or Chroma.
- Add conversation memory for follow-up questions.
- Experiment with chunk size and top-k and report what works best.

## Resume line
Built a retrieval-augmented chatbot over course documents using sentence-transformer embeddings and an LLM API, with source citations and a custom evaluation set.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
