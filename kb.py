"""Knowledge base: ingestion, chunking, embeddings, vector search (RAG layer)."""
import io, time, uuid, datetime as dt
import chromadb
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
 
_model = None
def embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Arabic + English
    return _model
 
class KnowledgeBase:
    def __init__(self, path="data/chroma"):
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            "notes_multi", metadata={"hnsw:space": "cosine"})  # new name: old English-only vectors are ignored
 
    # ---------- extraction ----------
    @staticmethod
    def extract(name: str, data: bytes) -> str:
        n = name.lower()
        if n.endswith(".pdf"):
            try:  # PyMuPDF handles Arabic text better than pypdf
                import fitz
                with fitz.open(stream=data, filetype="pdf") as doc:
                    return "\n".join(p.get_text() for p in doc)
            except ImportError:
                return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        if n.endswith(".docx"):
            return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        return data.decode("utf-8-sig", errors="ignore")  # txt / md
 
    @staticmethod
    def fetch_url(url: str) -> str:
        html = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]):
            t.decompose()
        return soup.get_text("\n", strip=True)
 
    @staticmethod
    def chunk(text: str, size=80, overlap=15):  # small chunks: this embedding model reads ~128 tokens
        words = text.split()
        step = size - overlap
        return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]
 
    # ---------- write ----------
    def add(self, source: str, text: str, kind="note") -> int:
        chunks = self.chunk(text)
        if not chunks:
            return 0
        self.col.delete(where={"source": source})   # same name = replace, never duplicate
        now = time.time()
        self.col.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embedder().encode(chunks, normalize_embeddings=True).tolist(),
            metadatas=[{"source": source, "kind": kind, "uploaded_at": now,
                        "date": dt.date.fromtimestamp(now).isoformat(), "chunk": i}
                       for i in range(len(chunks))])
        return len(chunks)
 
    def delete(self, source: str):
        self.col.delete(where={"source": source})
 
    # ---------- read ----------
    def search(self, query: str, k=6, since: str | None = None, until: str | None = None,
               source: str | None = None):
        """Semantic search with optional date range (YYYY-MM-DD) / source filter."""
        if self.col.count() == 0:
            return []
        conds = []
        if since:
            conds.append({"uploaded_at": {"$gte": dt.datetime.fromisoformat(since).timestamp()}})
        if until:
            end = dt.datetime.fromisoformat(until) + dt.timedelta(days=1)
            conds.append({"uploaded_at": {"$lt": end.timestamp()}})
        if source:
            conds.append({"source": source})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        q = embedder().encode([query], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=q, n_results=k, where=where)
        return [{"text": d, "source": m["source"], "date": m["date"], "score": round(1 - dist, 3)}
                for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
 
    def sample(self, source, n=6):
        """n evenly spaced chunks from one document (broad coverage, used for study plans)."""
        r = self.col.get(where={"source": source}, include=["documents", "metadatas"])
        items = sorted(zip(r["metadatas"], r["documents"]), key=lambda x: x[0]["chunk"])
        n = max(n, 2)
        if len(items) > n:
            items = [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]
        return [{"text": d, "source": m["source"], "date": m["date"], "score": 1.0} for m, d in items]
 
    def documents(self):
        metas = self.col.get(include=["metadatas"])["metadatas"]
        docs = {}
        for m in metas:
            d = docs.setdefault(m["source"], {"source": m["source"], "kind": m["kind"], "date": m["date"], "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["date"], reverse=True)