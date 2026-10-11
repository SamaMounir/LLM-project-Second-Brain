# 🧠 AI Personal Knowledge Manager

A personal "second brain": upload notes, PDFs, documents and bookmarks, then chat with them in **English or Arabic**. It combines **RAG** (retrieval-augmented generation), an **agent** that decides how to search and what to do with the results, and **persistent memory** for your preferences and past conversations.

Final project for the LLM course. Built with Streamlit, ChromaDB, sentence-transformers and a 4-bit quantized open LLM (Qwen2.5-7B-Instruct) served from a Kaggle GPU.

## Features

- **Ingest anything:** PDF, DOCX, TXT, Markdown, typed notes and web pages (bookmarks).
- **Bilingual semantic search:** multilingual embeddings, so an Arabic question can find English text and the reverse.
- **Agent with tools:** retrieves, summarizes, connects ideas across documents, generates tasks and answers questions with citations (document name and date).
- **Time-aware queries:** every chunk stores its upload date, so "what did I upload yesterday?" becomes a date-filtered search.
- **Cross-document reasoning:** for "how does X relate to Y?" questions, the app uses passages from your shortest document to find related passages in every other document ("bridge retrieval").
- **Memory:** remembers preferences ("short bullet-point answers", "reply in Arabic"), facts about you ("I'm studying for my DSP final") and recent turns. Stored on disk, so it survives restarts.
- **Study plans:** "make a study plan from my notes" samples passages across all documents and saves actionable tasks to a checklist.
- **Transparent:** an "Agent steps" panel shows every retrieval step, the retrieved chunks and similarity scores.

## Architecture

```
 Your computer                                   Kaggle (GPU T4)
┌──────────────────────────────────────┐       ┌──────────────────────────┐
│ Streamlit UI (app.py)                │       │ FastAPI  POST /generate  │
│   │                                  │ HTTPS │   │                      │
│   ▼                                  │(ngrok)│   ▼                      │
│ Agent (agent.py)  ReAct loop  ───────┼──────►│ Qwen2.5-7B-Instruct,     │
│   • auto / bridge retrieval          │◄──────┼── 4-bit (bitsandbytes)   │
│   • rule-based memory + study plans  │       └──────────────────────────┘
│   ▼                                  │
│ Knowledge base (kb.py)               │
│   ChromaDB + multilingual MiniLM     │
│ Memory (data/memory.json)            │
└──────────────────────────────────────┘
```

Your documents and vector database stay on your computer. Only the question and the few retrieved passages are sent to the model server.

## How it works

1. **Ingestion:** text is extracted (PyMuPDF for PDFs), split into ~80-word overlapping chunks, embedded with `paraphrase-multilingual-MiniLM-L12-v2` and stored in ChromaDB with metadata (source, kind, upload date). Re-adding a document with the same name replaces the old copy.
2. **Automatic retrieval:** every question first triggers a search (`auto_search`) whose results are placed in the prompt. Small models are unreliable at deciding when to search, so retrieval is not left to them.
3. **Agent loop (ReAct):** the model answers with `FINAL` or calls a tool (`search_notes`, `list_documents`, `create_tasks`, `remember`) through `ACTION` + JSON arguments, for up to 5 steps. Repeated identical tool calls are skipped.
4. **Cross-document questions:** when a question asks to relate or compare, the retrieval step becomes `bridge_search`, further searching is blocked, and the model gets an explicit task: what each source says, how they connect, what is not shown.
5. **Memory:** preference and fact statements ("I prefer…", "always reply in…", "remember that…", and Arabic equivalents) are detected by rules and saved directly. Saved preferences are repeated at the end of every prompt.
6. **Study plans:** task requests run a dedicated flow (`sample_passages` → `TASK: … | source` lines → saved to the Tasks tab).
7. **Safeguards:** language lock (Arabic or English, never Chinese, with one automatic rewrite if Chinese appears), a rule that text may only be attributed to the source named in its chunk, and "not shown in the retrieved passages" instead of claims that a document lacks a topic.

## Project structure

```
second_brain/
├── app.py                  # Streamlit UI (chat, library, tasks, memory)
├── agent.py                # LLM client, ReAct agent, retrieval modes, memory, tasks
├── kb.py                   # Ingestion, chunking, embeddings, vector search
├── llm_server_kaggle.ipynb # Notebook that serves the LLM (FastAPI + ngrok)
├── requirements.txt
├── README.md
└── data/                   # created on first run (chroma/ and memory.json), git-ignored
```

## Setup

### 1. Start the model server on Kaggle
1. Create a free account at [ngrok.com](https://ngrok.com) and copy your authtoken.
2. In Kaggle, create a notebook and import `llm_server_kaggle.ipynb` (File → Import Notebook).
3. Settings: Accelerator = **GPU T4**, Internet = **On**.
4. Add-ons → Secrets: add a secret named `NGROK_TOKEN` with your authtoken and attach it to the notebook.
5. Click **Run All**. When the last cell prints `PASTE THIS IN STREAMLIT: https://...ngrok-free.dev` and `Uvicorn running`, leave it running.

### 2. Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py --server.fileWatcherType none
```
Paste the ngrok URL (without `/generate`) and the token (`secret123` by default, set by `API_TOKEN` in the notebook) into the sidebar. The first run downloads the embedding model (~470 MB).

## Usage examples

- Upload a lecture PDF, click **Index files**, then ask: *"What is quantization?"*
- *"ما الفرق بين أخذ العينات والتكميم؟"* (Arabic question over an Arabic note)
- *"How does my Arabic note relate to Lecture 1?"* (cross-document connection)
- *"What did I upload yesterday?"* (date-aware)
- *"I prefer short bullet-point answers"* (saved to memory)
- *"Make a study plan from my DSP notes"* (tasks appear in the Tasks tab)

## Manual test cases

| Test | Expected |
|---|---|
| Question answered from a document | Answer matches the source and cites it |
| Topic not in the notes | Says nothing was found, does not invent an answer |
| "What did I upload today / yesterday?" | Uses upload dates correctly |
| Arabic question | Arabic answer based on the retrieved chunks |
| "How does my note relate to Lecture 1?" | `bridge_search`, passages from both sources, no mixing of sources |
| State a preference, restart the app | Preference still shown in the Memory tab |
| "Make a study plan…" | Tasks saved in the Tasks tab with exact source names |

## Case study: a failure and its fix

Asked "How does my Arabic note relate to Lecture 1?", the first version answered with content that came from the PDF but attributed it to the note. Cause: the question has no topic words, so the automatic search only matched the lecture, and the model then ran extra searches that returned unrelated intro slides and trusted the latest results. Fixes: bridge retrieval (the note's passages are used as queries against the lecture), a document list in every prompt, a blocked second search in this mode, a strict source-attribution rule and no claims of absence.


