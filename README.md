🧠 AI Personal Knowledge Manager

A personal "second brain" that lets you upload notes, PDFs, documents and bookmarks, then chat with them. It combines RAG (retrieval-augmented generation), an agent that decides how to search and what to do with the results, and persistent memory that remembers your preferences and past conversations.

Final project for the LLM course. Built with Streamlit, ChromaDB, sentence-transformers and a 4-bit quantized open LLM served from a Kaggle GPU.

Features
Ingest anything: PDF, DOCX, TXT, Markdown, typed notes, and web pages (bookmarks).
Semantic search (RAG): everything is chunked, embedded and stored in a vector database.
Agent with tools: retrieves relevant chunks, summarizes, connects ideas across documents, generates tasks and answers questions, citing the source document and date.
Time-aware queries: each chunk stores its upload date, so questions like "What did I study last month?" or "Compare that with the lecture I uploaded yesterday" become date-filtered searches.
Memory: remembers your preferences (e.g. "short bullet-point answers"), facts about you, and recent conversation turns. Stored on disk, so it survives restarts.
Transparent: an "Agent steps" panel shows every tool call and the retrieved chunks with similarity scores.
Task manager: the agent can turn your material into a study plan, saved as a checklist.

Usage examples:
Upload a lecture PDF, click Index files, then ask: "What do my notes say about signals?"
"What did I upload yesterday?"
"How does my Fourier Transform note relate to Lecture 1?" (cross-document connection)
"Make a study plan from Lecture 1" (tasks appear in the Tasks tab)
"I prefer short bullet-point answers" (saved to memory)

Architecture:
 Your computer                                   Kaggle (GPU T4)
┌─────────────────────────────────────┐        ┌──────────────────────────┐
│ Streamlit UI (app.py)               │        │ FastAPI  POST /generate  │
│   │                                 │  HTTPS │   │                      │
│   ▼                                 │ (ngrok)│   ▼                      │
│ Agent (agent.py)  ReAct loop  ──────┼───────►│ 4-bit quantized LLM      │
│   │  tools:                         │◄───────┼── (Mistral-Nemo-12B)     │
│   │  search_notes, list_documents,  │        └──────────────────────────┘
│   │  create_tasks, remember         │
│   ▼                                 │
│ Knowledge base (kb.py)              │
│   ChromaDB + MiniLM embeddings      │
│ Memory (data/memory.json)           │
└─────────────────────────────────────┘