"""Agent + memory on a self-hosted open LLM (Kaggle GPU -> FastAPI -> ngrok), ReAct-style."""
import json, re, datetime as dt, requests
from pathlib import Path
 
DATA = Path("data"); DATA.mkdir(exist_ok=True)
 
# ---------------- LLM client (same endpoint contract as the course ngrok notebook) ----------------
class LLMClient:
    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "ngrok-skip-browser-warning": "true"}
 
    def chat(self, messages, max_length=600):
        r = requests.post(f"{self.url}/generate", headers=self.headers, timeout=300,
                          json={"messages": messages, "max_length": max_length, "temperature": 0.2})
        r.raise_for_status()
        return r.json()["response"].strip()
 
# ---------------- Memory ----------------
class Memory:
    """Persistent memory: preferences + learned facts (long-term), chat history (short-term), tasks."""
    def __init__(self, path=DATA / "memory.json"):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "preferences": {}, "facts": [], "history": [], "tasks": []}
 
    def save(self): self.path.write_text(json.dumps(self.d, indent=2, ensure_ascii=False), encoding="utf-8")
 
    def add_turn(self, role, text):
        self.d["history"].append({"role": role, "content": text}); self.save()
 
    def recent(self, n=4):
        h = [dict(m) for m in self.d["history"][-n:]]
        while h and h[0]["role"] != "user": h = h[1:]
        return h
 
    def clear_history(self): self.d["history"] = []; self.save()
 
# ---------------- Tools ----------------
TOOLS_DOC = """search_notes  {"query": str, "since": "YYYY-MM-DD"?, "until": "YYYY-MM-DD"?, "source": str?, "k": int?}
   Semantic search over the user's notes/PDFs/docs. Use since/until for time words ("last month", "yesterday").
   Call it several times (e.g. once per document via "source") to connect ideas across documents.
list_documents  {}
   List all documents with upload dates.
create_tasks  {"tasks": [{"title": str, "source": str?, "due": str?}]}
   Save actionable study/to-do tasks.
remember  {"kind": "preference" | "fact", "key": str?, "value": str}
   Store a user preference or durable fact."""
 
def run_tool(name, args, kb, mem):
    if name == "search_notes":
        hits = kb.search(args["query"], min(int(args.get("k", 3)), 4), args.get("since"), args.get("until"), args.get("source"))
        for h in hits: h["text"] = h["text"][:700]      # keep the prompt short = faster replies
        return hits or "No results."
    if name == "list_documents":
        return kb.documents() or "Knowledge base is empty."
    if name == "create_tasks":
        for t in args["tasks"]:
            mem.d["tasks"].append({**t, "done": False, "created": dt.date.today().isoformat()})
        mem.save(); return f"Saved {len(args['tasks'])} tasks."
    if name == "remember":
        if args["kind"] == "preference":
            mem.d["preferences"][args.get("key", "note")] = args["value"]
        elif args["value"] not in mem.d["facts"]:
            mem.d["facts"].append(args["value"])
        mem.save(); return "Remembered."
    return f"Unknown tool '{name}'."
 
def system_prompt(mem):
    return f"""You are a personal "second brain" assistant. Today is {dt.date.today().isoformat()}.
You answer ONLY from the user's own notes, which you access with tools.
 
TOOLS:
{TOOLS_DOC}
 
On every turn reply in EXACTLY one of two formats.
To use a tool:
ACTION: <tool name>
ARGS: <one-line JSON>
To answer the user (after you have observations):
FINAL: <your answer>
 
Rules:
- The user message includes notes retrieved automatically. If they answer the question, answer from them. If not (comparing documents, a time period such as today/yesterday/last month, or another topic), call search_notes; use since/until only when the user mentions a time period, converted to explicit dates. If nothing relevant is found, say so. Never answer knowledge questions from your own memory.
- To relate two topics/documents, search each one separately, then explain what they share, how they differ and how one builds on the other.
- Cite sources as [document name, date]. If nothing is found, say so. Never invent notes.
- Every retrieved chunk has a 'source' field. Attribute text ONLY to the source named in that chunk and never mix sources up. If a document has no relevant chunk, say so instead of guessing what it contains.
- For tasks or study plans, call create_tasks, then give a FINAL answer.
- If the user states a preference or personal context, call remember.
- Never output more than one ACTION per turn. Wait for the Observation.
- Answer in the same language as the user's question (Arabic or English). The notes may be in either language, so write the search query in the language of the notes you expect, or search twice (Arabic and English). Keep the words ACTION, ARGS, FINAL and tool names in English.
 
User preferences: {json.dumps(mem.d['preferences'], ensure_ascii=False)}
Known facts about the user: {json.dumps(mem.d['facts'], ensure_ascii=False)}"""
 
def parse(text):
    """-> ('final', answer) | ('action', (name, args_or_None))"""
    if "FINAL:" in text:
        return "final", text.split("FINAL:", 1)[1].strip()
    m = re.search(r"ACTION:\s*(\w+)", text)
    if not m:
        return "final", text.strip()          # small models sometimes skip the format
    args, i = {}, text.find("{", m.end())
    if i != -1:
        try: args, _ = json.JSONDecoder().raw_decode(text[i:])
        except json.JSONDecodeError: args = None
    return "action", (m.group(1), args)
 
RELATE = re.compile(r"relat|connect|compar|link|similar|differ|both|between|together|علاق|ترتبط|يرتبط|قارن|مقارن|رابط|الفرق|بين", re.I)
 
def _clip(hits, n=500):
    for h in hits: h["text"] = h["text"][:n]
    return hits
 
AR = re.compile(r"[؀-ۿ]")
NOTE_WORD = re.compile(r"\bnotes?\b|ملاحظ|مذكر", re.I)
ARABIC_WORD = re.compile(r"arabic|عربي|العربي|بالعربية", re.I)
 
def resolve_sources(question, docs):
    """Which documents does the question refer to? By name ("Lecture 1", the note's title)
    or by description ("my Arabic note"). Returns a list of exact source names."""
    q = question.lower()
    found = []
    for d in docs:
        name = d["source"]
        stem = re.sub(r"\.(pdf|docx?|txt|md)$", "", name, flags=re.I).lower()
        if stem and stem.replace("_", " ") in q.replace("_", " "):
            found.append(name)
    if not found and NOTE_WORD.search(question):
        notes = [d["source"] for d in docs if "note" in str(d.get("kind", "")).lower()]
        if ARABIC_WORD.search(question):
            notes = [n for n in notes if AR.search(n)]
        if 1 <= len(notes) <= 2:
            found = notes
    return found
 
def gather_context(kb, question):
    """Automatic retrieval. For cross-document questions, use the smallest document's chunks as queries
    against the other documents ("bridge" retrieval) so related passages from EVERY source are found."""
    docs = kb.documents()
    names = [d["source"] for d in docs]
    if RELATE.search(question) and 2 <= len(names) <= 4:
        seed = min(docs, key=lambda d: d["chunks"])["source"]
        seed_hits = kb.search(question, 2, source=seed)
        hits = list(seed_hits)
        if seed_hits:
            for other in names:
                if other != seed:
                    hits += kb.search(seed_hits[0]["text"], 3, source=other)
        for other in names:                          # make sure every source is represented
            if other != seed and not any(h["source"] == other for h in hits):
                hits += kb.search(question, 1, source=other)
        seen, uniq = set(), []
        for h in hits:
            key = (h["source"], h["text"][:80])
            if key not in seen: seen.add(key); uniq.append(h)
        return _clip(uniq), "bridge_search", docs
    named = resolve_sources(question, docs)
    if named:                                        # question names a document: search only inside it
        per = 3 if len(named) == 1 else 2
        hits = []
        for n in named:
            hits += kb.search(question, per, source=n)
        if hits:
            return _clip(hits), "source_search", docs
    return _clip(kb.search(question, 3)), "auto_search", docs
 
PREF_OPEN = re.compile(r"^\s*(i prefer|i like|i want|i'd like|always|from now on|remember|أفضل|افضل|أحب|احب|أريد|اريد|دائما|دائماً|من الآن|تذكر)", re.I)
PREF_STYLE = re.compile(r"answer|repl|respon|summar|bullet|short|brief|detail|concise|format|language|arabic|english|إجاب|اجاب|رد|مختصر|نقاط|مفصل|عرب|انجليز|إنجليز|لغة", re.I)
PREF_LANG = re.compile(r"language|arabic|english|لغة|عرب|انجليز|إنجليز", re.I)
REMEMBER = re.compile(r"^\s*(remember|تذكر)", re.I)
 
def save_preference(user_msg, mem):
    """Rule-based memory writer (a 7B model is not reliable at calling the remember tool). Returns a dict or None."""
    if "?" in user_msg or "؟" in user_msg or not PREF_OPEN.match(user_msg):
        return None
    text = user_msg.strip()
    if REMEMBER.match(text):
        if text not in mem.d["facts"]: mem.d["facts"].append(text)
        mem.save(); return {"kind": "fact", "text": text}
    if not PREF_STYLE.search(text):
        return None
    key = "language" if PREF_LANG.search(text) else "answer_style"
    mem.d["preferences"][key] = text
    mem.save(); return {"kind": "preference", "key": key, "text": text}
 
TASK_REQ = re.compile(r"(make|create|generate|build|give me|turn|convert|write|draft|أنشئ|اعمل|كوّن|كون|اكتب|أعطني|اعطني|حوّل|حول).{0,50}(study plan|plan|tasks?|to-?do|checklist|خطة|مهام|مذاكرة)", re.I | re.S)
TASK_LINE = re.compile(r"^\W*TASK:\s*(.+?)\s*\|\s*(.+?)\s*$", re.I | re.M)
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
 
def make_tasks(llm, user_msg, kb, mem, lang, tag):
    """Deterministic study-plan flow: sample passages from every document, ask for TASK lines, save them."""
    names = [d["source"] for d in kb.documents()]
    if not names:
        return None
    hits = []
    for n in names[:4]:
        hits += kb.sample(n, 6 if len(names) == 1 else 4)
    for h in hits: h["text"] = h["text"][:350]
    ctx = json.dumps(hits, ensure_ascii=False)[:5000]
    prompt = (f"You are a study assistant. Today is {dt.date.today().isoformat()}. "
              f"Facts about the user: {json.dumps(mem.d['facts'], ensure_ascii=False)}.\n\n"
              f"Request: {user_msg}\n\n[Documents]\n" + ", ".join(names) + f"\n\n[Passages]\n{ctx}\n\n"
              f"[Task] Create 5 to 8 concrete, actionable study tasks based ONLY on the passages (cover the main topics of each document). "
              f"Output ONLY lines in exactly this format, no other text:\nTASK: <short task in {lang}> | <exact document name from the list>{tag}")
    out = llm.chat([{"role": "user", "content": prompt}], max_length=500)
    tasks, titles = [], {t["title"] for t in mem.d["tasks"]}
    for title, src in TASK_LINE.findall(out):
        title = title.strip()
        if title and not CJK.search(title) and title not in titles:
            titles.add(title)
            tasks.append({"title": title, "source": src.strip() if src.strip() in names else ""})
    if not tasks:
        return None
    for t in tasks:
        mem.d["tasks"].append({**t, "done": False, "created": dt.date.today().isoformat()})
    mem.save()
    bullets = "\n".join(f"* {t['title']}" + (f" [{t['source']}]" if t["source"] else "") for t in tasks)
    head = (f"أضفت {len(tasks)} مهام إلى تبويب المهام:" if lang == "Arabic"
            else f"I added {len(tasks)} tasks to your Tasks tab:")
    return f"{head}\n\n{bullets}", [{"tool": "sample_passages", "input": {"documents": names}, "output": hits},
                                      {"tool": "create_tasks", "input": {"tasks": tasks}, "output": f"Saved {len(tasks)} tasks"}]
 
def run_agent(llm, user_msg, kb, mem, max_steps=5):
    """ReAct loop. Returns (answer, trace)."""
    lang = "Arabic" if re.search(r"[\u0600-\u06FF]", user_msg) else "English"
    tag = f"\n\n[Reply language: {lang}, unless a saved preference says otherwise. Never reply in Chinese.]"
    saved = save_preference(user_msg, mem)
    if saved:                                         # preference/fact statement: store it, no retrieval needed
        ack = (f"تم، سأتذكر ذلك: «{saved['text']}»" if lang == "Arabic"
               else f"Got it, I'll remember: \u201c{saved['text']}\u201d")
        mem.add_turn("user", user_msg); mem.add_turn("assistant", ack)
        return ack, [{"tool": "remember", "input": saved, "output": "Saved to memory"}]
    if mem.d["preferences"]:
        tag += "\n[Follow the user's saved preferences: " + json.dumps(mem.d["preferences"], ensure_ascii=False) + "]"
    if TASK_REQ.search(user_msg):                     # study plan / task list request
        made = make_tasks(llm, user_msg, kb, mem, lang, tag)
        if made:
            mem.add_turn("user", user_msg); mem.add_turn("assistant", made[0])
            return made
    trace, done, retried = [], set(), False
    try:                                              # automatic retrieval first: reliable even for small models
        hits, mode, docs = gather_context(kb, user_msg)
    except Exception:
        hits, mode, docs = [], "auto_search", []
    trace.append({"tool": mode, "input": {"query": user_msg}, "output": hits})
    ctx = json.dumps(hits, ensure_ascii=False) if hits else "No results."
    doc_list = "\n".join(f"- {d['source']} ({d['kind']}, {d['date']})" for d in docs) or "(empty)"
    task = ("\n\n[Task] Retrieval is complete. Do NOT call tools. Using ONLY the chunks above, explain how the documents relate: "
            "(1) what each source says about the shared topic, taking each source's content only from chunks whose source field names it; "
            "(2) how they connect or differ; (3) what is not shown in the retrieved passages. Say 'not shown in the retrieved passages'; never claim that a document does not cover a topic, because you only see a few passages. Do not call the documents' wording different unless the passages show it. Reply with FINAL:."
            if mode == "bridge_search" else "")
    first = (f"{user_msg}\n\n[Your documents]\n{doc_list}\n\n"
             f"[Notes retrieved automatically; each chunk shows its own source]\n{ctx[:4500]}{task}{tag}")
    msgs = mem.recent() + [{"role": "user", "content": first}]
    msgs[0]["content"] = system_prompt(mem) + "\n\n---\n" + msgs[0]["content"]   # system text folded into 1st user turn
    for _ in range(max_steps):
        out = llm.chat(msgs)
        kind, val = parse(out)
        if kind == "final":
            if CJK.search(val) and not retried:       # Qwen sometimes drifts into Chinese: ask for one clean rewrite
                retried = True
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": f"Rewrite your answer in {lang} only, with no Chinese characters, "
                                                     "and copy document names exactly as listed in [Your documents]." + tag}]
                continue
            mem.add_turn("user", user_msg); mem.add_turn("assistant", val)
            return val, trace
        name, args = val
        if args is None:
            obs = "Observation: ARGS was not valid JSON. Retry with one-line valid JSON."
        elif (mode == "bridge_search" and name in ("search_notes", "list_documents")) \
                or (name, json.dumps(args, sort_keys=True, ensure_ascii=False)) in done:
            obs = "Observation: Retrieval is already complete. Do not search again. Reply now with FINAL using only the notes retrieved above."
        else:
            done.add((name, json.dumps(args, sort_keys=True, ensure_ascii=False)))
            try: result = run_tool(name, args, kb, mem)
            except Exception as e: result = f"Tool error: {e}"
            trace.append({"tool": name, "input": args, "output": result})
            obs = "Observation: " + (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))[:2500]
        msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": obs + tag}]
    return "I hit the step limit. Try a more specific question.", trace