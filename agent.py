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
        self.d = json.loads(path.read_text()) if path.exists() else {
            "preferences": {}, "facts": [], "history": [], "tasks": []}

    def save(self): self.path.write_text(json.dumps(self.d, indent=2))

    def add_turn(self, role, text):
        self.d["history"].append({"role": role, "content": text}); self.save()

    def recent(self, n=6):
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
        return kb.search(args["query"], args.get("k", 5), args.get("since"), args.get("until"), args.get("source")) or "No results."
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
- Always search before answering knowledge questions. Convert relative dates to explicit since/until dates.
- To relate two topics/documents, search each one separately, then explain what they share, how they differ and how one builds on the other.
- Cite sources as [document name, date]. If nothing is found, say so. Never invent notes.
- For tasks or study plans, call create_tasks, then give a FINAL answer.
- If the user states a preference or personal context, call remember.
- Never output more than one ACTION per turn. Wait for the Observation.

User preferences: {json.dumps(mem.d['preferences'])}
Known facts about the user: {json.dumps(mem.d['facts'])}"""

def parse(text):
    """-> ('final', answer) | ('action', (name, args_or_None))"""
    if "FINAL:" in text:
        return "final", text.split("FINAL:", 1)[1].strip()
    m = re.search(r"ACTION:\s*(\w+)", text)
    if not m:
        return "final", text.strip()          
    args, i = {}, text.find("{", m.end())
    if i != -1:
        try: args, _ = json.JSONDecoder().raw_decode(text[i:])
        except json.JSONDecodeError: args = None
    return "action", (m.group(1), args)

def run_agent(llm, user_msg, kb, mem, max_steps=5):
    """ReAct loop. Returns (answer, trace)."""
    msgs = mem.recent() + [{"role": "user", "content": user_msg}]
    msgs[0]["content"] = system_prompt(mem) + "\n\n---\n" + msgs[0]["content"]   
    trace = []
    for _ in range(max_steps):
        out = llm.chat(msgs)
        kind, val = parse(out)
        if kind == "final":
            mem.add_turn("user", user_msg); mem.add_turn("assistant", val)
            return val, trace
        name, args = val
        if args is None:
            obs = "Observation: ARGS was not valid JSON. Retry with one-line valid JSON."
        else:
            try: result = run_tool(name, args, kb, mem)
            except Exception as e: result = f"Tool error: {e}"
            trace.append({"tool": name, "input": args, "output": result})
            obs = "Observation: " + (result if isinstance(result, str) else json.dumps(result))[:3500]
        msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": obs}]
    return "I hit the step limit. Try a more specific question.", trace
