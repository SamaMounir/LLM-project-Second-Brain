import os
import streamlit as st
from kb import KnowledgeBase
from agent import Memory, LLMClient, run_agent
 
st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")
st.markdown("""<style>
.stMarkdown p, .stMarkdown li, [data-testid="stChatMessage"] p {unicode-bidi: plaintext; text-align: start;}
</style>""", unsafe_allow_html=True)  # Arabic text aligns right automatically
 
@st.cache_resource
def get_kb(): return KnowledgeBase()
kb = get_kb()
mem = Memory()
 
# ---------------- Sidebar: ingestion ----------------
with st.sidebar:
    st.title("🧠 Second Brain")
    llm_url = st.text_input("LLM server (ngrok URL)", value=os.getenv("LLM_URL", ""))
    token = st.text_input("Server token", type="password", value=os.getenv("LLM_TOKEN", "secret123"))
    st.divider()
    st.subheader("Add knowledge")
    files = st.file_uploader("PDF / DOCX / TXT / MD", accept_multiple_files=True,
                             type=["pdf", "docx", "txt", "md"])
    if st.button("Index files") and files:
        for f in files:
            n = kb.add(f.name, kb.extract(f.name, f.read()), kind="file")
            st.success(f"{f.name}: {n} chunks")
    with st.expander("Quick note"):
        title = st.text_input("Title"); body = st.text_area("Note")
        if st.button("Save note") and title and body:
            kb.add(title, body, kind="note"); st.success("Saved")
    with st.expander("Bookmark (URL)"):
        url = st.text_input("URL")
        if st.button("Fetch & index") and url:
            try: st.success(f"{kb.add(url, kb.fetch_url(url), kind='bookmark')} chunks")
            except Exception as e: st.error(e)
 
tab_chat, tab_lib, tab_tasks, tab_mem = st.tabs(["💬 Chat", "📚 Library", "✅ Tasks", "🧩 Memory"])
 
# ---------------- Chat ----------------
with tab_chat:
    for m in mem.d["history"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Ask your second brain…  e.g. 'What did I study about neural networks last month?'"):
        if not llm_url: st.error("Paste the ngrok URL of your Kaggle LLM server in the sidebar."); st.stop()
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try: ans, trace = run_agent(LLMClient(llm_url, token), q, kb, mem)
                except Exception as e: st.error(f"LLM server error: {e}"); st.stop()
            st.markdown(ans)
            if trace:
                with st.expander(f"🔍 Agent steps ({len(trace)} tool calls)"):
                    for t in trace:
                        st.code(f"{t['tool']}({t['input']})", language="python")
                        if t["tool"] in ("search_notes", "auto_search", "bridge_search", "sample_passages") and isinstance(t["output"], list):
                            for h in t["output"]:
                                st.caption(f"**{h['source']}** · {h['date']} · score {h['score']}")
                                st.write(h["text"][:300] + "…")
    if st.button("Clear conversation"): mem.clear_history(); st.rerun()
 
# ---------------- Library ----------------
with tab_lib:
    docs = kb.documents()
    st.metric("Documents", len(docs))
    for d in docs:
        c1, c2 = st.columns([6, 1])
        c1.write(f"**{d['source']}** · {d['kind']} · {d['date']} · {d['chunks']} chunks")
        if c2.button("🗑", key="del" + d["source"]): kb.delete(d["source"]); st.rerun()
 
# ---------------- Tasks ----------------
with tab_tasks:
    if not mem.d["tasks"]: st.info("Ask: 'Turn my CNN notes into a study plan'.")
    for i, t in enumerate(mem.d["tasks"]):
        done = st.checkbox(f"{t['title']}  _{t.get('source','')} {t.get('due','')}_", t["done"], key=f"t{i}")
        if done != t["done"]: mem.d["tasks"][i]["done"] = done; mem.save()
 
# ---------------- Memory ----------------
with tab_mem:
    st.subheader("Preferences"); st.json(mem.d["preferences"])
    st.subheader("Facts about you"); st.write(mem.d["facts"] or "—")
    st.caption("Try: 'I prefer short bullet-point summaries' — it persists across sessions.")