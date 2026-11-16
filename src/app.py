"""Small Streamlit UI for asking questions and inspecting evidence/trajectory."""
from __future__ import annotations

import os

# The corpus is immutable for this assignment and was downloaded by build_index.py.
# Avoid repeated Hugging Face HEAD requests when the UI loads the local cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import streamlit as st
import bm25s

from src.agent import SearchAgent
from src.config import settings
from src.data import load_corpus
from src.retrieval import create_search_tools

st.set_page_config(page_title="COMP5423 Search Agent", page_icon="🔎", layout="wide")
st.title("COMP5423 · Skill-Augmented Search Agent")
st.caption(f"BrowseComp-Plus · {settings.retrieval_mode} retrieval · iterative evidence gathering")
question = st.text_area("Research question", height=110, placeholder="Enter a multi-clue question…")


@st.cache_resource(show_spinner="Loading corpus and saved BM25 index…")
def get_agent() -> SearchAgent:
    corpus = load_corpus(local_files_only=True)
    index = bm25s.BM25.load(str(settings.index_dir), mmap=True)
    return SearchAgent(create_search_tools(corpus, index, top_k=settings.top_k))


if st.button("Search", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching documents and checking evidence…"):
            result = get_agent().run(question.strip())
        st.subheader("Answer")
        st.write(result["answer"])
        final_action = next((step["action"] for step in reversed(result["trajectory"])
                             if isinstance(step.get("action"), dict)
                             and step["action"].get("action") == "answer"), {})
        citations = final_action.get("supporting_doc_ids", [])
        if citations:
            st.caption("Answer evidence: " + ", ".join(citations))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Tool calls", result["tool_calls"])
        col2.metric("API tokens", result["total_tokens"])
        col3.metric("Skills selected", len(result["selected_skills"]))
        col4.metric("Action retries", result["invalid_action_retries"])
        col5.metric("Model calls", result["llm_calls"])
        st.caption(f"Stop reason: {result['stop_reason']}")
        st.write("Selected skills:", ", ".join(result["selected_skills"]) or "None")
        with st.expander("Agent trajectory", expanded=True):
            for step in result["trajectory"]:
                st.json(step)
        with st.expander("Collected evidence"):
            for evidence in result["state"]["evidence"]:
                st.markdown(f"**Document {evidence['doc_id']}** · {evidence.get('url', '')}")
                query_ranks = evidence.get("query_ranks", [])
                if query_ranks:
                    sources = "; ".join(f"rank {item['rank']}: {item['query']}" for item in query_ranks)
                    st.caption(f"Retrieved by: {sources}")
                st.write(evidence["text"][:1400])
    except Exception as exc:
        st.error(str(exc))
