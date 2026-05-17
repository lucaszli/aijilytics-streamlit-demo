import os

try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from insurance_rag_langgraph_revised import InsuranceRAGAgent


st.set_page_config(page_title="AIJILYTICS RAG Agent Demo", layout="wide")

st.title("AIJILYTICS Insurance RAG Agent Demo")
st.caption("Streamlit demo comparing Original RAG, Multi-Query RAG, and Corrective RAG")


try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to Streamlit secrets or your local environment.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


@st.cache_resource(show_spinner=True)
def load_agent(rag_mode: str):
    agent = InsuranceRAGAgent(
        persist_directory=f"./streamlit_chroma_db_{rag_mode}",
        collection_name=f"aijilytics_claims_docs_{rag_mode}",
        retrieval_k=3,
        rag_mode=rag_mode,
    )
    agent.load_sample_insurance_data(reset=False)
    return agent


with st.sidebar:
    st.header("Demo Guide")
    st.write("Compare three RAG architectures using the same AIJILYTICS knowledge base.")
    show_docs = st.checkbox("Show retrieved ChromaDB documents", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)

    st.divider()
    person_id = st.selectbox(
        "Select user profile",
        ["Customer", "Broker", "Underwriter", "Admin Demo"],
    )


def ensure_history(tab_key: str):
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    history_key = f"{person_id}_{tab_key}"
    if history_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[history_key] = []
    return history_key, st.session_state.chat_histories[history_key]


def render_debug(result):
    with st.expander("Debug details", expanded=False):
        st.write("Detected intent:", result.get("intent", ""))

        if show_metadata:
            st.write("Metadata")
            st.json(result.get("metadata", {}))

        if show_docs:
            st.write("Retrieved Documents from ChromaDB")
            docs = result.get("retrieved_docs", [])
            if not docs:
                st.info("No documents retrieved. This is expected for off-topic queries.")
            else:
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Retrieved Document {i}"):
                        if isinstance(doc, dict):
                            st.write(doc.get("content", ""))
                            st.json(doc.get("metadata", {}))
                        else:
                            st.write(str(doc))


def render_agent_tab(agent_mode, title, description):
    st.subheader(title)
    st.caption(description)

    agent = load_agent(agent_mode)

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if agent_mode not in st.session_state.chat_histories:
        st.session_state.chat_histories[agent_mode] = []

    messages = st.session_state.chat_histories[agent_mode]

    if st.button("Clear chat", key=f"clear_{agent_mode}"):
        st.session_state.chat_histories[agent_mode] = []
        st.rerun()

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "debug" in msg:
                render_debug(msg["debug"])

    user_message = st.chat_input(
        f"Ask the {title} a question...",
        key=f"chat_input_{agent_mode}"
    )

    if user_message:
        messages.append({"role": "user", "content": user_message})

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner(f"Running {title}..."):
                result = agent.query(user_message)

            final_answer = result.get("final_output", "")
            st.markdown(final_answer)
            render_debug(result)

        messages.append({
            "role": "assistant",
            "content": final_answer,
            "debug": result
        })

        st.rerun()


tab_original, tab_multi, tab_corrective = st.tabs(
    ["Original RAG Agent", "Multi-Query RAG Agent", "Corrective RAG Agent"]
)

with tab_original:
    render_agent_tab(
        "original",
        "Original RAG Agent",
        "Baseline architecture: classifies on-topic/off-topic, retrieves once from ChromaDB using the original user query, synthesizes, then formats the response.",
    )

with tab_multi:
    render_agent_tab(
        "multi_query",
        "Multi-Query RAG Agent",
        "Expanded retrieval architecture: generates multiple search queries, retrieves from ChromaDB in parallel, deduplicates chunks, synthesizes, then formats the response.",
    )

with tab_corrective:
    render_agent_tab(
        "corrective",
        "Corrective RAG Agent",
        "Self-checking architecture: retrieves context, uses an LLM-based sufficiency check, and retries with multi-query retrieval if the first retrieval is weak.",
    )

st.divider()
st.markdown(
    """
### Workflow

`User Query → classify_intent → conditional route → rag_research if on-topic → format_output → final answer`

- **Original RAG** retrieves once using the original query.
- **Multi-Query RAG** expands the query and retrieves from ChromaDB in parallel.
- **Corrective RAG** judges retrieval quality and retries retrieval when context is weak.
- **Off-topic queries** skip retrieval and receive a scope-limited response.
"""
)
