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
st.caption("Streamlit demo for LangGraph + ChromaDB + RAG workflow")


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
    st.write("Each tab uses the same AIJILYTICS documentation base but applies a different RAG strategy.")
    st.write("Off-topic questions skip retrieval.")
    person_id = st.selectbox(
        "Select user profile",
        ["Customer", "Broker", "Underwriter", "Admin Demo"],
    )
    show_docs = st.checkbox("Show retrieved ChromaDB documents", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)

    st.markdown("---")
    st.markdown("### Suggested tests")
    st.markdown(
        """
- What does the policy say about excess?
- What fields are in the motor accident report form?
- What is the claims notification clause?
- What does the motor certificate say about limitations as to use?
- What does the discharge voucher say about the settlement amount?
"""
    )


def render_debug(result):
    if show_docs:
        st.subheader("Retrieved Documents from ChromaDB / Exact Store")
        docs = result.get("retrieved_docs", [])
        if not docs:
            st.info("No documents retrieved. This is expected for off-topic queries.")
        else:
            for i, doc in enumerate(docs, 1):
                meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                title = f"Retrieved Document {i}"
                if meta:
                    title += f" — {meta.get('source', 'Unknown source')}"
                    if meta.get("page"):
                        title += f" | page {meta.get('page')}"
                with st.expander(title):
                    if isinstance(doc, dict):
                        st.markdown(doc.get("content", ""))
                        st.json(meta)
                    else:
                        st.write(str(doc))

    if show_metadata:
        st.subheader("Metadata")
        st.json(result.get("metadata", {}))


def render_agent_tab(agent_mode, title, description):
    st.subheader(title)
    st.caption(description)

    agent = load_agent(agent_mode)
    history_key = f"{person_id}_{agent_mode}"

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if history_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[history_key] = []

    messages = st.session_state.chat_histories[history_key]

    if st.button("Clear this tab's chat", key=f"clear_{history_key}"):
        st.session_state.chat_histories[history_key] = []
        st.rerun()

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "debug" in msg:
                with st.expander("Debug details from this response"):
                    render_debug(msg["debug"])

    with st.form(key=f"form_{history_key}", clear_on_submit=True):
        user_message = st.text_input(
            "Ask a question",
            placeholder="Example: What does the policy say about excess?",
            key=f"input_{history_key}",
        )
        submitted = st.form_submit_button("Run Agent")

    if submitted and user_message:
        messages.append({"role": "user", "content": user_message})

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner(f"Running {title}..."):
                # Use only the latest user message for intent classification and retrieval.
                result = agent.query(user_message)

            final_answer = result.get("final_output", "")
            st.markdown(final_answer)

            with st.expander("Debug details from this response"):
                render_debug(result)

        messages.append({
            "role": "assistant",
            "content": final_answer,
            "debug": result,
        })

        st.rerun()


tab1, tab2, tab3, tab4 = st.tabs([
    "Original RAG Agent",
    "Multi-Query RAG Agent",
    "Corrective RAG Agent",
    "Hybrid Exact-Match RAG Agent",
])

with tab1:
    render_agent_tab(
        "original",
        "Original RAG Agent",
        "Baseline architecture: classifies on-topic/off-topic, retrieves once from ChromaDB using the original user query, synthesizes, then formats the response.",
    )

with tab2:
    render_agent_tab(
        "multi_query",
        "Multi-Query RAG Agent",
        "Generates multiple retrieval queries, searches ChromaDB with each query, deduplicates results, synthesizes, then formats the response.",
    )

with tab3:
    render_agent_tab(
        "corrective",
        "Corrective RAG Agent",
        "Retrieves documents, checks whether the context is sufficient, and retries retrieval with improved queries when the first retrieval is weak.",
    )

with tab4:
    render_agent_tab(
        "hybrid_exact",
        "Hybrid Exact-Match RAG Agent",
        "Uses exact keyword/phrase matching over policy, certificate, claim-form, and discharge-voucher excerpts, plus vector fallback. This tab preserves exact document language word-for-word.",
    )

st.divider()
st.markdown(
    """
### Workflow

`User Query → classify_intent → conditional route → selected RAG strategy → format_output → final answer`

- **Original RAG** retrieves once from ChromaDB.
- **Multi-Query RAG** generates multiple search queries and retrieves broader context.
- **Corrective RAG** checks retrieval quality and retries if needed.
- **Hybrid Exact-Match RAG** looks for exact policy/form/certificate/DV language and preserves exact wording.
- **All agents** use the AIJILYTICS workflow summaries plus exact uploaded documentation excerpts.
"""
)
