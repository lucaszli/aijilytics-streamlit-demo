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
def load_agent():
    agent = InsuranceRAGAgent(persist_directory="./streamlit_chroma_db", retrieval_k=3)
    agent.load_sample_insurance_data(reset=False)
    return agent


agent = load_agent()

with st.sidebar:
    st.header("Demo Guide")
    st.write("On-topic questions use ChromaDB retrieval and RAG synthesis. Off-topic questions skip retrieval.")
    example_question = st.selectbox(
        "Example questions",
        [
            "How does a customer initiate a claim in AIJILYTICS?",
            "What are the NIIRA 2025 compliance requirements for claim settlement?",
            "How does broker onboarding work in AIJILYTICS?",
            "How does the business-facing AI agent support broker negotiation?",
            "What is the purpose of ChromaDB in this RAG prototype?",
            "How does the RAG agent use policy documents?",
            "How does car insurance work in the United States?",
            "What is the best stock to buy right now?",
        ],
    )
    show_docs = st.checkbox("Show retrieved ChromaDB documents", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)

query = st.text_area(
    "Ask a question about the AIJILYTICS workflow or RAG prototype",
    value=example_question,
    height=110,
)

if st.button("Run LangGraph Agent", type="primary"):
    with st.spinner("Running LangGraph workflow..."):
        result = agent.query(query)

    st.subheader("Detected Intent")
    if result.get("intent") == "on_topic":
        st.success(result.get("intent"))
    else:
        st.warning(result.get("intent"))

    st.subheader("Final Output")
    st.write(result.get("final_output", ""))

    if show_docs:
        st.subheader("Retrieved Documents from ChromaDB")
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

    if show_metadata:
        st.subheader("Metadata")
        st.json(result.get("metadata", {}))

st.divider()
st.markdown(
    """
### Workflow

`User Query → classify_intent → conditional route → rag_research if on-topic → format_output → final answer`

- **LangGraph** handles workflow orchestration and conditional routing.
- **ChromaDB** stores embedded AIJILYTICS documentation chunks.
- **RAG node** retrieves relevant chunks and synthesizes a grounded answer.
- **Off-topic queries** skip retrieval and receive a scope-limited response.
"""
)
