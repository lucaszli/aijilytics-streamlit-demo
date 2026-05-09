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
    show_docs = st.checkbox("Show retrieved ChromaDB documents", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)

# Pick a person / user setting
with st.sidebar:
    person_id = st.selectbox(
        "Select user profile",
        ["Customer", "Broker", "Underwriter", "Admin Demo"],
    )

# Create separate chat histories for each person
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if person_id not in st.session_state.chat_histories:
    st.session_state.chat_histories[person_id] = []

messages = st.session_state.chat_histories[person_id]

st.subheader(f"Chat Memory: {person_id}")

# Optional: button to clear this person's memory
if st.sidebar.button(f"Clear {person_id} chat"):
    st.session_state.chat_histories[person_id] = []
    st.rerun()

# Show previous messages for this selected person
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_message = st.chat_input("Ask a question about the AIJILYTICS workflow or RAG prototype")

if user_message:
    # Save user message
    messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.markdown(user_message)

    # Build memory context from previous messages
    recent_history = messages[-6:]  # last 6 messages to avoid huge prompts
    memory_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in recent_history
    )

    memory_augmented_query = f"""
Conversation history for this user profile:
{memory_context}

Current user question:
{user_message}
"""

    with st.chat_message("assistant"):
        with st.spinner("Running LangGraph workflow..."):
            result = agent.query(memory_augmented_query)

        final_answer = result.get("final_output", "")
        st.markdown(final_answer)

        with st.expander("Debug details"):
            st.write("Detected intent:", result.get("intent", ""))
            st.write("Metadata:", result.get("metadata", {}))

            docs = result.get("retrieved_docs", [])
            if not docs:
                st.info("No documents retrieved. This may be expected for off-topic queries.")
            else:
                for i, doc in enumerate(docs, 1):
                    st.write(f"Retrieved Document {i}")
                    if isinstance(doc, dict):
                        st.write(doc.get("content", ""))
                        st.json(doc.get("metadata", {}))
                    else:
                        st.write(str(doc))

    # Save assistant response
    messages.append({"role": "assistant", "content": final_answer})

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
