# AIJILYTICS Streamlit RAG Demo

This Streamlit app compares three LangGraph + ChromaDB RAG architectures for the AIJILYTICS insurance claims workflow.

## Tabs

1. **Original RAG Agent**  
   Single-query baseline retrieval from ChromaDB.

2. **Multi-Query RAG Agent**  
   Generates multiple related retrieval queries and runs ChromaDB retrieval in parallel.

3. **Corrective RAG Agent**  
   Retrieves context, uses an LLM-based sufficiency check, and retries with multi-query retrieval if the first retrieval is weak.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Secrets

Add this in Streamlit Cloud secrets:

```toml
OPENAI_API_KEY = "your_openai_key_here"
```

Do not commit API keys to GitHub.
