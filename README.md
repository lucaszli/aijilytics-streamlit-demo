# AIJILYTICS Streamlit RAG Demo

This Streamlit app demos a simple LangGraph + ChromaDB + RAG workflow.

## Files

- `streamlit_app.py` — Streamlit UI
- `insurance_rag_langgraph_revised.py` — LangGraph/RAG/ChromaDB backend
- `requirements.txt` — dependencies
- `.gitignore` — prevents secrets and local Chroma folders from being committed

## Local setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

Run:

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud

Push these files to GitHub, deploy `streamlit_app.py`, and add `OPENAI_API_KEY` in Streamlit secrets.
