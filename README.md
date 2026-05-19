# AIJILYTICS Streamlit RAG Demo

This Streamlit app demos a LangGraph + ChromaDB + RAG workflow with four comparable RAG tabs.

## Tabs

1. **Original RAG Agent**  
   Baseline retrieval: one user query → one ChromaDB similarity search → synthesis → formatting.

2. **Multi-Query RAG Agent**  
   Generates multiple retrieval queries, retrieves broader context from ChromaDB, deduplicates results, then synthesizes.

3. **Corrective RAG Agent**  
   Retrieves context, asks an LLM to judge whether the context is sufficient, and retries retrieval with improved queries when needed.

4. **Hybrid Exact-Match RAG Agent**  
   Uses exact keyword/phrase matching over policy, certificate, claim-form, and discharge-voucher excerpts, plus vector fallback. This tab preserves exact document language word-for-word.

## Exact documentation included

The knowledge base includes both:
- AIJILYTICS workflow summary documents
- Exact excerpts from uploaded insurance documentation, including:
  - `POLICY DOC - INFINITY MFB (MC210000052LA).pdf`
  - `MOTOR CLAIM FORM (NEW) (1).pdf`
  - `MOTOR CERT. INFINITY.pdf`
  - `DV FOR IK25C000050FR.pdf`
  - `Claim Form Digital Fire.pdf`

All RAG agents retrieve from this combined document base. The Hybrid Exact-Match tab is the only tab designed to quote the exact documentation excerpts word-for-word.

## Files

- `streamlit_app.py` — Streamlit UI with four tabs
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

## Suggested tests

- What does the policy say about excess?
- What is the claims notification clause?
- What fields are in the motor accident report form?
- What does the motor certificate say about limitations as to use?
- What does the discharge voucher say about the settlement amount?
- What does the fire claim form say about admission of liability?
