"""
AIJILYTICS Insurance Claims RAG Agent using LangGraph + ChromaDB.
Streamlit-ready version with three RAG modes:
- original: one-query baseline RAG
- multi_query: query expansion + parallel ChromaDB retrieval
- corrective: retrieval quality check + retry when context is weak
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

load_dotenv()


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    final_output: str
    metadata: Dict[str, Any]


class InsuranceRAGAgent:
    """LangGraph RAG agent for the AIJILYTICS insurance claims demo."""

    VALID_RAG_MODES = {"original", "multi_query", "corrective"}

    def __init__(
        self,
        persist_directory: Optional[str] = "./streamlit_chroma_db",
        collection_name: str = "aijilytics_claims_docs",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        retrieval_k: int = 3,
        rag_mode: str = "original",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set.")

        if rag_mode not in self.VALID_RAG_MODES:
            raise ValueError(f"rag_mode must be one of {self.VALID_RAG_MODES}")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.rag_mode = rag_mode
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.graph = self._build_graph()

    def _sample_documents(self) -> List[Document]:
        """Small AIJILYTICS knowledge base used for the prototype."""
        return [
            Document(
                page_content=(
                    "Broker Onboarding / Setup: The broker organization registers and creates a secure "
                    "tenant workspace. Brand & Logo Setup configures branding. People Management invites "
                    "staff and assigns roles such as admin, manager, claims handler, and viewer. "
                    "Underwriter Onboarding defines underwriter information, claim categories, policy "
                    "documents, and standardized claim forms. Uploading policy and underwriting content "
                    "triggers Vector Store Creation so content can be indexed for the RAG agent."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "broker_onboarding"},
            ),
            Document(
                page_content=(
                    "Customer Claim Initiation / Inquiry: Customers initiate claims through the portal or "
                    "chatbot by filling a claims form, making inquiries, and submitting documents. They may "
                    "also report claims through phone, letter, or email. Offline communications are passed "
                    "through an interceptor. The AI agent retrieves relevant policy documents and document "
                    "lists from the vector store, presents the correct claim form, tracks submitted documents, "
                    "and validates uploaded documents for completeness and correctness where possible."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "claim_initiation"},
            ),
            Document(
                page_content=(
                    "Insurance Purchase, Risk Assessment & Pricing: The customer-facing AI agent presents "
                    "product information, required documents, coverage details, exclusions, and pricing "
                    "factors. The business-facing AI agent retrieves historical customer context, evaluates "
                    "exposure and risk, checks claim history, and generates a pricing recommendation and "
                    "risk assessment score. These outputs support broker decision-making but do not replace "
                    "final underwriting judgment. Pricing decisions should be logged for auditability."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "risk_assessment"},
            ),
            Document(
                page_content=(
                    "Broker Negotiation & Underwriter Support: The business-facing AI agent acts as a "
                    "broker-facing copilot. It retrieves policy documents, standardized document lists, and "
                    "underwriting references from the vector store. It grounds analysis in customer claim "
                    "context, including submitted documents, customer history, and claim details. It can "
                    "generate coverage briefs, appraisal summaries, claim exposure views, completeness "
                    "analyses, underwriter clarification lists, and negotiation drafts with factual grounding."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "negotiation"},
            ),
            Document(
                page_content=(
                    "Case Preparation, Negotiation & Closure: The Customer Claim Record is the live system "
                    "of record. The business-facing AI agent prepares a complete Prepared Case File by "
                    "consolidating documents, summaries, appraisals, and negotiation context. Once settlement "
                    "is agreed, the final amount is recorded, the discharge voucher is generated, customer "
                    "acceptance is captured, payment processing is confirmed, and the claim is marked completed."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "case_closure"},
            ),
            Document(
                page_content=(
                    "NIIRA 2025 / Compliance: The Nigerian Insurance Industry Reform Act 2025 states that "
                    "admitted claims must be settled within 60 days of notification, except for special-risk "
                    "cases. The platform helps brokers operate in a time-bound, traceable, and compliant "
                    "environment. Claim decisions, documentation, communications, and pricing decisions should "
                    "be traceable and audit-ready."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "compliance"},
            ),
            Document(
                page_content=(
                    "Technical Architecture: This prototype uses LangGraph for workflow orchestration, "
                    "ChromaDB as the vector store, OpenAI embeddings for semantic search, and a chat model "
                    "for classification, synthesis, and formatting. ChromaDB stores embedded chunks of "
                    "AIJILYTICS workflow documentation. During retrieval, the query is matched against the "
                    "vector store using similarity search, and the top relevant chunks are passed into the "
                    "RAG synthesis node."
                ),
                metadata={"source": "Prototype Documentation", "category": "technical_architecture"},
            ),
        ]

    def initialize_vectorstore(self, documents: List[Document], reset: bool = False) -> None:
        """Create or load a ChromaDB vector store."""
        if reset and self.persist_directory and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        # Reuse existing Chroma collection when possible to avoid duplicate chunks on app reruns.
        if not reset and self.persist_directory and os.path.exists(self.persist_directory):
            try:
                existing_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                )
                if existing_store._collection.count() > 0:
                    self.vectorstore = existing_store
                    print(f"Loaded existing vector store with {existing_store._collection.count()} chunks.")
                    return
            except Exception:
                pass

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        print(f"Vector store initialized with {len(chunks)} chunks.")

    def load_sample_insurance_data(self, reset: bool = False) -> None:
        self.initialize_vectorstore(self._sample_documents(), reset=reset)

    def classify_intent(self, state: GraphState) -> GraphState:
        query = state["query"]
        intent_prompt = f"""
Classify the user query as exactly one of these two categories:

- on_topic: Related to AIJILYTICS, Nigerian insurance claims processing, broker onboarding,
customer claim initiation, risk assessment, pricing support, compliance, NIIRA/NAICOM,
broker-underwriter negotiation, case closure, LangGraph, RAG, ChromaDB, vector stores,
or this prototype's technical architecture.

- off_topic: Unrelated to this prototype. This includes insurance in other countries, unrelated
finance/investing, food, school, health insurance recommendations, travel insurance recommendations,
entertainment, or general knowledge not connected to AIJILYTICS.

Rules:
- If the query asks about this system, LangGraph, RAG, ChromaDB, or vector stores, classify as on_topic.
- If the query asks about U.S. car insurance or insurance rules outside Nigeria/AIJILYTICS, classify as off_topic.
- If the query asks for stock/investing advice, classify as off_topic.
- Respond with only one word: on_topic or off_topic.

User Query:
{query}
"""
        response = self.llm.invoke(intent_prompt)
        intent = response.content.strip().lower()
        if intent not in {"on_topic", "off_topic"}:
            intent = "off_topic"
        return {
            **state,
            "intent": intent,
            "metadata": {
                **state.get("metadata", {}),
                "intent_detection": "completed",
                "rag_mode": self.rag_mode,
            },
        }

    def route_after_intent(self, state: GraphState) -> str:
        return "format_output" if state["intent"] == "off_topic" else "rag_research"

    def _generate_search_queries(self, query: str, n: int = 3) -> List[str]:
        """Generate alternative search queries for multi-query and corrective RAG."""
        expansion_prompt = f"""
Generate {n} concise search queries for retrieving relevant AIJILYTICS documentation from a vector store.
Focus on insurance claims, broker workflows, compliance, risk assessment, negotiation support, ChromaDB,
RAG, or LangGraph when relevant.

User question:
{query}

Return exactly {n} lines. No numbering.
"""
        expansion_response = self.llm.invoke(expansion_prompt)
        generated_queries = [
            line.strip("-• 1234567890.").strip()
            for line in expansion_response.content.splitlines()
            if line.strip()
        ][:n]
        return [q for q in generated_queries if q]

    def _retrieve_for_queries_parallel(self, search_queries: List[str]) -> List[Document]:
        """Retrieve docs from ChromaDB for multiple queries in parallel and deduplicate chunks."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_sample_insurance_data() first.")

        docs_by_key: Dict[str, Document] = {}

        def search_one(search_query: str) -> List[Document]:
            return self.vectorstore.similarity_search(search_query, k=self.retrieval_k)

        max_workers = min(len(search_queries), 4) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(search_one, q): q for q in search_queries}
            for future in as_completed(futures):
                try:
                    docs = future.result()
                except Exception:
                    docs = []
                for doc in docs:
                    docs_by_key[doc.page_content[:250]] = doc

        return list(docs_by_key.values())

    def _retrieve_original(self, query: str) -> Dict[str, Any]:
        docs = self.vectorstore.similarity_search(query, k=self.retrieval_k)
        return {
            "docs": docs,
            "search_queries": [query],
            "retrieval_strategy": "single_query_similarity_search",
        }

    def _retrieve_multi_query(self, query: str) -> Dict[str, Any]:
        generated_queries = self._generate_search_queries(query, n=3)
        search_queries = [query] + generated_queries
        docs = self._retrieve_for_queries_parallel(search_queries)
        return {
            "docs": docs[: self.retrieval_k * 2],
            "search_queries": search_queries,
            "retrieval_strategy": "multi_query_parallel_similarity_search",
        }

    def _judge_context_quality(self, query: str, docs: List[Document]) -> Dict[str, str]:
        """Use the LLM as a lightweight judge for retrieved context sufficiency."""
        if not docs:
            return {"score": "insufficient", "reason": "No documents were retrieved."}

        context_preview = "\n\n".join(doc.page_content[:700] for doc in docs)
        judge_prompt = f"""
You are evaluating whether retrieved context is sufficient for answering a user's question.

User question:
{query}

Retrieved context:
{context_preview}

Return exactly two lines:
score: sufficient or insufficient
reason: short reason
"""
        judge_response = self.llm.invoke(judge_prompt).content.strip()
        score = "insufficient"
        reason = judge_response
        for line in judge_response.splitlines():
            lower = line.lower()
            if lower.startswith("score:"):
                value = lower.split(":", 1)[1].strip()
                if "sufficient" in value and "insufficient" not in value:
                    score = "sufficient"
                else:
                    score = "insufficient"
            elif lower.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        return {"score": score, "reason": reason}

    def _retrieve_corrective(self, query: str) -> Dict[str, Any]:
        """Retrieve, judge context quality, and retry with query expansion if weak."""
        initial_docs = self.vectorstore.similarity_search(query, k=self.retrieval_k)
        quality = self._judge_context_quality(query, initial_docs)

        if quality["score"] == "sufficient":
            return {
                "docs": initial_docs,
                "search_queries": [query],
                "retrieval_strategy": "corrective_initial_retrieval_sufficient",
                "retrieval_quality": quality,
                "correction_applied": False,
            }

        generated_queries = self._generate_search_queries(query, n=3)
        search_queries = [query] + generated_queries
        corrected_docs = self._retrieve_for_queries_parallel(search_queries)
        corrected_quality = self._judge_context_quality(query, corrected_docs[: self.retrieval_k * 2])

        return {
            "docs": corrected_docs[: self.retrieval_k * 2],
            "search_queries": search_queries,
            "retrieval_strategy": "corrective_retry_with_multi_query_retrieval",
            "retrieval_quality": corrected_quality,
            "initial_retrieval_quality": quality,
            "correction_applied": True,
        }

    def rag_research(self, state: GraphState) -> GraphState:
        query = state["query"]
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_sample_insurance_data() first.")

        if self.rag_mode == "original":
            retrieval_result = self._retrieve_original(query)
        elif self.rag_mode == "multi_query":
            retrieval_result = self._retrieve_multi_query(query)
        elif self.rag_mode == "corrective":
            retrieval_result = self._retrieve_corrective(query)
        else:
            raise ValueError(f"Unsupported rag_mode: {self.rag_mode}")

        retrieved_docs = retrieval_result["docs"]
        retrieved_payload = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in retrieved_docs
        ]

        context = "\n\n".join(
            f"Source metadata: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        synthesis_prompt = f"""
You are a RAG assistant for the AIJILYTICS insurance claims processing prototype.

RAG mode:
{self.rag_mode}

User question:
{query}

Retrieved context from ChromaDB:
{context}

Write a grounded answer using only the retrieved context where possible.
Guidelines:
- Explain the answer in clear business/product language.
- If technical, explain how the technical component supports the workflow.
- Do not invent exact regulations, policy clauses, or pricing.
- If context is insufficient, say what is missing and give a cautious answer.
- AI outputs are advisory and should remain human-reviewable when discussing decisions.
"""
        response = self.llm.invoke(synthesis_prompt)

        metadata = {
            **state.get("metadata", {}),
            "rag_research": "completed",
            "rag_mode": self.rag_mode,
            "retrieval_strategy": retrieval_result.get("retrieval_strategy"),
            "search_queries": retrieval_result.get("search_queries", []),
            "retrieved_doc_count": len(retrieved_payload),
        }
        if "retrieval_quality" in retrieval_result:
            metadata["retrieval_quality"] = retrieval_result["retrieval_quality"]
        if "initial_retrieval_quality" in retrieval_result:
            metadata["initial_retrieval_quality"] = retrieval_result["initial_retrieval_quality"]
        if "correction_applied" in retrieval_result:
            metadata["correction_applied"] = retrieval_result["correction_applied"]

        return {
            **state,
            "retrieved_docs": retrieved_payload,
            "response": response.content,
            "metadata": metadata,
        }

    def format_output(self, state: GraphState) -> GraphState:
        query = state["query"]
        intent = state["intent"]
        response = state["response"]

        if state["intent"] == "off_topic":
            final_output = (
                "This question is outside the scope of this AIJILYTICS demo.\n\n"
                "This prototype answers questions about the AIJILYTICS Nigerian insurance claims workflow, "
                "broker onboarding, risk assessment, compliance, broker-underwriter negotiation, case closure, "
                "and the LangGraph/RAG/ChromaDB technical architecture.\n\n"
                "Example questions you can ask:\n"
                "- How does a customer initiate a claim in AIJILYTICS?\n"
                "- What are the NIIRA 2025 compliance requirements?\n"
                "- What is the purpose of ChromaDB in this RAG prototype?"
            )
            return {
                **state,
                "retrieved_docs": [],
                "response": "",
                "final_output": final_output,
                "metadata": {
                    **state.get("metadata", {}),
                    "formatting": "completed_off_topic",
                    "rag_research": "skipped_by_conditional_route",
                },
            }

        format_prompt = f"""
You are formatting the final response for a non-technical user in a Streamlit web app.

Original User Query:
{query}

Detected Intent:
{intent}

Synthesized Response:
{response}

Your task:
Rewrite the synthesized response into a clear, readable final answer for the user.

Strict formatting rules:
- Do NOT write Python code.
- Do NOT write Streamlit code.
- Do NOT include import statements.
- Do NOT use st.title, st.header, st.write, st.markdown, or any code-like syntax.
- Do NOT wrap the answer in code fences.
- Output only the final user-facing answer.
- Use normal Markdown only, such as short headings and bullet points.
- Use plain English.
- Keep the tone professional and user-friendly.
- If the answer is about compliance or claims, be cautious and avoid legal guarantees.
- If the answer is off-topic, briefly explain that it is outside the scope of this AIJILYTICS demo.

Preferred structure:
### Summary
Give a short 2-3 sentence summary.

### Key Points
Use bullet points for the main answer.

### Next Steps
Give practical next steps if relevant.

Final user-facing answer:
"""
        formatted_response = self.llm.invoke(format_prompt)
        final_output = formatted_response.content
        return {
            **state,
            "final_output": final_output,
            "metadata": {**state.get("metadata", {}), "formatting": "completed"},
        }

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("rag_research", self.rag_research)
        graph.add_node("format_output", self.format_output)
        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {"rag_research": "rag_research", "format_output": "format_output"},
        )
        graph.add_edge("rag_research", "format_output")
        graph.add_edge("format_output", END)
        return graph.compile()

    def query(self, user_query: str) -> GraphState:
        initial_state: GraphState = {
            "query": user_query,
            "intent": "",
            "retrieved_docs": [],
            "response": "",
            "final_output": "",
            "metadata": {},
        }
        return self.graph.invoke(initial_state)

    def stream_query(self, user_query: str):
        initial_state: GraphState = {
            "query": user_query,
            "intent": "",
            "retrieved_docs": [],
            "response": "",
            "final_output": "",
            "metadata": {},
        }
        return self.graph.stream(initial_state)
