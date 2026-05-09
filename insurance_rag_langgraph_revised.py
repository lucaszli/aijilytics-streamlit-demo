"""
AIJILYTICS Insurance Claims RAG Agent using LangGraph + ChromaDB.
Streamlit-ready version.
"""

import os
import shutil
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
    def __init__(
        self,
        persist_directory: Optional[str] = "./streamlit_chroma_db",
        collection_name: str = "aijilytics_claims_docs",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        retrieval_k: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set.")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.graph = self._build_graph()

    def _sample_documents(self) -> List[Document]:
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
        if reset and self.persist_directory and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)

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
            "metadata": {**state.get("metadata", {}), "intent_detection": "completed"},
        }

    def route_after_intent(self, state: GraphState) -> str:
        return "format_output" if state["intent"] == "off_topic" else "rag_research"

    def rag_research(self, state: GraphState) -> GraphState:
        query = state["query"]
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_sample_insurance_data() first.")

        expansion_prompt = f"""
Generate 3 concise search queries for retrieving relevant AIJILYTICS documentation from a vector store.
Focus on insurance claims, broker workflows, compliance, risk assessment, negotiation support, ChromaDB,
RAG, or LangGraph when relevant.

User question:
{query}

Return exactly 3 lines. No numbering.
"""
        expansion_response = self.llm.invoke(expansion_prompt)
        search_queries = [line.strip("-• 1234567890.").strip()
                          for line in expansion_response.content.splitlines()
                          if line.strip()][:3]
        search_queries = [query] + search_queries

        docs_by_key: Dict[str, Document] = {}
        for search_query in search_queries:
            docs = self.vectorstore.similarity_search(search_query, k=self.retrieval_k)
            for doc in docs:
                docs_by_key[doc.page_content[:250]] = doc

        retrieved_docs = list(docs_by_key.values())[: self.retrieval_k * 2]
        retrieved_payload = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]

        context = "\n\n".join(
            f"Source metadata: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        synthesis_prompt = f"""
You are a RAG assistant for the AIJILYTICS insurance claims processing prototype.

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
        return {
            **state,
            "retrieved_docs": retrieved_payload,
            "response": response.content,
            "metadata": {
                **state.get("metadata", {}),
                "rag_research": "completed",
                "search_queries": search_queries,
                "retrieved_doc_count": len(retrieved_payload),
            },
        }

    def format_output(self, state: GraphState) -> GraphState:
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
Rewrite the synthesized response into a clear, readable answer.

Formatting rules:
- Do NOT write Python code.
- Do NOT write Streamlit code.
- Do NOT include import statements.
- Do NOT use st.title, st.header, st.write, st.markdown, or any code-like syntax.
- Do NOT wrap the answer in code fences.
- Output only the final user-facing answer.
- Use plain English.
- Use short headings and bullet points where helpful.
- Keep the tone professional and user-friendly.
- If the answer is about compliance or claims, be cautious and avoid legal guarantees.
- If the answer is off-topic, briefly explain that it is outside the scope of this AIJILYTICS demo.

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
