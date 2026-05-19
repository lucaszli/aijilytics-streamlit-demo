"""
AIJILYTICS Insurance Claims RAG Agent using LangGraph + ChromaDB.
Streamlit-ready version with four RAG modes:
- original
- multi_query
- corrective
- hybrid_exact
"""

import os
import re
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
        rag_mode: str = "original",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set.")

        allowed_modes = {"original", "multi_query", "corrective", "hybrid_exact"}
        if rag_mode not in allowed_modes:
            raise ValueError(f"rag_mode must be one of {allowed_modes}")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.rag_mode = rag_mode
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.graph = self._build_graph()

    def _workflow_summary_documents(self) -> List[Document]:
        """AIJILYTICS product/workflow summaries."""
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
                metadata={"source": "AIJILYTICS PRD", "category": "broker_onboarding", "doc_kind": "summary"},
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
                metadata={"source": "AIJILYTICS PRD", "category": "claim_initiation", "doc_kind": "summary"},
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
                metadata={"source": "AIJILYTICS PRD", "category": "risk_assessment", "doc_kind": "summary"},
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
                metadata={"source": "AIJILYTICS PRD", "category": "negotiation", "doc_kind": "summary"},
            ),
            Document(
                page_content=(
                    "Case Preparation, Negotiation & Closure: The Customer Claim Record is the live system "
                    "of record. The business-facing AI agent prepares a complete Prepared Case File by "
                    "consolidating documents, summaries, appraisals, and negotiation context. Once settlement "
                    "is agreed, the final amount is recorded, the discharge voucher is generated, customer "
                    "acceptance is captured, payment processing is confirmed, and the claim is marked completed."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "case_closure", "doc_kind": "summary"},
            ),
            Document(
                page_content=(
                    "NIIRA 2025 / Compliance: The Nigerian Insurance Industry Reform Act 2025 states that "
                    "admitted claims must be settled within 60 days of notification, except for special-risk "
                    "cases. The platform helps brokers operate in a time-bound, traceable, and compliant "
                    "environment. Claim decisions, documentation, communications, and pricing decisions should "
                    "be traceable and audit-ready."
                ),
                metadata={"source": "AIJILYTICS PRD", "category": "compliance", "doc_kind": "summary"},
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
                metadata={"source": "Prototype Documentation", "category": "technical_architecture", "doc_kind": "summary"},
            ),
        ]

    def _exact_document_excerpts(self) -> List[Document]:
        """
        Exact excerpts from the uploaded policy/certificate/forms.
        These are added to every agent's vector store.
        The hybrid_exact mode uses them for exact-match, word-for-word output.
        """
        return [
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Policy Details: "
                    "THE POLICY HOLDER: Infinity Micro Finance Bank Limited. "
                    "POLICY NUMBER: MC210000052LA. "
                    "BROKER/AGENT: MIP INSURANCE BROKERS LIMITED. "
                    "PRODUCT: MOTOR CYCLE. "
                    "DATE OF ISSUE: 22-03-2021. "
                    "COVER FROM: 19-03-2021. "
                    "COVER TO: 18-03-2022. "
                    "RENEWAL DATE: 19-03-2022. "
                    "SUM INSURED: NGN400,000. "
                    "PREMIUM: NGN20,000."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 2, "category": "policy_details", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Limits of Liability and Excess: "
                    "Limit of the amount of the company’s liability under Section I-1(ii) in respect of any "
                    "one claim or series of claims arising out of one event 250,000.00. "
                    "Limit of the amount of the company’s liability under Section I-1(i) Unlimited but reasonable. "
                    "EXCESS N10,000 OR 10% OF CLAIM WHICHEVER IS HIGHER."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 3, "category": "limits_excess", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Section I – Liability to Third Parties: "
                    "Subject to the Limits of Liability the company will indemnify the insured in the against all sums "
                    "including claimant’s cost and expenses which the insured shall become legally liable to pay in respect of: "
                    "(i) death of or bodily injury to any person caused by or arising out of the use "
                    "(including the loading and/or unloading) of the Motor Vehicle. "
                    "(ii) damage to property caused by the use (including the loading and/or unloading) of the Motor Vehicle."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 4, "category": "third_party_liability", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Section II – Own Damage: "
                    "The Company will indemnify the insured against loss of or damage to the Motor Car and/or its accessories "
                    "whilst thereon (a) by accidental collision or overturning consequent upon mechanical breakdown or consequent "
                    "upon wear and tear. (b) by fire, external explosion, self ignition or lighting or burglary housebreaking or theft. "
                    "(c) by malicious act. (d) whilst in transit (including the process of loading and unloading incidental to such transit) "
                    "by road, rail, inland water way, lift or elevator."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 5, "category": "own_damage", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Conditions - Notice of Accident or Claim: "
                    "Notice shall be given in writing to the Company immediately upon the occurrence of any accident "
                    "or loss or damage and in the event of any claim. Every letter claim writ summons and/or process "
                    "shall be forwarded to the Company immediately on receipt by the Insured. "
                    "In case of theft or other criminal act which may be the subject of a claim under this Policy the "
                    "Insured shall give immediate notice to the Police and co-operate with the Company in securing the conviction of the offender."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 8, "category": "claim_notice_conditions", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Claims Notification Clause: "
                    "It is hereby declared and agreed that notwithstanding anything contained herein to the contrary "
                    "that the Company shall be under no liability whatsoever in respect of any accident/loss resulting "
                    "in claim reported after \"30 days\" of the occurrence of such accident/loss."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 10, "category": "claims_notification_clause", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | No Premium No Cover Clause: "
                    "Notwithstanding anything contained herein to the contrary, it is hereby declared and agreed that any "
                    "reference either in the Recital or Operative Clause or anywhere else on the Policy or any of the conditions "
                    "attaching thereto, to the insured agreeing to pay premium is deemed to be reworded as \"The Insured Having Paid\" the premium."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 10, "category": "npnc_clause", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Constructive Total Loss Settlement Clause: "
                    "It is hereby declared and agreed that if the submitted estimate of repairs in respect of the insured vehicles "
                    "shall exceed 60% of the insured value the company shall be entitled to treat the claims as a TOTAL LOSS."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 11, "category": "constructive_total_loss", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "POLICY DOC - INFINITY MFB (MC210000052LA) | Anti-Theft/Tracking Devices: "
                    "the cover granted by this policy in respect of the vehicle insured herein shall only be operative subject to the following conditions: "
                    "(I) That the insured vehicle shall be fitted with an Immobilizer or a Tracking Device "
                    "(ii) That the evidence of such installations shall be produced at the time of obtaining insurance and/or at the time of a claim as a result of theft. "
                    "(iii) That the devices shall be put in operation at all times when the vehicle is not in use."
                ),
                metadata={"source": "POLICY DOC - INFINITY MFB (MC210000052LA).pdf", "page": 11, "category": "anti_theft_tracking", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CLAIM FORM (NEW) (1) | Motor Accident Report Form fields include: "
                    "VEHICLE INSURED PARTICULARS: Make, Reg. No., C.C., Year of make, Eng. No., Chasis No., Mileage covered, Purpose being used. "
                    "DRIVER AT THE TIME OF ACCIDENT: Name, Age, Address, Driving License No., category, endorsement, Date of Issue, Date of Expiry, Place of Issue, Learners’ Permit, Relation of Driver to insured. "
                    "PARTICULAR OF ACCIDENT: Date, Time, Exact Location of Accident, Road Condition, Weather Condition, Speed of your Vehicle, Condition of brakes, Address of Police Station Accident was reported."
                ),
                metadata={"source": "MOTOR CLAIM FORM (NEW) (1).pdf", "page": 1, "category": "motor_claim_form_fields", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CLAIM FORM (NEW) (1) | Accident and Third Party Details: "
                    "FULL DESCRIPTION OF ACCIDENT: Full statement of the Driver may be on a separate sheet. "
                    "SKETCH: Please show point of impact and position of vehicles and person concerned at the time of accident; indicate by arrow which direction they were traveling. "
                    "DAMAGE TO INSURED VEHICLE: Full details of Damaged Part, Present Location of Vehicle, Rough Estimate Of repair, Repairer’s Name and Address, Inventory of damaged part. "
                    "IF ANY WRITTEN COMMUNICATION IS RECEIVED, PLEASE FORWARD IT IMMEDIATELY UNANSWERED."
                ),
                metadata={"source": "MOTOR CLAIM FORM (NEW) (1).pdf", "page": 2, "category": "motor_claim_accident_details", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CLAIM FORM (NEW) (1) | Declaration: "
                    "I/We declare the foregoing particulars to be true and I/ We here authorize TANGERINE GENERAL INSURANCE LTD "
                    "and / or their legal representatives to deal with all matters arising from this accident at their discretion "
                    "and if they deem it expedient to admit liability and / or negligence on the part of myself / our servants or agents."
                ),
                metadata={"source": "MOTOR CLAIM FORM (NEW) (1).pdf", "page": 3, "category": "motor_claim_declaration", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CLAIM FORM (NEW) (1) | Completion instruction: "
                    "PLEASE MAKE SURE THAT ALL QUESTIONS HAVE BEEN ANSWERED. "
                    "( The Company does not admit liability by the issue of this form)"
                ),
                metadata={"source": "MOTOR CLAIM FORM (NEW) (1).pdf", "page": 4, "category": "motor_claim_completion_instruction", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CERT. INFINITY | Certificate of Insurance: "
                    "MOTOR VEHICLES (THIRD PARTY INSURANCE) ACT 1945 (NIGERIA). "
                    "Certificate No: 26/60941/UNI WAX 4. Product PRIVATE MOTOR. Policy No IKJPM007295/UNI. "
                    "Registration No EPE432GB. Make of Vehicle TOYOTA - Corolla. "
                    "Name of Policy Holder INFINITY MICROFINANCE BANK LIMITED. "
                    "Effective date of commencenment of insurance for the purpose of the Ordinance(s) January 07, 2026. "
                    "Date of expiry of insurance January 06, 2027."
                ),
                metadata={"source": "MOTOR CERT. INFINITY.pdf", "page": 1, "category": "motor_certificate", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CERT. INFINITY | Persons or classes of persons entitled to drive: "
                    "Any person who is driving on the policy holder’s order or with his permission. "
                    "Provided that the person driving is permitted in accordance with the licensing or other laws or regulations "
                    "to drive the Motor Vehicle or has been so permitted and is not disqualified by order of a Court of Law "
                    "or by reason of any enactment or regulation in that behalf from driving such Motor Vehicle."
                ),
                metadata={"source": "MOTOR CERT. INFINITY.pdf", "page": 1, "category": "persons_entitled_to_drive", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "MOTOR CERT. INFINITY | Limitations as to use: "
                    "Use only for social domestic and pleasure purposes and for the policy holder’s business. "
                    "The Policy does not cover use for hire or reward racing pace-making reliability trail speed-testing "
                    "or use for any purpose in connection with the Motor Trade."
                ),
                metadata={"source": "MOTOR CERT. INFINITY.pdf", "page": 1, "category": "limitations_as_to_use", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "DV FOR IK25C000050FR | Discharge/Acceptance Form: "
                    "CLAIM NUMBER: IK25C000050FR POLICY NUMBER: CP190000090LA. "
                    "Received this 17-DEC-25 from ........................................................................................ the sum of NGN 4,066,666.67 "
                    "(Four Million Sixty-Six Thousand Six Hundred Sixty-Six and Sixty-Seven Naira Only) in full discharge of all claims that [INSURED] "
                    "may have against the company under policy number CP190000090LA in respect of INFINITY MFB /CHIBUEKE OBIOMA "
                    "FIRE CLAIM MANDILLAS which occurred on 16th September 2025 as a result of [FIRE]."
                ),
                metadata={"source": "DV FOR IK25C000050FR.pdf", "page": 1, "category": "discharge_voucher", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "DV FOR IK25C000050FR | Payment options and claimant account details: "
                    "Please note that we shall not be responsible for delay or wrongful credit to account number incorrectly stated or presented "
                    "by you and shall not be required to confirm the correctness or otherwise of the account stated below. "
                    "CLAIMANT'S ACCOUNT DETAILS: Account Name, Bank Name, Bank Branch, Account No, Bank Sort Code."
                ),
                metadata={"source": "DV FOR IK25C000050FR.pdf", "page": 1, "category": "dv_payment_details", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "Claim Form Digital Fire | Fire Claims Form: "
                    "POLICY NUMBER. Note: The form can be completed for a fire claim under Householders/Houseowners policy. "
                    "Please refer to INSTRUCTIONS on the back of this form when preparing your claim. "
                    "The form includes declaration fields for the insured, address, policy, time/date of fire, fire/flood/other cause, "
                    "location of occurrence, cause, insurance item number, value immediately before the fire, amount of damage sustained, "
                    "statement of insurance, amount claimed, signature of claimant(s)."
                ),
                metadata={"source": "Claim Form Digital Fire.pdf", "page": 1, "category": "fire_claim_form", "doc_kind": "exact"},
            ),
            Document(
                page_content=(
                    "Claim Form Digital Fire | Liability note: "
                    "NB: The forwarding of this Form for completion does not constitute admission of liability."
                ),
                metadata={"source": "Claim Form Digital Fire.pdf", "page": 1, "category": "fire_claim_liability_note", "doc_kind": "exact"},
            ),
        ]

    def _sample_documents(self) -> List[Document]:
        """All documents available to every RAG mode."""
        return self._workflow_summary_documents() + self._exact_document_excerpts()

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

- on_topic: The query is related to AIJILYTICS, Nigerian insurance claims processing, broker onboarding,
customer claim initiation, risk assessment, pricing support, compliance, NIIRA/NAICOM,
broker-underwriter negotiation, case closure, exact policy documentation, claim forms, discharge vouchers,
motor certificates, LangGraph, RAG, ChromaDB, vector stores, retrieval, multi-query RAG, corrective RAG,
hybrid RAG, exact-match retrieval, or this prototype's technical architecture.

- off_topic: The query is unrelated to this prototype. This includes U.S. car insurance,
insurance rules outside Nigeria/AIJILYTICS, unrelated finance/investing, food, school,
health insurance recommendations, travel insurance recommendations, entertainment,
or general knowledge not connected to AIJILYTICS.

Rules:
- If the query asks about this system, AIJILYTICS, LangGraph, RAG, ChromaDB, vector stores, retrieval,
  corrective RAG, multi-query RAG, hybrid RAG, exact policy documents, claim forms, motor certificates,
  discharge vouchers, or the prototype's architecture, classify as on_topic.
- If the query asks about claims, broker onboarding, compliance, risk assessment, negotiation, case closure,
  or Nigerian insurance workflows, classify as on_topic.
- If the query asks about U.S. car insurance or insurance rules outside Nigeria/AIJILYTICS, classify as off_topic.
- If the query asks for stock/investing advice, food, entertainment, schoolwork, or unrelated general knowledge, classify as off_topic.
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
            "metadata": {**state.get("metadata", {}), "intent_detection": "completed", "rag_mode": self.rag_mode},
        }

    def route_after_intent(self, state: GraphState) -> str:
        return "format_output" if state["intent"] == "off_topic" else "rag_research"

    def _documents_to_payload(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def _dedupe_docs(self, docs: List[Document], limit: Optional[int] = None) -> List[Document]:
        docs_by_key: Dict[str, Document] = {}
        for doc in docs:
            key = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page', '')}|{doc.page_content[:180]}"
            docs_by_key[key] = doc
        deduped = list(docs_by_key.values())
        return deduped[:limit] if limit else deduped

    def _single_query_retrieval(self, query: str, k: Optional[int] = None) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_sample_insurance_data() first.")
        return self.vectorstore.similarity_search(query, k=k or self.retrieval_k)

    def _generate_multi_queries(self, query: str) -> List[str]:
        expansion_prompt = f"""
Generate 3 concise search queries for retrieving relevant AIJILYTICS insurance documentation from a vector store.
Focus on claim forms, motor policy clauses, discharge vouchers, certificates, broker workflows, compliance,
risk assessment, negotiation support, ChromaDB, RAG, or LangGraph when relevant.

User question:
{query}

Return exactly 3 lines. No numbering.
"""
        expansion_response = self.llm.invoke(expansion_prompt)
        generated = [
            line.strip("-• 1234567890.").strip()
            for line in expansion_response.content.splitlines()
            if line.strip()
        ][:3]
        return [query] + generated

    def _multi_query_retrieval(self, query: str) -> tuple[List[Document], List[str]]:
        search_queries = self._generate_multi_queries(query)
        all_docs: List[Document] = []
        for search_query in search_queries:
            all_docs.extend(self._single_query_retrieval(search_query, k=self.retrieval_k))
        return self._dedupe_docs(all_docs, limit=self.retrieval_k * 3), search_queries

    def _judge_context_sufficiency(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        context = "\n\n".join(doc.page_content for doc in docs)
        judge_prompt = f"""
You are evaluating whether retrieved context is sufficient to answer a user question.

User question:
{query}

Retrieved context:
{context}

Return exactly this format:
SUFFICIENT: yes or no
REASON: one short sentence
IMPROVED QUERIES:
- query 1
- query 2
- query 3
"""
        judge_response = self.llm.invoke(judge_prompt).content
        sufficient = "sufficient: yes" in judge_response.lower()
        improved_queries = []
        for line in judge_response.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                improved_queries.append(stripped.lstrip("-").strip())
        return {
            "sufficient": sufficient,
            "raw_judgment": judge_response,
            "improved_queries": improved_queries[:3],
        }

    def _corrective_retrieval(self, query: str) -> tuple[List[Document], List[str], Dict[str, Any]]:
        initial_docs = self._single_query_retrieval(query, k=self.retrieval_k)
        judgment = self._judge_context_sufficiency(query, initial_docs)

        search_queries = [query]
        all_docs = list(initial_docs)

        if not judgment["sufficient"] and judgment["improved_queries"]:
            for improved_query in judgment["improved_queries"]:
                search_queries.append(improved_query)
                all_docs.extend(self._single_query_retrieval(improved_query, k=self.retrieval_k))

        return self._dedupe_docs(all_docs, limit=self.retrieval_k * 3), search_queries, judgment

    def _extract_keywords(self, query: str) -> List[str]:
        query_lower = query.lower()
        known_phrases = [
            "policy number", "claim number", "certificate", "excess", "30 days", "claims notification",
            "no premium no cover", "constructive total loss", "total loss", "tracking device", "anti-theft",
            "discharge voucher", "claimant account", "motor accident report", "driving license",
            "third party", "own damage", "fire claim", "admission of liability", "limitations as to use",
            "persons entitled to drive", "sum insured", "premium", "infinity", "tangerine", "leadway",
            "unitrust", "toyota", "corolla", "ngn", "repair estimate", "police station"
        ]
        matches = [phrase for phrase in known_phrases if phrase in query_lower]
        tokens = re.findall(r"[a-zA-Z0-9]+", query_lower)
        tokens = [t for t in tokens if len(t) >= 4]
        return matches + tokens

    def _hybrid_exact_retrieval(self, query: str) -> tuple[List[Document], List[str]]:
        """
        Hybrid approach:
        1. Exact keyword/phrase matching over exact documentation excerpts.
        2. Add vector search fallback from ChromaDB.
        The final answer preserves exact excerpts word-for-word.
        """
        exact_docs = self._exact_document_excerpts()
        keywords = self._extract_keywords(query)
        scored: List[tuple[int, Document]] = []

        for doc in exact_docs:
            text = doc.page_content.lower()
            score = 0
            for kw in keywords:
                if kw.lower() in text:
                    score += 3 if " " in kw else 1
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        exact_matches = [doc for _, doc in scored[: self.retrieval_k]]

        vector_docs = self._single_query_retrieval(query, k=self.retrieval_k)
        combined = self._dedupe_docs(exact_matches + vector_docs, limit=self.retrieval_k * 2)

        return combined, keywords

    def _synthesize_from_docs(
        self,
        query: str,
        retrieved_docs: List[Document],
        search_queries: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        retrieved_payload = self._documents_to_payload(retrieved_docs)
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
- Use exact document details when they are retrieved, including source names where useful.
- Do not invent exact regulations, policy clauses, forms, limits, or pricing.
- If context is insufficient, say what is missing and give a cautious answer.
- AI outputs are advisory and should remain human-reviewable when discussing decisions.
"""
        response = self.llm.invoke(synthesis_prompt)
        metadata = {
            "rag_research": "completed",
            "rag_mode": self.rag_mode,
            "search_queries": search_queries,
            "retrieved_doc_count": len(retrieved_payload),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return {
            "query": query,
            "intent": "on_topic",
            "retrieved_docs": retrieved_payload,
            "response": response.content,
            "final_output": "",
            "metadata": metadata,
        }

    def _hybrid_exact_answer(self, query: str, retrieved_docs: List[Document], keywords: List[str]) -> str:
        if not retrieved_docs:
            return (
                "### Summary\n\n"
                "I could not find a strong exact documentation match for this query.\n\n"
                "### Exact-Match Notes\n\n"
                "- Try asking with an exact term such as `excess`, `policy number`, `claims notification`, "
                "`discharge voucher`, `motor accident report`, or `certificate`.\n\n"
                "### Next Steps\n\n"
                "Use the Original, Multi-Query, or Corrective RAG tabs for broader semantic research."
            )

        lines = [
            "### Summary",
            "",
            "The Hybrid Exact-Match RAG agent found documentation excerpts that match exact terms in the question. "
            "Unlike the other RAG tabs, this tab preserves the retrieved policy/form language word-for-word below.",
            "",
            "### Exact Documentation Excerpts",
            "",
        ]

        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "N/A")
            category = doc.metadata.get("category", "uncategorized")
            lines.extend([
                f"**Match {i}: {source} | Page {page} | {category}**",
                "",
                f"> {doc.page_content}",
                "",
            ])

        lines.extend([
            "### Search Terms Used",
            "",
            ", ".join(keywords[:12]) if keywords else "No strong exact keywords detected.",
            "",
            "### Next Steps",
            "",
            "Use this tab when the team needs exact policy, certificate, claim-form, or discharge-voucher language. "
            "For broader research or synthesis, compare the result with the Original, Multi-Query, and Corrective RAG tabs.",
        ])
        return "\n".join(lines)

    def rag_research(self, state: GraphState) -> GraphState:
        query = state["query"]

        if self.rag_mode == "original":
            retrieved_docs = self._single_query_retrieval(query, k=self.retrieval_k)
            result = self._synthesize_from_docs(query, retrieved_docs, [query])
        elif self.rag_mode == "multi_query":
            retrieved_docs, search_queries = self._multi_query_retrieval(query)
            result = self._synthesize_from_docs(query, retrieved_docs, search_queries)
        elif self.rag_mode == "corrective":
            retrieved_docs, search_queries, judgment = self._corrective_retrieval(query)
            result = self._synthesize_from_docs(
                query,
                retrieved_docs,
                search_queries,
                extra_metadata={
                    "corrective_judgment": judgment.get("raw_judgment", ""),
                    "retrieval_was_sufficient": judgment.get("sufficient", False),
                },
            )
        elif self.rag_mode == "hybrid_exact":
            retrieved_docs, keywords = self._hybrid_exact_retrieval(query)
            exact_output = self._hybrid_exact_answer(query, retrieved_docs, keywords)
            result = {
                **state,
                "retrieved_docs": self._documents_to_payload(retrieved_docs),
                "response": exact_output,
                "final_output": "",
                "metadata": {
                    **state.get("metadata", {}),
                    "rag_research": "completed",
                    "rag_mode": self.rag_mode,
                    "search_type": "keyword_exact_match_plus_vector_fallback",
                    "exact_keywords": keywords,
                    "retrieved_doc_count": len(retrieved_docs),
                    "preserve_word_for_word": True,
                },
            }
        else:
            raise ValueError(f"Unknown rag_mode: {self.rag_mode}")

        return {
            **state,
            "retrieved_docs": result["retrieved_docs"],
            "response": result["response"],
            "metadata": {**state.get("metadata", {}), **result["metadata"]},
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
                "exact policy documentation, claim forms, motor certificates, discharge vouchers, "
                "and the LangGraph/RAG/ChromaDB technical architecture.\n\n"
                "Example questions you can ask:\n"
                "- What does the policy say about excess?\n"
                "- What fields are in the motor accident report form?\n"
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

        # Preserve exact wording for hybrid mode. Do not reformat with the LLM.
        if state.get("metadata", {}).get("preserve_word_for_word"):
            return {
                **state,
                "final_output": response,
                "metadata": {**state.get("metadata", {}), "formatting": "preserved_exact_output"},
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
