"""Agentic RAG Pipeline with query routing, document grading, and query rewriting.

This pipeline uses LangGraph to implement a sophisticated workflow that:
1. Routes queries - decides if retrieval is needed
2. Retrieves documents - uses the same Qdrant setup as VietnameseEmbeddingPipeline
3. Grades relevance - filters out irrelevant documents
4. Rewrites queries - up to 3 retries if no relevant docs found
5. Generates response - with or without context based on routing
"""

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient

from core.config import config
from services.adapters import (
    VietnameseDenseAdapter,
    VietnameseEmbedding,
    VietnameseSparseAdapter,
)
from services.pipelines.base import RAGPipeline
from services.rerankers import BaseReranker, CrossEncoderReranker

logger = logging.getLogger(__name__)

# Map string mode to langchain_qdrant RetrievalMode enum
_RETRIEVAL_MODE_MAP = {
    "hybrid": RetrievalMode.HYBRID,
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
}

# Maximum number of query rewrite attempts
MAX_REWRITE_ATTEMPTS = 3


class AgenticState(TypedDict):
    """State for the agentic RAG workflow."""

    query: str  # Original user query
    current_query: str  # May be rewritten for better retrieval
    history: List[BaseMessage]  # Chat history
    documents: List[Document]  # All retrieved documents
    relevant_documents: List[Document]  # Filtered by grading
    rewrite_count: int  # Track retry attempts
    needs_retrieval: bool  # Route decision result
    answer: str  # Final response
    sources: List[Dict[str, Any]]  # Document metadata


class AgenticRAGPipeline(RAGPipeline):
    """
    Agentic RAG pipeline with intelligent query routing and document grading.

    This pipeline extends the standard retrieval process with:
    - Query routing: Decides if retrieval is needed for the query
    - Document grading: Evaluates relevance of retrieved documents
    - Query rewriting: Reformulates queries when no relevant docs found
    - Adaptive generation: Generates with or without context as needed

    Args:
        retrieval_mode: "hybrid" (dense+sparse), "dense", or "sparse".
                       Defaults to config.RETRIEVAL_MODE.
        use_reranker: Enable cross-encoder reranking (creates default reranker).
        reranker: Optional custom reranker instance (takes precedence over use_reranker).
        collection_name: Qdrant collection (defaults to config.VIETNAMESE_COLLECTION_NAME).
        qdrant_host: Qdrant host (defaults to "qdrant" for Docker).
        qdrant_port: Qdrant port (defaults to config.QDRANT_PORT).
        retrieval_k: Number of candidates to retrieve (defaults to config.RETRIEVAL_K).
        top_k: Number of documents after reranking (defaults to config.TOP_K).
        max_rewrite_attempts: Maximum query rewrite attempts (defaults to 3).
    """

    def __init__(
        self,
        retrieval_mode: Optional[str] = "dense",
        use_reranker: bool = False,
        reranker: Optional[BaseReranker] = None,
        collection_name: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        retrieval_k: Optional[int] = None,
        top_k: Optional[int] = None,
        max_rewrite_attempts: int = MAX_REWRITE_ATTEMPTS,
    ) -> None:
        """Initialize the pipeline."""
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.base_retriever = None
        self.reranker: Optional[BaseReranker] = reranker
        self._initialized = False
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker or (reranker is not None)
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._retrieval_k = retrieval_k
        self._top_k = top_k
        self.max_rewrite_attempts = max_rewrite_attempts
        self._workflow = None

    @property
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        return self._initialized

    def startup(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("--- Starting AgenticRAGPipeline Initialization ---")
        t_start = time.time()

        # Resolve configuration with defaults
        retrieval_mode = self.retrieval_mode or config.RETRIEVAL_MODE
        collection_name = self.collection_name or config.VIETNAMESE_COLLECTION_NAME
        qdrant_host = self.qdrant_host or config.QDRANT_HOST
        qdrant_port = self.qdrant_port or config.QDRANT_PORT
        self._resolved_retrieval_k = self._retrieval_k or config.RETRIEVAL_K
        self._resolved_top_k = self._top_k or config.TOP_K

        # Validate retrieval mode
        if retrieval_mode not in _RETRIEVAL_MODE_MAP:
            raise ValueError(
                f"Invalid retrieval_mode: {retrieval_mode}. "
                f"Must be one of: {list(_RETRIEVAL_MODE_MAP.keys())}"
            )

        # 1. Initialize Gemini LLM
        self.model = ChatGoogleGenerativeAI(
            model=config.MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3,
        )

        # 2. Initialize Vietnamese Embedding
        vietnamese_engine = VietnameseEmbedding(
            model_name=config.VIETNAMESE_EMBEDDING_MODEL,
            device="gpu",
        )

        dense_embeddings = None
        sparse_embeddings = None

        if retrieval_mode in ("hybrid", "dense"):
            dense_embeddings = VietnameseDenseAdapter(vietnamese_engine)
            logger.info("Dense embeddings enabled")

        if retrieval_mode in ("hybrid", "sparse"):
            sparse_embeddings = VietnameseSparseAdapter(vietnamese_engine)
            logger.info("Sparse embeddings enabled")

        # 3. Initialize Qdrant Vector Store
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings if sparse_embeddings else None,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=_RETRIEVAL_MODE_MAP[retrieval_mode],
            )
            self.base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self._resolved_retrieval_k}
            )
            logger.info(
                f"Connected to Qdrant: {qdrant_host}:{qdrant_port}/{collection_name} "
                f"(mode={retrieval_mode}, retrieval_k={self._resolved_retrieval_k}, "
                f"top_k={self._resolved_top_k})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(
                f"Failed to connect to Qdrant at {qdrant_host}:{qdrant_port}. "
                f"Ensure Qdrant is running and collection '{collection_name}' exists. "
                f"Original error: {e}"
            ) from e

        # 4. Initialize Reranker (if enabled)
        if self.use_reranker and self.reranker is None:
            self.reranker = CrossEncoderReranker()

        if self.reranker is not None and not self.reranker.is_initialized:
            try:
                self.reranker.startup()
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")
                self.reranker = None

        # 5. Build LangGraph workflow
        self._workflow = self._build_workflow()

        self._initialized = True
        logger.info(
            f"--- AgenticRAGPipeline Initialization Complete in {time.time() - t_start:.2f}s ---"
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agentic RAG."""
        workflow = StateGraph(AgenticState)

        # Add nodes
        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("generate_with_context", self._generate_with_context_node)
        workflow.add_node("generate_direct", self._generate_direct_node)
        workflow.add_node("handle_no_docs", self._handle_no_docs_node)

        # Set entry point
        workflow.set_entry_point("route_query")

        # Add conditional edges from route_query
        workflow.add_conditional_edges(
            "route_query",
            self._route_after_routing,
            {
                "retrieve": "retrieve",
                "generate_direct": "generate_direct",
            },
        )

        # Add edge from retrieve to grade_documents
        workflow.add_edge("retrieve", "grade_documents")

        # Add conditional edges from grade_documents
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate_with_context": "generate_with_context",
                "rewrite_query": "rewrite_query",
                "handle_no_docs": "handle_no_docs",
            },
        )

        # Add edge from rewrite_query back to retrieve
        workflow.add_edge("rewrite_query", "retrieve")

        # Terminal nodes
        workflow.add_edge("generate_with_context", END)
        workflow.add_edge("generate_direct", END)
        workflow.add_edge("handle_no_docs", END)

        return workflow.compile()

    # =========================================================================
    # Node Implementations
    # =========================================================================

    def _route_query_node(self, state: AgenticState) -> Dict[str, Any]:
        """Decide if the query requires document retrieval."""
        query = state["query"]
        logger.info(f"[Route] Evaluating query: '{query}'")

        route_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query router for a Vietnamese law assistant.
Determine if the following query requires searching legal documents.

Return "yes" if the query is about:
- Legal questions, regulations, laws, decrees, circulars
- Court procedures, legal processes, penalties
- Rights, obligations, legal definitions
- Specific legal cases or situations

Return "no" if the query is:
- A greeting or casual conversation (xin chào, cảm ơn, etc.)
- A question you can answer from general knowledge
- Not related to Vietnamese law at all

You must respond with ONLY "yes" or "no", nothing else.""",
                ),
                ("human", "Query: {query}\n\nOutput:"),
            ]
        )

        chain = route_prompt | self.model | StrOutputParser()

        try:
            result = chain.invoke({"query": query}).strip().lower()
            needs_retrieval = result == "yes"
            logger.info(f"[Route] Decision: needs_retrieval={needs_retrieval}")
        except Exception as e:
            logger.error(f"[Route] Error during routing: {e}")
            # Default to retrieval on error
            needs_retrieval = True

        return {"needs_retrieval": needs_retrieval, "current_query": query}

    def _retrieve_node(self, state: AgenticState) -> Dict[str, Any]:
        """Retrieve documents from Qdrant."""
        current_query = state["current_query"]
        logger.info(f"[Retrieve] Searching for: '{current_query}'")

        t_start = time.time()
        initial_docs = self.base_retriever.invoke(current_query)
        t_elapsed = time.time() - t_start

        # Apply reranking if enabled
        if self.reranker and self.reranker.is_initialized and initial_docs:
            logger.info("[Retrieve] Applying reranking...")
            try:
                docs = self.reranker.rerank(
                    current_query, initial_docs, self._resolved_top_k
                )
            except Exception as e:
                logger.error(f"[Retrieve] Reranking failed: {e}")
                docs = initial_docs[: self._resolved_top_k]
        else:
            docs = initial_docs[: self._resolved_top_k]

        logger.info(
            f"[Retrieve] Found {len(docs)} documents in {t_elapsed:.4f}s "
            f"(reranker={'enabled' if self.reranker else 'disabled'})"
        )

        return {"documents": docs}

    def _grade_documents_node(self, state: AgenticState) -> Dict[str, Any]:
        """Grade the relevance of retrieved documents."""
        query = state["query"]
        documents = state["documents"]
        logger.info(f"[Grade] Evaluating {len(documents)} documents for relevance")

        if not documents:
            logger.info("[Grade] No documents to grade")
            return {"relevant_documents": []}

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a relevance grader for Vietnamese legal documents.
Given a user query and a document, determine if the document contains information
that is relevant to answering the query.

A document is relevant if it:
- Contains information directly related to the query topic
- Provides legal context, definitions, or procedures mentioned in the query
- Could help answer the user's legal question

You must respond with ONLY "yes" or "no", nothing else.""",
                ),
                (
                    "human",
                    "Query: {query}\n\nDocument:\n{document}\n\nIs this document relevant? Output:",
                ),
            ]
        )

        chain = grade_prompt | self.model | StrOutputParser()

        relevant_docs = []
        for i, doc in enumerate(documents):
            try:
                # Truncate document content to avoid token limits
                doc_content = doc.page_content[:2000]
                result = (
                    chain.invoke({"query": query, "document": doc_content})
                    .strip()
                    .lower()
                )

                is_relevant = result == "yes"
                logger.debug(f"[Grade] Document {i + 1}: relevant={is_relevant}")

                if is_relevant:
                    relevant_docs.append(doc)
            except Exception as e:
                logger.error(f"[Grade] Error grading document {i + 1}: {e}")
                # Include document on error to avoid losing potentially relevant content
                relevant_docs.append(doc)

        logger.info(
            f"[Grade] {len(relevant_docs)}/{len(documents)} documents marked as relevant"
        )

        return {"relevant_documents": relevant_docs}

    def _rewrite_query_node(self, state: AgenticState) -> Dict[str, Any]:
        """Rewrite the query for better retrieval."""
        original_query = state["query"]
        current_query = state["current_query"]
        rewrite_count = state.get("rewrite_count", 0) + 1

        logger.info(
            f"[Rewrite] Attempt {rewrite_count}/{self.max_rewrite_attempts} "
            f"for query: '{current_query}'"
        )

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query rewriter for a Vietnamese legal search system.
The current query did not retrieve relevant documents from the legal database.

Your task is to reformulate the query to improve search results. Consider:
- Using more specific legal terminology
- Adding relevant Vietnamese legal terms (luật, nghị định, thông tư, etc.)
- Expanding abbreviations
- Rephrasing the question to match how legal documents are written

Respond with ONLY the rewritten query in Vietnamese, nothing else.""",
                ),
                (
                    "human",
                    "Original query: {original_query}\n\nCurrent query: {current_query}\n\nRewritten query:",
                ),
            ]
        )

        chain = rewrite_prompt | self.model | StrOutputParser()

        try:
            new_query = chain.invoke(
                {"original_query": original_query, "current_query": current_query}
            ).strip()
            logger.info(f"[Rewrite] New query: '{new_query}'")
        except Exception as e:
            logger.error(f"[Rewrite] Error during rewriting: {e}")
            # Keep current query on error
            new_query = current_query

        return {"current_query": new_query, "rewrite_count": rewrite_count}

    def _generate_with_context_node(self, state: AgenticState) -> Dict[str, Any]:
        """Generate response using relevant documents as context."""
        query = state["query"]
        relevant_docs = state["relevant_documents"]
        history = state.get("history", [])

        logger.info(
            f"[Generate] Generating response with {len(relevant_docs)} relevant documents"
        )

        # Build context from relevant documents only
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = [doc.metadata for doc in relevant_docs]

        # Generate response
        answer = self._generate_response(query, context_text, history)

        return {"answer": answer, "sources": sources}

    def _generate_direct_node(self, state: AgenticState) -> Dict[str, Any]:
        """Generate response without document retrieval."""
        query = state["query"]
        history = state.get("history", [])

        logger.info("[Generate] Generating direct response (no retrieval needed)")

        direct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a friendly and helpful Vietnamese law assistant.
The user has sent a message that does not require searching legal documents.
Respond naturally and helpfully in Vietnamese.

If the user greets you, greet them back and briefly introduce your capabilities.
If the user asks about topics outside Vietnamese law, politely explain that you
specialize in Vietnamese law and suggest they ask a related question.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ]
        )

        chain = direct_prompt | self.model | StrOutputParser()

        try:
            answer = chain.invoke({"query": query, "history": history})
        except Exception as e:
            logger.error(f"[Generate] Error during direct generation: {e}")
            answer = "Sorry, an error occurred while processing your request. Please try again."

        return {"answer": answer, "sources": []}

    def _handle_no_docs_node(self, state: AgenticState) -> Dict[str, Any]:
        """Handle case when no relevant documents found after max retries."""
        query = state["query"]
        rewrite_count = state.get("rewrite_count", 0)

        logger.info(
            f"[NoDocsHandler] No relevant documents after {rewrite_count} rewrite attempts"
        )

        no_docs_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Vietnamese law assistant. The system could not find any relevant
legal documents matching the user's query after multiple search attempts.

Please:
1. Politely inform that no matching information was found in the database
2. Suggest the user try rephrasing their question more specifically
3. Recommend consulting official sources like the National Legal Information Portal

Respond in Vietnamese in a professional and helpful manner.""",
                ),
                ("human", "User's question: {query}"),
            ]
        )

        chain = no_docs_prompt | self.model | StrOutputParser()

        try:
            answer = chain.invoke({"query": query})
        except Exception as e:
            logger.error(f"[NoDocsHandler] Error: {e}")
            answer = (
                "Xin lỗi, tôi không tìm thấy văn bản pháp luật phù hợp với câu hỏi của bạn. "
                "Vui lòng thử diễn đạt lại câu hỏi hoặc tham khảo Cổng thông tin pháp luật "
                "quốc gia (vbpl.vn) để tra cứu thêm."
            )

        return {"answer": answer, "sources": []}

    # =========================================================================
    # Routing Functions
    # =========================================================================

    def _route_after_routing(self, state: AgenticState) -> str:
        """Determine next node after query routing."""
        if state.get("needs_retrieval", True):
            return "retrieve"
        return "generate_direct"

    def _route_after_grading(self, state: AgenticState) -> str:
        """Determine next node after document grading."""
        relevant_docs = state.get("relevant_documents", [])
        rewrite_count = state.get("rewrite_count", 0)

        if len(relevant_docs) > 0:
            return "generate_with_context"
        elif rewrite_count < self.max_rewrite_attempts:
            return "rewrite_query"
        else:
            return "handle_no_docs"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_response(
        self,
        query: str,
        context_text: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """Generate response using Gemini LLM with context."""
        t_start = time.time()

        system_template = """You are a legal expert specializing in Vietnamese law, with in-depth knowledge of legal
regulations and their practical applications. Your task is to answer legal questions
accurately, clearly, and professionally.

### Instructions:
1. It is mandatory to provide answers in Vietnamese.
2. Base your answers solely on the provided information and avoid adding any assumptions
or external knowledge.
3. Ensure that your response is well-structured, concise, and relevant to the question.
4. If the provided information does not contain the answer, state that clearly and
suggest seeking further clarification.

CONTEXT:
{context}"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt_template | self.model | StrOutputParser()

        try:
            history_messages = history if history else []
            response_text = chain.invoke(
                {
                    "context": context_text,
                    "question": query,
                    "history": history_messages,
                }
            )
        except Exception as e:
            logger.error(f"[Generate] Gemini Error: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời. Vui lòng thử lại."

        t_elapsed = time.time() - t_start
        logger.info(f"[Generate] Response generated in {t_elapsed:.4f}s")

        return response_text

    # =========================================================================
    # Public API (RAGPipeline Interface)
    # =========================================================================

    def retrieve_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve context documents without generating a response.

        Args:
            query: The user's question.

        Returns:
            Dictionary with "context" and "sources" keys.
        """
        if not self._initialized:
            self.startup()

        logger.info(f"Retrieving context for: '{query}'")

        # Run just the retrieval portion
        docs = self.base_retriever.invoke(query)

        if self.reranker and self.reranker.is_initialized and docs:
            docs = self.reranker.rerank(query, docs, self._resolved_top_k)
        else:
            docs = docs[: self._resolved_top_k]

        context_text = "\n\n".join([d.page_content for d in docs])
        source_documents = [d.metadata for d in docs]

        return {
            "context": context_text,
            "sources": source_documents,
        }

    def respond(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full agentic RAG pipeline.

        Args:
            query: The user's question.
            history: Optional list of previous chat messages.

        Returns:
            Dictionary with "answer" and "sources" keys.
        """
        if not self._initialized:
            self.startup()

        logger.info(f"[AgenticRAG] Processing Query: '{query}'")
        total_start_time = time.time()

        # Initialize state
        initial_state: AgenticState = {
            "query": query,
            "current_query": query,
            "history": history or [],
            "documents": [],
            "relevant_documents": [],
            "rewrite_count": 0,
            "needs_retrieval": True,
            "answer": "",
            "sources": [],
        }

        # Run the workflow
        try:
            final_state = self._workflow.invoke(initial_state)
        except Exception as e:
            logger.error(f"[AgenticRAG] Workflow error: {e}")
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại.",
                "sources": [],
            }

        total_time = time.time() - total_start_time
        logger.info(f"[AgenticRAG] === Total Request Time: {total_time:.4f}s ===")

        return {
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
        }
