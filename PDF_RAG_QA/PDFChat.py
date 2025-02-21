import streamlit as st
import tempfile
import os
import logging
from dotenv import load_dotenv
import cohere

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import BaseRetriever
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize LLM and Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# Streamlit UI
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content.")

# Custom retriever for reranked documents
class RankedDocumentsRetriever(BaseRetriever, BaseModel):
    ranked_docs: list = Field(default_factory=list)

    def get_relevant_documents(self, query: str):
        return self.ranked_docs

# Load and split PDF documents
def load_split(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as file:
        file.write(uploaded_file.getvalue())
        temp_file_path = file.name
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Create retrievers
def create_faiss_retriever(docs):
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def create_bm25_retriever(docs):
    retriever = BM25Retriever.from_documents(documents=docs)
    retriever.k = 5
    return retriever

def create_ensemble_retriever(docs):
    return EnsembleRetriever(
        retrievers=[create_faiss_retriever(docs), create_bm25_retriever(docs)], 
        weights=[0.7, 0.3]
    )

# Rerank results using Cohere
def rerank_results(user_input, retrieved_docs):
    doc_texts = [doc.page_content for doc in retrieved_docs]
    results = cohere_client.rerank(query=user_input, documents=doc_texts, model="rerank-english-v2.0", top_n=3)
    ranked_docs = [retrieved_docs[res.index] for res in results.results]
    return ranked_docs

# Chat session handling
def get_session_history(session: str) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Prompts for RAG
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are assisting in retrieving document-based answers. Rewrite the question to be fully self-contained."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = (
    "You are an assistant that answers questions strictly based on retrieved documents. "
    "ONLY use the retrieved context to answer. "
    "If the answer is not found in the provided context, respond with: 'I don't know.' "
    "Do NOT use external knowledge or make assumptions.\n\n"
    "Context:\n{context}\n"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Streamlit App Logic
if os.getenv("GROQ_API_KEY"):
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        docs = load_split(uploaded_file)
        user_input = st.text_input("Your question:")

        if user_input:
            retriever_ensemble = create_ensemble_retriever(docs)
            retrieved_docs = retriever_ensemble.get_relevant_documents(user_input)
            ranked_docs = rerank_results(user_input, retrieved_docs)
            
            retriever = RankedDocumentsRetriever(ranked_docs=ranked_docs)
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
