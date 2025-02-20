## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
import tempfile
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import logging

import cohere
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Configure logging
logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

#Groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile")

#Cohere
cohere_client=cohere.Client(api_key=os.getenv("COHERE_API_KEY"))


#load and split the documents
def load_split(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as file:
        file.write(uploaded_file.getvalue())
        temp_file_path =file.name
    loader=PyPDFLoader(temp_file_path)
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

def create_faiss_retriever(docs):
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def create_bm25_retriever(docs):
    retriever = BM25Retriever.from_documents(documents=docs, embedding=embeddings)
    retriever.k = 5
    return retriever

def create_ensemble_retriever(docs):
    faiss_retriever = create_faiss_retriever(docs)
    bm25_retriever = create_bm25_retriever(docs)    
    hybrid_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.7, 0.3])
    return hybrid_retriever

# def rerank_results(user_input, retrived_docs):
#     doc_texts = [doc.page_content for doc in retrived_docs]
#     results = cohere_client.rerank(query=user_input, documents=doc_texts, model="rerank-english-v2.0", top_n=3)
#     ranked_docs = [retrived_docs[result.index] for result in results]
#     return ranked_docs


def get_session_history(session:str)->BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store={}

    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

contextualize_q_system_prompt = (
    "You are assisting in retrieving document-based answers. "
    "Rewrite the question to be fully self-contained, "
    "ensuring it remains relevant to the uploaded documents. "
    "If the question is unrelated to the uploaded documents, return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )





# Answer question
system_prompt = (
    "You are an assistant that answers questions strictly based on retrieved documents. "
    "ONLY use the retrieved context to answer. "
    "If the answer is not found in the provided context, respond with: 'I don't know.' "
    "Do NOT use external knowledge or make assumptions."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )







## Check if groq api key is provided
if api_key:


    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf" ,accept_multiple_files=False)
    ## Process uploaded  PDF
    if uploaded_files:
        docs=load_split(uploaded_files)

        ## Create retriever
        retriever=create_ensemble_retriever(docs)

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            #st.write(api_key)
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key "abc123" in `store`.
            )
            
            # Log the embeddings
            # embeddings = response.get('embeddings', [])
            # logging.debug(f"Embeddings: {embeddings}")
            
            # if not embeddings:
            #     st.error("Embeddings are empty. Please check the embedding generation process.")
            # else:
            #     st.write(st.session_state.store)
            #     st.write("Assistant:", response['answer'])
            #     st.write("Chat History:", session_history.messages)

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)