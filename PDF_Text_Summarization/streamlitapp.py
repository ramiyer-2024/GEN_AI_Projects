import streamlit as st
from langchain_community.llms import Ollama 
from langchain_community.embeddings import OllamaEmbeddings
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
import openai
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity



# Configure logging
logging.basicConfig(level=logging.DEBUG)

st.title("Text Summarization")

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]

session_id=st.text_input("Session ID",value="default_session")
if 'store' not in st.session_state:
    st.session_state.store={}

#Ollama Embeddings
#embedddings = OllamaEmbeddings(model="gemma:2b")

#HuggingFace embeddings
os.environ["HF_TOKEN"]=st.secrets["HF_TOKEN"]
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



#Groq
os.environ["GROQ_API_KEY"]=st.secrets["GROQ_API_KEY"]
groq_api_key=st.secrets["GROQ_API_KEY"]







refine_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the given text.\n\n{text}"
)

#LLM slect
llm_type=st.selectbox("Select LLM",["mixtral-8x7b-32768","llama-3.3-70b-versatile","Gemma2-9b-It"])

#PDF Upload
uploaded_files = st.file_uploader("Choose a file", type='pdf',  accept_multiple_files=True)

def read_files(uploaded_files):
    if uploaded_files:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load_and_split()
        return docs


def select_llm(llm_type):
    llm=ChatGroq(groq_api_key=groq_api_key,model_name=llm_type)
    return llm



def summarize(llm,docs):

    word_length = sum(len(doc.page_content.split()) for doc in docs)

    if word_length<=1000:

        # Summarize using Stuff Documentation chain
        #prompt template
        prompt_template="""
        Provide a summary of the following content:
        Content:{text}

        """
        prompt=PromptTemplate(input_variables=['text'], 
                            template=prompt_template)
        st.subheader("Summarization using Stuff Documentation chain")   
        chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        output_summary=chain.run(docs)
       

    else:
        # Summarize using Map Reduce Documentation chain
        chunks_prompt="""
        Provide a summary of the following content :
        Content:{text}
        """
        map_reduce_template=PromptTemplate(input_variables=['text'], 
                            template=chunks_prompt)

        final_prompt='''
        Provide the final summary of the entire PDF document.
        Add a Title.
        Content:{text}

        '''
        final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

        st.subheader("Summarization using Map Reduce Documentation chain")   
        final_docs = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)
        mapchain=load_summarize_chain(llm=llm, chain_type="map_reduce",map_prompt=map_reduce_template,combine_prompt=final_prompt_template)
        output_summary=mapchain.run(final_docs)
    
    return output_summary





#Text Summarization
if st.button("Summarize"):
    #model
    llm=select_llm(llm_type)
    if uploaded_files:
        st.write(f"Selected Model: {llm_type}")
        for uploaded_file in uploaded_files:
            if uploaded_file.type == 'application/pdf':
                st.write(f"**Processing file:** {uploaded_file.name}")
                docs=read_files(uploaded_files)
                output_summary = summarize(llm,docs)
                st.write(output_summary)

                
        
            else:
                st.write("Unsupported file format")
        
        
    else:

        st.write("Please upload a PDF file")