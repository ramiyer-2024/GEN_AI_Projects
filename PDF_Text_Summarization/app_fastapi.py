

import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
import openai
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, File, UploadFile

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

app = FastAPI()

def read_files(uploaded_files):
   
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_files.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load_and_split()
    return docs

def select_llm(model_name):
    llm=ChatGroq(groq_api_key=groq_api_key,model_name=model_name)
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

        
        final_docs = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)
        mapchain=load_summarize_chain(llm=llm, chain_type="map_reduce",map_prompt=map_reduce_template,combine_prompt=final_prompt_template)
        output_summary=mapchain.run(final_docs)
    
    return output_summary





@app.post("/summarize/")
async def create_upload_file(file: UploadFile = File(...)):
    docs=read_files(file.file)
    llm=select_llm("llama-3.3-70b-versatile")
    output_summary=summarize(llm,docs)
    return {"summary":output_summary}
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)