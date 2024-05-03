import os
from langchain import  hub
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import

import streamlit as st


# streamlit framework
# st.title('Langchain Demo with OpenAI api')
# input_text = st.text_input('Search the topic you want')

PDF_PATH = 'data/fundamentals-of-foodnutrition-and-diet-therapy.pdf'


loader = PyPDFLoader(PDF_PATH)
data = loader.load()

# Split your data up into smaller documents with Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(data)

vector_store = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())

retriever = vector_store.as_retriever()
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# OpenAI LLM
llm = Ollama(model='llama3')


# Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke('nutrition for cancer'))

# print(llm('nutrition for cancer'))
# if input_text:
#     st.write(llm(input_text))
