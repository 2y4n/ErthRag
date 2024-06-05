import streamlit as st
import os
import openai
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize model and parser
model = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o")
parser = StrOutputParser()

# Define the prompt template
template = """
Answer the questions based on the context below in Arabic.
The context below contains information about Saudi Arabia's culture, heritage, and historical sites.
Do not mention the context explicitly in your answer ever.
If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Load and split text
data_path = "data.txt"
loader = TextLoader(data_path)
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splitted = text_splitter.split_documents(text)

# Initialize embeddings and vectorstore
embedding = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding)

# Set Pinecone API key
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("erth")

index_name = "erth"

pinecone = PineconeVectorStore.from_documents(
    splitted, embedding, index_name=index_name
)

# Define the chain
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Streamlit UI
st.image("Erth.png", use_column_width=True)  
st.title("Erth | إرث")

tab1, tab2 = st.tabs(["FT-AraGPT2 Text-to-text", " "]) 

with tab1:
    st.header("Fine-Tuned AraGPT2 Text-To-Text")
    if "gpt2_messages" not in st.session_state:
        st.session_state.gpt2_messages = []

    for message in st.session_state.gpt2_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("إسألني عن التراث السعودي (AraGPT2)"):
        st.session_state.gpt2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chain.invoke(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.gpt2_messages.append({"role": "assistant", "content": response})
