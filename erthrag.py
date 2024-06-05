import streamlit as st
import os
import openai
import glob
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone

# Set OpenAI API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize model and parser
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")
parser = StrOutputParser()

# Define the prompt template
template = """
Anwser the questions based on the context below in arabic.
the context below having information about Saudi Arabia's Culture, heritage, and historical sites.
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
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding)

# Set Pinecone API key
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
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
st.title("Saudi Arabia Culture Chatbot")

question = st.text_input("Enter your question about Saudi Arabia's culture, heritage, or historical sites:")

if st.button("Ask"):
    if question:
        response = chain.invoke(question)
        st.write("Answer:", response)
    else:
        st.write("Please enter a question.")

