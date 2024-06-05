import os
import streamlit as st
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

st.image("Erth.png", use_column_width=True)  
st.title("Erth | إرث")

# Access the API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["openai_api_key"]
PINECONE_API_KEY = st.secrets["api_keys"]["pinecone_api_key"]
INDEX_NAME = "erth"

# Initialize OpenAI model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

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
parser = StrOutputParser()

# Load and split the data
loader = TextLoader("data.txt")  # Adjust the path as needed
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splitted = text_splitter.split_documents(text)

# Initialize embeddings and vector store
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding)

# Initialize Pinecone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
pinecone_store = PineconeVectorStore.from_documents(splitted, embedding, index_name=INDEX_NAME)

# Define the RAG chain
chain = (
    {"context": pinecone_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Streamlit UI
st.title("RAG-based Chatbot")

user_input = st.text_input("Ask your question:")
if user_input:
    response = chain.invoke(user_input)
    st.write("Response:", response)

# To run the Streamlit app, use the command: `streamlit run app.py`
