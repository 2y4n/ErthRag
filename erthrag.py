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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "erth"

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

template = """
Answer the question based on the context below in Arabic.
The context below contains information about Saudi Arabia's culture, heritage, and historical sites.
Do not mention the context explicitly in your answer ever.
If the context does not contain the answer, reply "I don't know", and recommend to ask somthing from context below that you can answer good and explain it.

Context: {context}
Question: {question}
"""
sys_prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

loader = TextLoader("data.txt") 
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splitted = text_splitter.split_documents(text)

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding)

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
pinecone_store = PineconeVectorStore.from_documents(splitted, embedding, index_name=INDEX_NAME)

def run_rag_chain(question):
    # Retrieve context from Pinecone
    retriever = pinecone_store.as_retriever()
    docs = retriever.get_relevant_documents(question)
    
    # Join the documents to create a context string
    context = " ".join([doc.page_content for doc in docs])

    # Format prompt with context and question
    formatted_prompt = sys_prompt.format(context=context, question=question)

    # Get response from the model
    response = model.predict(formatted_prompt)
    
    # Parse the response
    parsed_response = parser.parse(response)
    
    return parsed_response

# Streamlit UI
st.image("Erth.png", use_column_width=True)
st.title("Erth | إرث")

#tabs
tab1, tab2 = st.tabs(["  ", " "])

with tab1:
    st.header("Erth Chatbot")
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("إسألني عن التراث السعودي (مثلا : المناطق الاثرية في الرياض والقصيم وحائل)"):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = run_rag_chain(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.rag_messages.append({"role": "assistant", "content": response})
