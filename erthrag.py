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

# Access the API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
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

# Define the retrieval-augmented generation (RAG) chain
def run_rag_chain(question):
    # Retrieve context from Pinecone
    retriever = pinecone_store.as_retriever()
    docs = retriever.get_relevant_documents(question)
    
    # Join the documents to create a context string
    context = " ".join([doc.page_content for doc in docs])
    
    # Format prompt with context and question
    formatted_prompt = prompt.format(context=context, question=question)
    
    # Get response from the model
    response = model.predict(formatted_prompt)
    
    # Parse the response
    parsed_response = parser.parse(response)
    
    return parsed_response

# Inject custom CSS
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 80vh;
    }
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column-reverse;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.image("Erth.png", use_column_width=True)
st.title("Erth | إرث")

# Tabs
tab1, tab2 = st.tabs(["RAG-based Chatbot", ""])

with tab1:
    st.header("RAG-based Chatbot")
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Input box
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    if prompt := st.chat_input("إسألني عن التراث السعودي (RAG-based)"):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = run_rag_chain(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.rag_messages.append({"role": "assistant", "content": response})
    st.markdown('</div>', unsafe_allow_html=True)

    # Display chat messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for message in reversed(st.session_state.rag_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
