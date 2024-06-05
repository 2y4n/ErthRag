import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Access the API keys from environment variables or directly
OPENAI_API_KEY = OPENAI_API_KEY
PINECONE_API_KEY = PINECONE_API_KEY
INDEX_NAME = "erth"

# Initialize Pinecone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load and split the data
loader = TextLoader("data.txt")  # Adjust the path as needed
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splitted = text_splitter.split_documents(text)

# Initialize embeddings and vector store
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a Pinecone index and add the vectors
pinecone_index = Pinecone.create_index(INDEX_NAME, dimension=1536)  # Adjust dimension if necessary
pinecone_store = PineconeVectorStore(pinecone_index, embedding)

# Add documents to the Pinecone vector store
pinecone_store.add_documents(splitted)
