from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embedding, filter_to_minimal_docs, load_pdf_file, text_split

load_dotenv()

MEDICALBOT_PINECONE_API_KEY = os.getenv("MEDICALBOT_PINECONE_API_KEY")
GOOGLE_GEN_AI = os.getenv("GOOGLE_GEN_AI")

os.environ["MEDICALBOT_PINECONE_API_KEY"] = MEDICALBOT_PINECONE_API_KEY
os.environ["GOOGLE_GEN_AI"] = GOOGLE_GEN_AI

extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_embedding()

pinecone_api_key = MEDICALBOT_PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Change this before a run to remove the duplicate chunking issue
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)