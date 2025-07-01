#RAG Chatbot
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from pathlib import Path
import time
import warnings
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#activate your environment
load_dotenv()

#Create model
model=ChatOpenAI(model="gpt-4o")

start_time=time.time()
#1. Indexing technique

#Load all .pdf files from a rag_dir directory
documents = []
pdf_dir = Path("rag_dir")
for pdf_file in pdf_dir.rglob("*.pdf"):
    #PyMuPDFLoader: Is fast and efficient for large docs ;Supports tables, images, per-page metadata

    loader = PyMuPDFLoader(pdf_file, mode="page", extract_tables="markdown")

    for doc in loader.lazy_load():
        documents.append(doc)

print(f"Total pages loaded with PyMuPDFLoader: {len(documents)}")


#Split all the documents into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,
                                             chunk_overlap=150
                                             )
chunks=text_splitter.split_documents(documents)

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# print(len(chunks))
# print(chunks)

#We will create embeddings of the using the model
embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")

#Use FAISS vector database to store the embeddings
vectorstore=FAISS.from_documents(documents=chunks,
                                 embedding=embedding_model
                                 )

# Save vectorstore to disk (optional)
vectorstore.save_local("faiss_indexes")

end_time=time.time()
total_time=end_time-start_time
print('Indexing complete.')
print(f"Total indexing time: {total_time:.2f} seconds")


