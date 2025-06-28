#RAG Chatbot
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_community.retrievers import BM25Retriever#,EnsembelRetrievers
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#activate your environment
load_dotenv()

#Create model
model=ChatOpenAI(model="gpt-4o")

#1. Indexing technique

# Load all .pdf files from a rag_dir directory
# loader = DirectoryLoader(
#     path="rag_dir", 
#     glob="*.pdf"  ,
#     loader_cls=PyPDFLoader
# )

# docs = loader.load()


documents = []
pdf_dir = Path("rag_dir")
for pdf_file in pdf_dir.rglob("*.pdf"):
    #PyMuPDFLoader: Is fast and efficient for large docs ;Supports tables, images, per-page metadata

    loader = PyMuPDFLoader(pdf_file, mode="page", extract_tables="markdown")

    for doc in loader.lazy_load():
        documents.append(doc)

print(f"Total pages loaded with PyMuPDFLoader: {len(documents)}")


#Split all the documents into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,
                                             chunk_overlap=10
                                             )
chunks=text_splitter.split_documents(documents)
# print(len(chunks))
# print(chunks)

#We will create embeddings of the using the model
embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")

#Use FAISS vector database to store the embeddings
vectorstore=FAISS.from_documents(documents=chunks,
                                 embedding=embedding_model
                                 )


# Memory - Summarized to avoid token overload
memory = ConversationSummaryBufferMemory(
    llm=model,
    memory_key="chat_history",
    input_key="question",   # 🔑 Required by ConversationalRetrievalChain
    output_key="answer",    # 🔑 This matches what we want to store
    return_messages=True
)

# Retriever with Custom Search Params
#search_type ='hybrid' to use Mix of BM5 and Dense retriever
retriever_dense = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_BM25=BM25Retriever.from_documents(chunks,k=3)

retriever=EnsembleRetriever(retrievers=[retriever_dense,retriever_BM25],
                            weights=[0.8,0.2],)

# Conversational RAG Chain with Source Documents
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False

)

# Continuous Chat
print("📚 ChatBot Ready. Ask your question (type 'exit' or 'quit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit","bye"]:
        break

    result = rag_chain(query)
    print("\nBot:", result["answer"])
