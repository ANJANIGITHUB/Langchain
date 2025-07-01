#RAG Chatbot
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
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
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#activate your environment
load_dotenv()

#Create model
model=ChatOpenAI(model="gpt-4o")

#We will create embeddings of the using the model
embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")


# Load pre-chunked documents
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS Embeddings
vectorstore = FAISS.load_local(folder_path="faiss_indexes", 
                               embeddings=embedding_model,
                               allow_dangerous_deserialization=True
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
#search_type ='hybrid' to use Mix of BM25 and Dense retriever
retriever_dense = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever_BM25  = BM25Retriever.from_documents(chunks,k=5)

retriever=EnsembleRetriever(retrievers=[retriever_dense,retriever_BM25],
                            weights=[0.8,0.2],)



# # Conversational RAG Chain with Source Documents
# rag_chain = ConversationalRetrievalChain.from_llm(
#     retriever              =retriever,
#     llm                    =model,
#     memory                 =memory,
#     condense_question_llm  =model,
#     return_source_documents=True,
#     verbose                =False

# )

# Define Prompt Template
prompt = PromptTemplate(
    template="""You are a helpful assistant.
Try to answer the question as best as you can using the following context and chat history.
If the context is insufficient, just say you do not know.

Chat History:
{chat_history}

Context:
{context}

Question: {question}""",
    input_variables=["chat_history", "context", "question"]
)
 
# Context Formatter
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Parallel Input Preparation Chain
parallel_chain = RunnableParallel({
    "chat_history": RunnableLambda(lambda x: memory.load_memory_variables(x)["chat_history"]),
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# print(parallel_chain.invoke('What is Agentic AI'))

# Combine Full Chain
parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser

# Ensure memory is persistent across Streamlit reruns
if "memory" not in st.session_state:
    st.session_state.memory = memory
else:
    memory = st.session_state.memory

# UI Header
st.subheader("💬 RAG-powered Chatbot for NLP Research Papers")
st.markdown("Ask a question based on your documents.")

# Sidebar Info
st.sidebar.title("📘 Project Info")
st.sidebar.markdown("""
**👤 Author**: Anjani Kumar  
**🗓️ Version**: 1.0  
**🛠️ Tech Stack**: LangChain, FAISS, Streamlit, OpenAI  
**📄 Description**:  
This chatbot answers queries from research papers using retrieval-augmented generation.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("💡 Ask me about any topic covered in your uploaded documents!")

# Chat History Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input Capture
def submit():
    st.session_state.submitted_input = st.session_state.user_input
    st.session_state.user_input = ""

st.text_input("You:", key="user_input", on_change=submit)
user_input = st.session_state.get("submitted_input", "")

# Chat History Display
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")

# Process Query
if user_input:
    if user_input.strip() == "":
        st.warning("Please enter a valid question.")
        st.stop()

    st.markdown(f"**You:** {user_input}")
    with st.spinner("Thinking..."):

        # 🔄 Run RAG chain
        response = main_chain.invoke(user_input)

        # 🔄 Save interaction to memory
        memory.save_context({"question": user_input}, {"answer": response})

        # 🧪 Debug block
        retrieved = retriever.invoke(user_input)
        debug_context = format_docs(retrieved)
        debug_memory = memory.load_memory_variables({"question": user_input})["chat_history"]

        with st.expander("🧠 Debug Info"):
            st.markdown("**Retrieved Context:**")
            st.code(debug_context)
            st.markdown("**Memory (Chat History):**")
            st.code(debug_memory)

    # 💬 Update session chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
    st.markdown(f"**Bot:** {response}")