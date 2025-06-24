from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

#activate your environment
load_dotenv()

#Create model
model=ChatOpenAI(model="gpt-4o")

#1. Indexing technique

# Load all .pdf files from a rag_dir directory
loader = DirectoryLoader(
    path="rag_dir", 
    glob="*.pdf"  ,
    loader_cls=PyPDFLoader
)

docs = loader.load()
#print(len(docs))
# print(docs[1].page_content)
# print(docs[1].metadata)
#Note:we can perform lazy_load if we wanted to improve te performance of the load

#Split all the documents into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,
                                             chunk_overlap=10
                                             )
chunks=text_splitter.split_documents(docs)
# print(len(chunks))
# print(chunks)

#We will create embeddings of the using the model
embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")

#Use FAISS vector database to store the embeddings
vectorstore=FAISS.from_documents(documents=chunks,
                                 embedding=embedding_model)

#Store Embeddings Locally
vectorstore.save_local("faiss_embeddings")

#2.Retrieval


retriever=vectorstore.as_retriever(search_kargs={'k':2})


# for i,doc in enumerate(result):
#     print(f"\n-------------Result {i+1}----------------")
#     print(doc.page_content)


#3.Augmentation

prompt=PromptTemplate(template=""" You are a helpful assistant.
                          Answer only for the provided context.
                          If the context is in sufficient,Just say you do not know.
                      
                          {context}
                          question:{question}""",
                       input_variables=['context','question']
                      )       

#commented start   
#context_text="\n\n".join(doc.page_content for doc in result)

#final_prompt=prompt.invoke({'context':context_text,'question':query})

#4.Generation

# final_result=model.invoke(final_prompt)
# print(final_result)
# print(final_result.content)

#commented end 

query ="What is total experience in IT ?"
result=retriever.invoke(query)

#Create a function to create context_text
def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in result)
    return context_text

#5.Chain Creation
parallel_chain=RunnableParallel({'context':retriever | RunnableLambda(format_docs),
                                 'question':RunnablePassthrough()}
                                )

#print(parallel_chain.invoke(query))

parser=StrOutputParser()

main_chain=parallel_chain | prompt | model | parser 

result=main_chain.invoke(query)

print(result)