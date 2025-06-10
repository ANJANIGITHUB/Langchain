from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=64)

query="What is the Capital of Karnataka"

result=embeddings.embed_query(query)

print('Embeddings for "{}" is as below:'.format(query))
print(str(result))

#-------------------------------------------------------------------
print("----------------------------------------------------------------------------------------------------")

documents=["New Delhi is Capital of India",
           "Paris is the Capital of France"]

result2=embeddings.embed_documents(documents)

print('Embeddings for "{}" is as below:'.format(documents))
print(str(result2))

