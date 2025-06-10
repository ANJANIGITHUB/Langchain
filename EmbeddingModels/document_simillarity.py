from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=128)

docs=['Kapil Dev Played Cricket','Ronaldo Played Football','Milkha Singh was a Athelete']
embeded_docs=embeddings.embed_documents(docs)

#print(str(embeded_docs))

embeded_qury=embeddings.embed_query('Who was Milkha Singh ?')

#print(str(embeded_qury))

similarity_score=cosine_similarity([embeded_qury],embeded_docs)[0]

indexed_list = list(enumerate(similarity_score))
sorted_by_value = sorted(indexed_list, key=lambda x: x[1])
print(sorted_by_value[-1])
print("Matched Value:")
print(docs[sorted_by_value[-1][0]])
