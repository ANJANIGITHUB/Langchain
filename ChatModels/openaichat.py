from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#load env

load_dotenv()

llm=ChatOpenAI(model='gpt-4')

result=llm.invoke("what is the capital of india ? Why it was made Capital ?",temperature=1.5)

print(result.content)