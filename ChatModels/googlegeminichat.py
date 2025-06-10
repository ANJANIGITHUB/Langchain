from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

#load env

load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')

result=llm.invoke("what is the capital of india ?")

print(result.content)