from langchain_openai import OpenAI
from dotenv import load_dotenv

#load the environment
load_dotenv()

llm=OpenAI(model='gpt-3.5-turbo-instruct')

result=llm.invoke("What is the difference betwwen Linear and Logistic regression")

print(result)