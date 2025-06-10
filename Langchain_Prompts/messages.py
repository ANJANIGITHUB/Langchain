from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

messages=[SystemMessage(content="You are a helpful AI Assistant"),
          HumanMessage(content="Tell me about Transformers in NLP in 50 words")
         ]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)