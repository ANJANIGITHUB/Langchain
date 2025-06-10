from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from dotenv import load_dotenv

#load environment variables

load_dotenv()

model=ChatOpenAI(model='gpt-4o',temperature=0.5)
chat_history=[SystemMessage(content='You are helpfull AI assistant')]

while True:
    user_input=input("You :")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    #print("User Query:",user_input)
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:',result.content)

print(chat_history)