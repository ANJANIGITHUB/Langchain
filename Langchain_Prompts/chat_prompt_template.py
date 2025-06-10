from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_template=ChatPromptTemplate([('system','You are a helpfull {role}'),
                                  ('human','Tell me about {topic}')])
result=chat_template.invoke({'role':'AI Agent','topic':'LSTM'})

print(result)
print('system message:',result.messages[0].content)
print('human message :',result.messages[1].content)