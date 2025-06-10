from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

prompt_template=ChatPromptTemplate([('system','You are a helpful assistant'),
                                    MessagesPlaceholder(variable_name='chat_history'),
                                    ('human','{query}')
                                   ])

chat_history=[]

with open('C:/Users/AKU493/OneDrive - Maersk Group/GenAI_Practice/llm/Langchain_Prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines)

result=prompt_template.invoke({'chat_history':chat_history,'query':'where is my order ?'})
print(result)