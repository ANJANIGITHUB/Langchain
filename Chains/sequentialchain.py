from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model =ChatOpenAI(model='gpt-4o-mini')

template1=PromptTemplate(template='Give me summary about {topic}',
                        input_variables=['topic'])

template2=PromptTemplate(template='Get main topics out of the {text}',
                        input_variables=['text'])


parser=StrOutputParser()

#simplechain

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'Generative AI'})

print(result)
chain.get_graph().print_ascii()