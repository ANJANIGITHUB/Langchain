from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model =ChatOpenAI(model='gpt-4o-mini')

template=PromptTemplate(template='Give me top 3 highest paying salaries in {dep} in India',
                        input_variables=['dep'])

parser=StrOutputParser()

#simplechain

chain=template | model | parser

result=chain.invoke({'dep':'I.T.'})

print(result)
chain.get_graph().print_ascii()