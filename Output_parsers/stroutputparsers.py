from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#load environment
load_dotenv()

#create llm object
llm=ChatOpenAI(model='gpt-4o-mini')

#template1
template1=PromptTemplate(template='Write a detailed report on {topic}',
                         input_variables=['topic'])

#template2
template2=PromptTemplate(template='Write a 5 lines bullet points summary on {text}',
                         input_variables=['text'])

#output parser object
output_parser=StrOutputParser()

#chain creation
chain=template1 | llm | output_parser | template2 | llm | output_parser

#result
result=chain.invoke({'topic':'Leadership'})

print(result)