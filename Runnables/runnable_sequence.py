from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

#load env
load_dotenv()

#create model
model=ChatOpenAI(model='gpt-4o-mini')

#prompt

template=PromptTemplate(template="write a note about qualifying {topic}",
                        input_variables=['topic'])

template2=PromptTemplate(template="Summarize the given {text} in 5 points",
                        input_variables=['text'])

#parser
parser=StrOutputParser()

#chain

chain=RunnableSequence(template,model,parser,template2,model,parser)

#chain invoking

print(chain.invoke({'topic':'Commercial Pilot'}))