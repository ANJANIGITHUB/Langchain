from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough

#load env
load_dotenv()

#create model
model=ChatOpenAI(model='gpt-4o-mini')

#prompt

template=PromptTemplate(template="write a Joke about {topic}",
                        input_variables=['topic'])

template2=PromptTemplate(template="Write a summary about the joke {text}",
                        input_variables=['text'])

#parser
parser=StrOutputParser()

runnablesequencegenerator=RunnableSequence(template,model,parser)

#runnable Parallel usage
Runnable=RunnableParallel({'Joke'   :RunnablePassthrough(),
                           'Summary':RunnableSequence(template2,model,parser)})

final_chain=RunnableSequence(runnablesequencegenerator,Runnable)
result=final_chain.invoke({'topic':'AI'})

print(result)