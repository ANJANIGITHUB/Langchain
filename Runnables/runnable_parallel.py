from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel

#load env
load_dotenv()

#create model
model=ChatOpenAI(model='gpt-4o-mini')

#prompt

template=PromptTemplate(template="write a tweet about {topic}",
                        input_variables=['topic'])

template2=PromptTemplate(template="Write a linkedIn post  about {topic}",
                        input_variables=['topic'])

#parser
parser=StrOutputParser()

#runnable Parallel usage
Runnable=RunnableParallel({'tweet'   :RunnableSequence(template,model,parser),
                           'linkedin':RunnableSequence(template2,model,parser)})

result=Runnable.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])