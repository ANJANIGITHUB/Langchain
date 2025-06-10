#Conditional Chains
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field,BaseModel
from typing import Literal

load_dotenv()

model_openai=ChatOpenAI(model='gpt-4o-mini')

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['Positive','Negative'] =Field(description='Give the sentiment of the user feedback')

parser_pydantic=PydanticOutputParser(pydantic_object=Feedback)

prompt=PromptTemplate(template='Classify the given user feedback as Postive or Negative \n {feedback} \n {format_instruction}',
                       input_variables=['feedback'],
                       partial_variables={'format_instruction':parser_pydantic.get_format_instructions()}
                       )

#chain=prompt | model_openai | parser
classifier_chain=prompt | model_openai |parser_pydantic

# result=chain.invoke({'feedback':'This Smartphone is Good .It does not hangs.'})

# print(result.sentiment)

prompt1=PromptTemplate(template='Write an appropriate response to this Postive feedback \n {feedback}',
                       input_variables=['feedback'])

prompt2=PromptTemplate(template='Write an appropriate response to this Negative feedback \n {feedback}',
                       input_variables=['feedback'])

branch_chain=RunnableBranch((lambda x:x.sentiment  =='Positive',prompt1 | model_openai | parser ),
                             (lambda x:x.sentiment =='Negative',prompt2 | model_openai | parser ),
                             RunnableLambda(lambda x:'Could Not Find Sentiment')
                            )

#Final Chain

chain= classifier_chain | branch_chain

result=chain.invoke({'feedback':'This is a great phone'})

print(result)