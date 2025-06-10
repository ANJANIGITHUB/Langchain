from pydantic import BaseModel,EmailStr,Field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate 

load_dotenv()

model=ChatOpenAI(model='gpt-4o-mini')

class Person(BaseModel):
    name:str =Field(description='Name of the person')
    age :int =Field(gt=18)
    city:str =Field(description='City of the Person')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(template='Give me the name,age and city of first president of {country_name}\n {format_instruction}',
                        input_variables=['country_name'],
                        partial_variables={'format_instruction':parser.get_format_instructions()}
                        )

#chain

chain=template | model | parser

#result

result=chain.invoke({'country_name':'USA'})

print(result)