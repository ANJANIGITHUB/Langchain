from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

#load environment
load_dotenv()

#create llm model
model=ChatOpenAI(model='gpt-4o-mini')

#response schema

schema=[ResponseSchema(name='President1',description='Give me his tenure details in 20 words'),
        ResponseSchema(name='President1_Age',description='Give me his age at the time of president'),
        ResponseSchema(name='President2',description='Give me his tenure details in 20 words'),
        ResponseSchema(name='President2_Age',description='Give me his age at the time of president'),
        ResponseSchema(name='President3',description='Give me his tenure details in 20 words'),
        ResponseSchema(name='President3_Age',description='Give me his age at the time of president')
       ]

#parser
parser=StructuredOutputParser.from_response_schemas(schema)

#prompt template

template=PromptTemplate(template='Give me first 3 presidents of {country_name}\n {format_instruction}',
                        input_variables=['country_name'],
                        partial_variables={'format_instruction':parser.get_format_instructions}
                        )

#chain

chain=template | model | parser

result=chain.invoke({'country_name':'india'})

print(result)
