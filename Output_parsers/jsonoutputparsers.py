#Using Hugging Face code
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

# load_dotenv()

# # Define the model
# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# parser=JsonOutputParser()

# # template=PromptTemplate(template='give me name name,age and location of any fictious object \n {format_instruction}',
# #                         input_variables=[],
# #                         partial_variables={'format_instruction':parser.get_format_instructions})

# template=PromptTemplate(template='Give me 5 points about the {topic} \n {format_instruction}',
#                         input_variables=['topic'],
#                         partial_variables={'format_instruction':parser.get_format_instructions})

# chain = template | model | parser

# result = chain.invoke({'topic':'LLM'})

# print(result)


#Using OpenAI code
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
model = ChatOpenAI(model='gpt-4o-mini')

parser=JsonOutputParser()

# template=PromptTemplate(template='give me name name,age and location of any fictious object \n {format_instruction}',
#                         input_variables=[],
#                         partial_variables={'format_instruction':parser.get_format_instructions})

template=PromptTemplate(template='Give me 5 points about the {topic} \n {format_instruction}',
                        input_variables=['topic'],
                        partial_variables={'format_instruction':parser.get_format_instructions})

chain = template | model | parser

result = chain.invoke({'topic':'LLM'})

print(result)
print(type(result))