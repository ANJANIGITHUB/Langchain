#Paraller Chains
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model_openai=ChatOpenAI(model='gpt-4o-mini')
#model_anthro=ChatAnthropic(model_name='clause-3')

prompt1=PromptTemplate(template='Generate Notes from the Given text document \n {text}',
                       input_variables=['text']
                       )

prompt2=PromptTemplate(template='Generate 5 Quiz questions from the Given text document \n {text}',
                       input_variables=['text']
                       )

prompt3=PromptTemplate(template='Merge Notes and Quiz into a single document \n {notes} and {quiz}',
                       input_variables=['notes','quiz']
                       )

parser=StrOutputParser()

parallel_chain=RunnableParallel({'notes':prompt1 | model_openai | parser,
                                 'quiz' :prompt2 | model_openai | parser
                                 }
                                )

merge_chain   = prompt3 | model_openai | parser

#Final Chain
chain = parallel_chain | merge_chain

text_data=""" **
The goal of linear regression is to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the sum of squared errors between predicted and actual values—a method known as **Ordinary Least Squares (OLS)**.

Linear regression assumes a linear relationship, homoscedasticity (constant variance of errors), independence of observations, and normally distributed residuals. If these assumptions hold, linear regression provides accurate predictions and interpretable coefficients.

It is widely used across industries for forecasting, trend analysis, and identifying relationships between variables. Despite its simplicity, linear regression forms the basis for more complex models like logistic regression and ridge regression. It's also favored for its speed, interpretability, and ease of implementation in tools like Python (scikit-learn), R, and Excel.
"""

result=chain.invoke({'text':text_data})

print(result)

chain.get_graph().print_ascii()