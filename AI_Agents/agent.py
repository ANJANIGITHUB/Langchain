from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import AgentExecutor,create_react_agent
from langchain import hub #for importing predefined prompts


#load the environment variables
load_dotenv()

#tool
search_tool=DuckDuckGoSearchRun()
query="what is top 3 trending news in short ?"
result=search_tool.invoke(query)

#print(result)

#llm
llm=ChatOpenAI()
#result=llm.invoke('Who is current president of india')
#print(result.content)

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()

#propmt

prompt=hub.pull('hwchase17/react')

#create the react(reasoning and action) agent call

agent=create_react_agent(llm=llm,
                         tools=[search_tool,get_weather_data],
                         prompt=prompt)


agent_executor=AgentExecutor(agent=agent,
                              tools=[search_tool,get_weather_data],
                              verbose=True)

response=agent_executor.invoke({'input':'Who won the IPL 2025 ? what was the weather of that city where final match of IPL 2025 held'})

#print(response)
print(response['output'])