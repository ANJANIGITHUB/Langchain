from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",  # Must match model's capability
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India.How many states india having.Give me bullet point wise answer")

print(result.content)