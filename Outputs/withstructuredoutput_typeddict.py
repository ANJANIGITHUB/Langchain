from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()

class structured_output(TypedDict):
    key_themes    :Annotated[list[str],"Give me all the key themes discussed in the review in this list"]
    summary       :Annotated[str,"Give me summary of the given review"]
    sentiment     :Annotated[str,"Give sentiment of the given review"]
    pros          :Annotated[Optional[list[str]],"In case pros are given ,give me in this list"]

model=ChatOpenAI(model='gpt-4o-mini')

structured_model=model.with_structured_output(structured_output)

result=structured_model.invoke("""Product: Philips Air Fryer HD9252/90

I bought this air fryer hoping to replicate some of my favorite restaurant appetizers, and it exceeded my expectations! From crispy fries to perfectly golden samosas and even tandoori-style paneer, this little machine has been a total game changer.

It heats up fast, cooks evenly, and uses very little oil. The results are honestly close to what I’d get at a decent casual dining place. Cleanup is also super easy — just pull out the basket and rinse.

If you're someone who craves restaurant-style food but wants to keep it healthier (and cheaper!), this is 100% worth it.

Pros: Fast cooking, easy cleanup, restaurant-style crispiness
Cons: Basket size is a bit small for a family of 4

Highly recommend!""")

print('key_themes:',result['key_themes'])
print('Summary:',result['summary'])
print('Sentiment:',result['sentiment'])
print('Pros:',result['pros'])