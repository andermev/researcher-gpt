# Importing required libraries
import os
import requests
import json
import googlemaps
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain.schema import SystemMessage
from fastapi import FastAPI

# Load environment variables
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
trip_advisor_api_key = os.getenv("TRIP_ADVISOR_API_KEY")

# Initialize Google Maps client
gmaps = googlemaps.Client(key=google_maps_api_key)


class Query(BaseModel):
    query: str


# Tool for search
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text


# Tool for scraping
def scrape_website(objective: str, url: str):
    print("Scraping website...")
    headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = (
        f"https://chrome.browserless.io/content?token={browserless_api_key}"
    )
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = f"""
    Write a summary of the following text for {objective}:
    "{content}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["content", "objective"]
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output


class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# Tool for searching places
def search_places(query):
    places = gmaps.places(query)
    return places


# Tool for searching trip advisor
# Function to search for places using the TripAdvisor API
def search_tripadvisor(query):
    """
    Search for places using the TripAdvisor API.

    Args:
    query (str): The query string for searching places.

    Returns:
    dict: Data received from the TripAdvisor API or None if the request fails.
    """
    url = "https://api.content.tripadvisor.com/api/v1/location/nearby_search"
    params = {
        "key": trip_advisor_api_key,  # Replace with your API key
        "location": "6.272402557612021, -75.56121463470522",  # Example: New York City coordinates
        "radius": "1000",  # Search within a 5000 meter radius
        # Add other parameters as required by the API
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return None


class SearchPlacesTool(BaseTool):
    name = "search_places"
    description = """Expertly crafted for a comprehensive exploration of destinations, this tool extends beyond basic location searches by incorporating insights from popular platforms like Facebook, and other reputable sources. It specializes in fetching and presenting locations that are not only relevant to the user's query but also highly rated and frequently reviewed by the community. The tool prioritizes places with a substantial volume of feedback and high ratings, ensuring users receive recommendations that are both popular and well-regarded. This approach ensures a more valuable and trustworthy selection, steering clear of destinations with minimal user interactions or low ratings."""
    args_schema: Type[BaseModel] = Query

    def _run(self, query: str):
        return search_places(query)

    def _arun(self, query: str):
        raise NotImplementedError("Async run not implemented for this tool")


# Tool class for searching places using the TripAdvisor API
class SearchTripAdvisorTool(BaseTool):
    name = "search_tripadvisor"
    description = "Useful for searching places based on user input using the TripAdvisor API"
    args_schema: Type[BaseModel] = Query

    def _run(self, query: str):
        """
        Synchronous method to run the TripAdvisor search.

        Args:
        query (str): The query string for searching places.

        Returns:
        Any: The result of the TripAdvisor search.
        """
        return search_tripadvisor(query)

    def _arun(self, query: str):
        """
        Asynchronous method for the TripAdvisor search. Currently not implemented.

        Raises:
        NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Async run not implemented for this tool")


# Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for answering questions about current events, data",
    ),
    ScrapeWebsiteTool(),
    # SearchPlacesTool(),
    SearchTripAdvisorTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# Set this as an API endpoint via FastAPI
app = FastAPI()


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content["output"]
    return actual_content
