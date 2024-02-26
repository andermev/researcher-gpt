import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import openai
import langchain

# Load environment variables
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")


def detect_nearby_places(user_input):
    """
    Detects nearby places based on user input and scrapes websites for additional information.

    Args:
    user_input (str): User input describing their preferences.

    Returns:
    list: A list of recommended places with additional scraped information.
    """
    # Process user input using OpenAI's language models
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    processed_input = response.choices[0].text.strip()

    # Extract relevant information using LangChain
    extracted_info = langchain.extract(processed_input)

    # Perform a search and filter results
    search_results = perform_search(extracted_info)
    filtered_results = filter_results(search_results, extracted_info)

    # Scrape websites for additional information
    for result in filtered_results:
        result["additional_info"] = scrape_website(
            result["objective"], result["url"]
        )

    return filtered_results


def perform_search(extracted_info):
    """
    Performs a search based on extracted information using an external API.

    Args:
    extracted_info (dict): Extracted information from user input.

    Returns:
    list: Search results.
    """
    search_query = extracted_info["cuisine"] + " restaurants near me"
    search_results = requests.get(
        f"https://api.example.com/search?q={search_query}"
    ).json()
    return search_results


def filter_results(search_results, extracted_info):
    """
    Filters and ranks search results based on user preferences.

    Args:
    search_results (list): Raw search results.
    extracted_info (dict): Extracted information from user input.

    Returns:
    list: Filtered and ranked results.
    """
    filtered_results = [
        result
        for result in search_results
        if result["rating"] >= extracted_info["min_rating"]
    ]
    filtered_results.sort(key=lambda x: x["distance"])
    return filtered_results


def scrape_website(objective, url):
    """
    Scrapes a website and summarizes its content based on the objective.

    Args:
    objective (str): The scraping objective.
    url (str): URL of the website to be scraped.

    Returns:
    str: Scraped content or its summary.
    """
    print("Scraping website...")
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = (
        f"https://chrome.browserless.io/content?token={browserless_api_key}"
    )
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# Example usage
user_input = "I like to eat hamburgers"
recommended_places = detect_nearby_places(user_input)
print(recommended_places)
