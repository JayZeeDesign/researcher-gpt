import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")


llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
# search = SerpAPIWrapper()


def summary(objective, content):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


def scrape_website(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post("https://chrome.browserless.io/content?token=2db344e9-a08a-4179-8f48-195a2f7ea6ee", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")        


# def write_twitter_threads(objective: str, context: str):
#     prompt = """
#     You are a world class twitter influencers & thought leader;
#     Pleae write a twitter thread about {objective} based on the following content:
#     {context}

#     1/ The thread needs to be engaging, informative with good data
#     2/ The thread needs to be around than 3-5 tweets
#     3/ The thread needs to address the {objective} topic very well
#     4/ The thread needs to be viral, and get at least 1000 likes
#     5/ The thread needs to be written in a way that is easy to read and understand
#     6/ The thread needs to give audience actionable advice & insights too

#     TWITTER THREAD:
#     """

#     prompt_template = PromptTemplate(template=prompt, input_variables=["objective", "context"])

#     twitter_thread_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

#     twitter_thread = twitter_thread_chain.predict(objective=objective, context=context)

#     return twitter_thread


# class TwitterThreadInput(BaseModel):
#     """Inputs for scrape_website"""
#     objective: str = Field(description="The objective & task that users give to the agent")
#     context: str = Field(description="All the relevant knowledge & context to help the agent write the twitter thread for the objective, the context should be as detailed as possible, full of data & information, as well as reference links")

# class TwitterThreadTool(BaseTool):
#     name = "write_twitter_thread"
#     description = "useful when you need to write a high quality twitter thread for a given objective, passing both objective and context to the function"
#     args_schema: Type[BaseModel] = TwitterThreadInput

#     def _run(self, objective: str, context: str):
#         return write_twitter_threads(objective, context)
    
#     def _arun(self, objective: str):
#         raise NotImplementedError("get_stock_performance does not support async")



class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    
    def _arun(self, url: str):
        raise NotImplementedError("get_stock_performance does not support async")


def search(query):    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": query
    })

    headers = {
    'X-API-KEY': '0058d13f094639b9313b5c7a11779791e19f75b9',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

tools = [
        Tool(
            name = "Search",
            func = search,
            description = "useful for when you need to answer questions about current events, data. You should ask targeted questions"
        ),    
        ScrapeWebsiteTool(),
        # TwitterThreadTool()
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
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs = agent_kwargs,
    memory = memory,
    )


def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")
    
    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")    

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()