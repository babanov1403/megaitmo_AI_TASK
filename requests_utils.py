import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

from ml_tools import IsAnswerInText

# return a list of urls
def GetURLs(query: str, tavily_token):
    exclude_domains = [
        "vuzopedia.ru",
        "cn.itmo.ru",
        "wp.wiki-wiki.ru"
    ]
    client = TavilyClient(api_key=tavily_token)
    response = client.search(
        query=query,
        max_results=5,
        exclude_domains=exclude_domains)['results']

    urls = []

    for info_dic in response:
        urls.append(info_dic['url'])
    print(urls)
    return urls

def NormalizeString(string: str):
    string = string.replace("«", "")
    string = string.replace("»", "")
    return string

def GetInfoFromURL(url: str, question, model):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers = headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(strip=True)
        # paragraphs = [p.get_text() for p in soup.find_all('p')]
        # string = ""
        # idx = 0
        # for paragraph in paragraphs:
        #     string += paragraph + ".\n"
                
        
        # return string
    else:
        return None