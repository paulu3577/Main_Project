import requests
import re
from bs4 import BeautifulSoup

def drug_details(name):
  api_key = "AIzaSyDBwiSDcnwHXlzqo2IEZXc2pes25P9nJeY"
  search_engine_id = "b64a85c77b5b84273"
  query = name
  url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"

  response = requests.get(url)
  results = response.json()
  results_items = results.get("items", [])
  for item in results.get("items", []):
    print(item["link"])
  first_item = results_items[0]
  link = first_item["link"]
  print(link)
  result = requests.get(link)
  s=BeautifulSoup(result.text,'html.parser')
  result = requests.get(link)
  tag = 'div'
  content = s.find(tag,{'id':"content"})
  # print(content.text)
  return content.text
    
drug_details("Aspirin")