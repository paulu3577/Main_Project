import requests
import re
from bs4 import BeautifulSoup

def get_interaction(drug_name_1,drug_name_2):
  api_key = "AIzaSyDBwiSDcnwHXlzqo2IEZXc2pes25P9nJeY"
  search_engine_id = "601f84ab1efdb4723"
  # query = f"Medication for {name}"
  query = f"Interaction between drugs {drug_name_1} and {drug_name_2}"
  url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"

  response = requests.get(url)
  results = response.json()
  results_items = results.get("items", [])
  for item in results.get("items", []):
    # print(item["title"])
    print(item["link"])
    # print(item["snippet"])
  first_item = results_items[0]
  print(first_item["link"])
  result = requests.get(first_item["link"])
  html_content = result.text
  soup = BeautifulSoup(html_content, 'html.parser')
  tag = 'div'
  content = soup.find(tag,{'class':"interactions-reference"})
  status = content.find("span").text
  main_content = content.find_all("p")[0].text
  interaction_data = {
    "status": status,
    "warning": main_content
  }  
# print(interaction_data)
  return(interaction_data)
#   print(header)
#   print(text)
# print(get_interaction("Aspirin","Ibuprofen"))





# string = ''
# for _ in text:
#     string+=_.text
# print(string)
# for item in results.get("items", []):
#     print(item["title"])
#     print(item["link"])
#     print(item["snippet"])