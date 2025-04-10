import requests
import re
from bs4 import BeautifulSoup

def get_medication_details(name):
  api_key = "AIzaSyDBwiSDcnwHXlzqo2IEZXc2pes25P9nJeY"
  search_engine_id = "601f84ab1efdb4723"
  query = f"Medication for {name}"
  url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"

  response = requests.get(url)
  results = response.json()
  results_items = results.get("items", [])
  first_item = results_items[0]
  print(first_item["link"])
  link = first_item["link"]+"?page_all=1"
  result = requests.get(link)
# s=BeautifulSoup(result.text,'html.parser')
# tag = 'a'
# content = s.find(tag,{'class':"ddc-paging-item ddc-paging-show-all"})
# print(content.get('href'))
# link = content.get('href')
# print(link)
  html_content = result.text
  soup = BeautifulSoup(html_content, 'html.parser')
  medication_table = soup.find('table', class_='ddc-table-secondary ddc-table-sortable')
  medications = []
  if medication_table:
        # Get all medication rows (they have class "ddc-table-row-medication data-table-row")
        medication_rows = medication_table.find_all('tr', class_='ddc-table-row-medication')
                
        # Limit to first 10 medications
        for i, med_row in enumerate(medication_rows):
            if i >= 10:  # Only get the first 10 medications
                break
                
            med_name_element = med_row.find('a', class_='ddc-text-wordbreak')  # Get the medication name
            if med_name_element:
                med_name = med_name_element.text.strip()
            else:
                med_name = "N/A"
                
            rating_element = med_row.find('span', class_='ddc-text-nowrap') # Get the rating
            if rating_element and 'title' in rating_element.attrs:
                rating = rating_element.text.strip()
            else:
                rating = "N/A"
                
            # Extract activity percentage from the style attribute
            activity_div = med_row.find('div', class_='ddc-rating-bar')
            if activity_div:
                activity_element = activity_div.find('div')
                if activity_element and 'style' in activity_element.attrs:
                    # Extract percentage from style="width:XX%;"
                    style = activity_element.get('style')
                    activity = style.split('width:')[1].split(';')[0]
                else:
                    activity = "N/A"
            else:
                activity = "N/A"
                
            medications.append({
                'name': med_name,
                'rating': rating,
                'activity': activity,
                # 'link' : first_item["link"]
            })
            
            print(f"{med_name:<30} {rating:<10} {activity:<10}")
  else:
        print("Could not find the medication table in the provided HTML.")
  return medications
# print(get_medication_details("Cholera"))  




# return(first_item["link"])
# Process the results
# for item in results.get("items", []):
#     print(item["title"])
#     print(item["link"])
#     print(item["snippet"])


