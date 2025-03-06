import json
import pandas as pd
import requests

def get_patient_info(patient_id):
    df = pd.read_csv("patient_info.csv",dtype={"Patient_ID": str})
    # print(df)
    if patient_id in df["Patient_ID"].values:
      name = df.loc[df["Patient_ID"] == patient_id, "Patient Name"].values[0]
      return json.dumps({"Patient name":name})
    else:
      return json.dumps({"Patient name":"unknown"})
  
def get_current_medications(patient_id):
    df = pd.read_csv("medications.csv",dtype={"Patient_ID": str})
    print(df)
    if patient_id in df["Patient_ID"].values:
      Medication = df.loc[df["Patient_ID"] == patient_id, "Medications"].values[0]
      return json.dumps({"Current Medication":Medication})
    else:
      return json.dumps({"Current Medication":"unknown"})

def get_allergies(patient_id):
    df = pd.read_csv("allergies.csv")
    # print(df)
    if patient_id in df["Patient_ID"].values:
     allergy = df.loc[df["Patient_ID"] == patient_id, "Allergies"].values[0]
     return json.dumps({"Allergies":allergy})
    else:
     return json.dumps({"Allergies":"unknown"})
    
def get_complications(patient_id):
    df = pd.read_csv("complications.csv")
    # print(df)
    if patient_id in df["Patient_ID"].values:
     complication = df.loc[df["Patient_ID"] == patient_id, "Complications"].values[0] 
     return json.dumps({"Complications":complication})
    else:
     return json.dumps({"Complications":"unknown"})
    
def get_current_stock(name):
    df = pd.read_csv("Medicine_list.csv")
    # print(df)
    if name.lower() in df["Name"].str.lower().values:
      quantity = int(df.loc[df["Name"].str.lower() == name.lower(), "Quantity"].values[0])
      return json.dumps({"quantity":quantity})
    else:
      return json.dumps({"quantity":"unknown"})
  
# def get_drug_details(drug_name):
#     base_url = "https://api.fda.gov/drug/label.json"
#     api_key = "xS2omrNfbJ0eRJdFCJ0oULCS7hHuCOYjeGqu1yJt"
#     # search_query = f'generic_name:"{drug_name}"'
#     search_query = "Aspirin"
#     url = f"{base_url}?api_key={api_key}&search={search_query}"
#     response = requests.get(url)
#     print(response.text)
#     if response.status_code == 200:
#       return json.dumps({"Drug details":response})
#     else:
#       return json.dumps({"Drug details":response})
    
