from Functions import *
from Drug_Interactions import *

def interaction(patient_id,drug_name):

    data_string = get_current_medications(patient_id)
    data = json.loads(data_string)
    drug_name_1 = data["Current Medication"]
    result = get_interaction(drug_name,drug_name_1)
    return {
        "patient_id": patient_id,
        "Current medication": drug_name,
        "interaction_result": result     
    }

# print(interaction("P100","ibuprofen"))