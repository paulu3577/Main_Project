
import google.generativeai as genai
import json
import pandas as pd

# Configure the API key
genai.configure(api_key="..............................")

# def get_patient_info(patient_id):
#   df = pd.read_csv("patient_info.csv",dtype={"Patient_ID": str})
#   print(df)
#   if patient_id in df["Patient_ID"].values:
#     name = df.loc[df["Patient_ID"] == patient_id, "Patient Name"].values[0]
#     return json.dumps({"Patient name":name})
#   else:
#     return json.dumps({"Patient name":"unknown"})
  
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
    print(df)
    if patient_id in df["Patient_ID"].values:
     allergy = df.loc[df["Patient_ID"] == patient_id, "Allergies"].values[0]
     return json.dumps({"Allergies":allergy})
    else:
     return json.dumps({"Allergies":"unknown"})
    
def get_complications(patient_id):
    df = pd.read_csv("complications.csv")
    print(df)
    if patient_id in df["Patient_ID"].values:
     complication = df.loc[df["Patient_ID"] == patient_id, "Complications"].values[0] 
     return json.dumps({"Complications":complication})
    else:
     return json.dumps({"Complications":"unknown"})
    
def get_current_stock(name):
  df = pd.read_csv("Medicine_list.csv")
  print(df)
  if name.lower() in df["Name"].str.lower().values:
    quantity = int(df.loc[df["Name"].str.lower() == name.lower(), "Quantity"].values[0])
    return json.dumps({"quantity":quantity})
  else:
    return json.dumps({"quantity":"unknown"})
  
# def get_current_weather(location, unit="celsius"):
#     """Get the current weather in a given location"""
#     if "tokyo" in location.lower():
#         return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
#     elif "rome" in location.lower():
#         return json.dumps({"location": "Rome", "temperature": "72", "unit": unit})
#     elif "paris" in location.lower():
#         return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
#     else:
#         return json.dumps({"location": location, "temperature": "unknown"})

# def get_vacation(user):
#     """Get vacation days for a user"""
#     if "john" in user.lower():
#         return json.dumps({"vacation": "5 days"})
#     if "mary" in user.lower():
#         return json.dumps({"vacation": "10 days"})
#     if "bob" in user.lower():
#         return json.dumps({"vacation": "20 days"})
#     else:
#         return json.dumps({"vacation": "first tell me your name?"})

def agent_orchestrator(prompt):
    # Create model instance
    model = genai.GenerativeModel('gemini-1.0-pro')
    # safe = [
    #     {
    #         "category": "HARM_CATEGORY_DANGEROUS",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_HARASSMENT",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_HATE_SPEECH",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    #         "threshold": "BLOCK_NONE",
    #     },
    #     {
    #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    #         "threshold": "BLOCK_NONE",
    #     },
    # ]
    
    # Define function declarations
    tools = [
        {
            "function_declarations": [
                # {
                #     "name": "get_current_weather",
                #     "description": "Get current weather for a city. City name is required",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "location": {
                #                 "type": "string",
                #                 "description": "Location",
                #             },
                #             "unit": {
                #                 "type": "string",
                #                 "enum": ["celsius", "fahrenheit"]
                #             }
                #         },
                #         "required": ["location"]
                #     }
                # },
                # {
                #     "name": "get_vacation",
                #     "description": "Get vacation in days for a user. User name is required",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "user": {
                #                 "type": "string",
                #                 "description": "name of the user",
                #             }
                #         },
                #         "required": ["user"]
                #     }
                # },
                # {
                #     "name": "get_patient_info",
                #     "description": "Get name of the patient. Patient ID is required",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "patient_id": {
                #                 "type": "string",
                #                 "description": "ID of the patient",
                #             }
                #         },
                #         "required": ["patient_id"]
                #     },
                # },
                {
                    "name": "get_current_stock",
                    "description": "Get current stock of the medicine. Medicine name is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the medicine ",
                            }
                        },
                        "required": ["name"]
                    }
                },
                {
                    "name": "get_current_medications",
                    "description": "Get current medication of the patient. Patient ID is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "patient id",
                            }
                        },
                        "required": ["patient_id"]
                    }
                },
                {
                    "name": "get_allergies",
                    "description": "Get allergies of the patient. Patient ID is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "ID of the patient",
                            }
                        },
                        "required": ["patient_id"]
                    }
                },
                {
                    "name": "get_complications",
                    "description": "Get complications of the patient. Patient ID is required.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "ID of the patient",
                            }
                        },
                        "required": ["patient_id"]
                    }
                },
            ]
        }
    ]
    
    available_functions = {
        # "get_current_weather": get_current_weather,
        # "get_vacation": get_vacation,
        "get_current_stock": get_current_stock,
        "get_current_medications": get_current_medications,
        "get_allergies": get_allergies,
        "get_complications": get_complications,
        # "get_patient_info": get_patient_info,
    }
    
    # Generate initial response
    response = model.generate_content(
        prompt,
        tools = tools,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        )
    )

    # Process function calls
    if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call'):
                function_name = part.function_call.name
                function_args = part.function_call.args
                if function_args==None:
                    response = model.generate_content(
        prompt,
        )
                if function_name in available_functions:
                    # Execute function and get response
                    # if function_name == "get_patient_info":
                    #     function_response = available_functions[function_name](
                    #         name=function_args["patient_id"]
                    #     )                  
                    # if function_name == "get_current_weather":
                    #     function_response = available_functions[function_name](
                    #         location=function_args["location"],
                    #         unit=function_args.get("unit", "celsius")
                    #     )
                    # elif function_name == "get_vacation":
                    #     function_response = available_functions[function_name](
                    #         user=function_args["user"]
                    #     )
                    if function_name == "get_current_stock":
                        function_response = available_functions[function_name](
                            name=function_args["name"]
                        )
                    elif function_name == "get_current_medications":
                        function_response = available_functions[function_name](
                            patient_id=function_args["patient_id"]
                        )
                    elif function_name == "get_allergies":
                        function_response = available_functions[function_name](
                            patient_id=function_args["patient_id"]
                        )
                    elif function_name == "get_complications":
                        function_response = available_functions[function_name](
                            patient_id=function_args["patient_id"]
                        )
                    # Generate final response using function result
                    # print(function_response)
                    final_response = model.generate_content(
                        [   {"text":""" Return only the complication. Example :  if the function response is {"Complications": "Migraine"} then return just the complication in a proper sentence. Don't explain about it."""},
                            {"text": prompt},
                            {
                                "text": f"Function {function_name} returned: {function_response}"
                            },
                        ],
                             generation_config=genai.types.GenerationConfig(
            temperature=0.0),
                            # safety_settings= safe,
                    )
                    # print(final_response)
                    return final_response.text
                
                else:
                    final_response = model.generate_content(
                        [
                            {"text": prompt},

                        ]
                    )
                    return final_response.text
    
    return response.text

def main():
    print("Welcome to the Gemini Agent. Type 'exit' to quit.")
    while True:
        user_input = input("[User Prompt]: ")
        if user_input.lower() == 'exit':
            break
        response = agent_orchestrator(user_input)
        print(f"[Agent]: {response}")

if __name__ == "__main__":
    main()


