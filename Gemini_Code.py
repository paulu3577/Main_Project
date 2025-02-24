

import google.generativeai as genai
import json
import pandas as pd
import requests
import streamlit as st
from langgraph.graph import Graph
from LLM_MySQL import *


# Configure the API key
genai.configure(api_key="AIzaSyBMFMAJp1UNf5PHJ0Z4-HBJIpFS0YsIYxM")

agent_graph = Graph()
memory = {}

def store_memory(user_prompt: str, response: str):
    """Stores user query and response in memory."""
    memory[user_prompt] = response
    return response

def retrieve_memory(user_prompt: str):
    """Retrieves previous conversation if available."""
    return memory.get(user_prompt, None)

def get_patient_info(patient_id):
    df = pd.read_csv("patient_info.csv",dtype={"Patient_ID": str})
    print(df)
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
  
def get_drug_details(drug_name):
    base_url = "https://api.fda.gov/drug/label.json"
    api_key = "xS2omrNfbJ0eRJdFCJ0oULCS7hHuCOYjeGqu1yJt"
    # search_query = f'generic_name:"{drug_name}"'
    search_query = "Aspirin"

    url = f"{base_url}?api_key={api_key}&search={search_query}"
    response = requests.get(url)
    print(response.text)
    if response.status_code == 200:
      return json.dumps({"Drug details":response})
    else:
      return json.dumps({"Drug details":response})
    
def get_database_details(prompt):
    response  = initial_function(prompt)
    return response
  

def agent_orchestrator(prompt):
    # Create model instance
    model = genai.GenerativeModel('gemini-1.0-pro')
    # Check if the query is in memory
    previous_response = retrieve_memory(prompt)
    if previous_response:
        return previous_response
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
                {
                    "name": "get_patient_info",
                    "description": "Get name of the patient. Patient ID is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "ID of the patient",
                            }
                        },
                        "required": ["patient_id"]
                    },
                },
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
                {
                    "name": "get_drug_details",
                    "description": "Get details of the drug. Drug name is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "Name of the drug",
                            }
                        },
                        "required": ["drug_name"]
                    },
                },
                {
                    "name": "get_database_details",
                    "description": "Get details from the database. Prompt is required",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Prompt from the user",
                            }
                        },
                        "required": ["prompt"]
                    },
                },
            ]
        }
    ]
    
    available_functions = {
        "get_patient_info": get_patient_info,
        "get_current_stock": get_current_stock,
        "get_current_medications": get_current_medications,
        "get_allergies": get_allergies,
        "get_complications": get_complications,
        "get_drug_details": get_drug_details,
        "get_database_details": get_database_details,
        
    }

    # # Initialize LangChain memory for storing the conversation
    # memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

    # # Check if the query is in memory
    # if prompt in memory.load_memory_variables({}):
    #     return memory.load_memory_variables({})[prompt]
    
    # Generate initial response
    response = model.generate_content(
        prompt,
        tools = tools,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        ),   
    )
    print(response)

    # Process function calls
    if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call'):
                function_name = part.function_call.name
                function_args = part.function_call.args

                if function_name in available_functions:
                    # Execute function and get response
                    if function_name == "get_patient_info":
                        function_response = available_functions[function_name](
                            patient_id=function_args["patient_id"]
                        )                  
                    elif function_name == "get_current_stock":
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
                    elif function_name == "get_drug_details":
                        function_response = available_functions[function_name](
                            drug_name=function_args["drug_name"]
                        )
                    elif function_name == "get_database_details":
                        function_response = available_functions[function_name](
                            prompt=function_args["prompt"]
                        )
                    final_response = model.generate_content(
                        [  
                            {"text":""" Return only the complication. Example :  if the function response is {"Complications": "Migraine"} then return just the complication in a proper sentence. Don't explain about it."""},
                            {"text":"""When the details of a drug is asked, respond with only the important details(dosage, sideffects, interactions, warnings etc.) from the response from the API"""},
                            {"text": prompt},
                            {
                                "text": f"Function {function_name} returned: {function_response}"
                            },
                        ],
                             generation_config=genai.types.GenerationConfig(
            temperature=0.0),
                            # safety_settings= safe,
                    )
                    store_memory(prompt, final_response.text)
                    return final_response.text
                
                else:
                    final_response = model.generate_content(
                        [
                            {"text": prompt},
                        ],  generation_config=genai.types.GenerationConfig(
                            temperature=0.0
                            ),
                )
                    return final_response.text
    
    return response.text

# def main():
#     print("Welcome to the Gemini Agent. Type 'exit' to quit.")
#     while True:
#         user_input = input("[User Prompt]: ")
#         if user_input.lower() == 'exit':
#             break
#         response = agent_orchestrator(user_input)
#         print(f"[Agent]: {response}")

# if __name__ == "__main__":
#     main()

# Check the memory here itself..No need to go to the agent
    
def main():
    st.set_page_config(page_title="AI Agent", page_icon=":robot:")
    st.title("AI Agent for Medical Assistance")

    with st.sidebar:
      st.subheader("Settings")
    #   st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
      st.text_input("Host", value="localhost", key="Host")
      st.text_input("Port", value="3306", key="Port")
      st.text_input("User", value="root", key="User")
      st.text_input("Password", type="password", value="admin", key="Password")
      st.text_input("Database", value=" ", key="Database")
    
      if st.button("Connect"):
          with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        response = agent_orchestrator(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()





