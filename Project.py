import ollama
import json
import pandas as pd
import requests
import streamlit as st
from Functions import *
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from Functions import *
from RAG_FAISS import *

chat_history = [AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database.")]
if 'memory' not in st.session_state:
    st.session_state.memory = {}

def store_memory(user_prompt: str, response: str):
    """Stores user query and response in memory."""
    st.session_state.memory[user_prompt] = response 
    return response

def retrieve_memory(user_prompt: str):
    """Retrieves previous conversation if available."""
    return st.session_state.memory.get(user_prompt, None)

# Initialize database connection
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Generate SQL query using LLM
def get_sql_chain(db):
    template = """
    You are a medical assistant in a hospital. You are interacting with a doctor who is asking you questions for details in the medical database.
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks or backslashes.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    For example:
    Question: Give me the medications of P001?
    SQL Query: SELECT Medications from medications where ID="P001";
    Question: Give me the name of P010.
    SQL Query: SELECT Name from patient_info where ID="P010";
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    GROQ_API_KEY = "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"
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0,api_key=GROQ_API_KEY)  # Using Groq LLM
   
    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Generate a natural language response
def get_response(user_query: str, db: SQLDatabase,chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
    You are a medical assistant in a hospital. You are interacting with a doctor who is asking you questions for details in the medical database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    GROQ_API_KEY = "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"
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY)  # Using Groq LLM
   

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"].strip())
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def get_database_details(prompt):
    global chat_history
    chat_history.append(HumanMessage(content=prompt))
    db = st.session_state.db
    response = get_response(prompt, db, chat_history)
    chat_history.append(AIMessage(content=response))
    return response  

# Define available functions
def agent_orchestrator(prompt):
    previous_response = retrieve_memory(prompt)
    if previous_response:
        return previous_response
    
    conversation_history = []
    if "messages" in st.session_state:
        conversation_history = st.session_state.messages
    
    # Convert Streamlit message history to a format usable for context
    formatted_history = ""
    for msg in conversation_history:
        formatted_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    # Include chat history in the prompt
    prompt_with_history = f"""
    Previous conversation:
    {formatted_history}
    
    Current question: {prompt}
    """
    
    safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    functions = [
            {
            "type": "function",
            "function": {
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
                }
            },
            {
            "type": "function",
            "function": {
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
                }
            },
            {
            "type": "function",
            "function": {
                    "name": "get_allergies",
                    "description": "Get allergies of the patient. Patient ID is required",
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
                }
            },
            {
            "type": "function",
            "function": {
                    "name": "get_complications",
                    "description": "Get complications of the patient. Patient ID is required",
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
                }
            },
            {
            "type": "function",
            "function": {
                    "name": "query_vector_database",
                    "description": "Get details of the drug from the vector database. ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Prompt": {
                                "type": "string",
                                "description": "Drug name from the user",
                            }
                        },
                        "required": ["Prompt"]
                    }
                }
            },
            # {
            # "type": "function",
            # "function": {
            #         "name": "get_drug_details",
            #         "description": "Get details of the drug. Drug name is required.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "drug_name": {
            #                     "type": "string",
            #                     "description": "Name of the drug",
            #                 }
            #             },
            #             "required": ["patient_id"]
            #         }
            #     }
            # },
            # {
            # "type": "function",
            # "function": {
            #         "name": "get_database_details",
            #         "description": "Get details from the database. Only use this for getting the details of patients from the database",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "prompt": {
            #                     "type": "string",
            #                     "description": "Prompt from the user",
            #                 }
            #             },
            #             "required": ["prompt"]
            #         }
            #     }
            # },
    ]

    available_functions = {
        "get_patient_info": get_patient_info,
        "get_current_stock": get_current_stock,
        "get_current_medications": get_current_medications,
        "get_allergies": get_allergies,
        "get_complications": get_complications,
        "query_vector_database": query_vector_database,
        # "get_drug_details": get_drug_details,
        # "get_database_details": get_database_details,
    }

    function_instructions = {
      "get_complications": "When providing the response from the get_complications function, if the function response is {'Complications': 'Migraine'} then return just the complication in a proper sentence. Don't explain about it.",
      "query_vector_database": "When the details of a drug is asked, respond with only the important details from the function response. Please provide information in the following structure:\n-Indication and Dosage:\n- Administration:\n- Overdosage:\n- Contraindications:\n- Special precautions:\n- Adverse Drug Reactions:\n- Drug Interactions:\n- Storage:\n- Mechanism of Action:\n- Brand name\nOnly use information from the function results. If information for a section is not found in the results, state 'No information available'.",
      "get_current_medications": "Provide a clear list of the patient's current medications based on the function results.",
      "get_allergies": "List the patient's allergies in a clear format.",
      "get_current_stock": "Provide the current stock information in a clear format.",
    # "get_database_details": "Provide a clear, complete answer based on the database query results.",
    }
 
    # Use Ollama to call a function
    response = ollama.chat(
      model="llama3.1:8b", # Change model if needed
      messages=[
                {"role": "system", "content": "You are an assistant. Don't use tools for general questions. Use them only to answer about drugs (or medicines) or patients"},
                {"role": "user", "content": prompt_with_history}],
      tools = functions,
      options={"temperature":0.0}
    )
    print(response)
    
    if 'message' in response and hasattr(response['message'], 'tool_calls') and response['message'].tool_calls:
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                if function_name in available_functions:
                    # Execute function and get response
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
                    elif function_name == "get_drug_details":
                        function_response = available_functions[function_name](
                            drug_name=function_args["drug_name"]
                        )
                    elif function_name == "get_database_details":
                        function_response = available_functions[function_name](
                            prompt=function_args["prompt"]
                        )
                    elif function_name == "query_vector_database":
                         function_response = available_functions[function_name](
                              drug_name=function_args["Prompt"]
                         )
                    
                    current_instruction = function_instructions.get(function_name,"")
                
                    # Store response messages
                    # msg.append({
                    #     "instruction": current_instruction,
                    #     "function": function_name,
                    #     "response": function_response
                    # })

                    final_response = ollama.chat(
                    model="llama3.1:8b",
                    messages=[
                    {"role": "system","content":current_instruction},
                    {"role": "system", "content": "You are a helpful assistant. Respond based on function results."},
                    {"role": "user", "content": f"Previous conversation:\n{formatted_history}\n\Original question: '{prompt}' \n\nI called the function {function_name} with arguments {function_args} and it returned: {function_response}. Use the function result to provide the response. "}
                    # {"role": "system","content":"Provide responses in proper sentences"},
                    ],
                    options={"temperature": 0.0}
                    )
                    store_memory(prompt, final_response["message"]["content"])
                    return final_response["message"]["content"]
            
            # If we reached here without returning, no valid function calls were made
            return response["message"]["content"]
    else:
            # No function calls, just return the response
            return response["message"]["content"]
    
# def main():
#     print("Welcome to the AI Agent. Type 'exit' to quit.")
#     while True:
#         user_input = input("[User Prompt]: ")
#         if user_input.lower() == 'exit':
#             break
#         response = agent_orchestrator(user_input)
#         print(f"[Agent]: {response}")

# if __name__ == "__main__":
#     main()


st.set_page_config(page_title="AI Agent", page_icon=":robot:")
# st.title("AI Agent")
st.markdown("<h1 style='text-align: center;'>AI Agent</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")

    with st.expander("Database Connection", expanded=False):
      st.text_input("Host", value="localhost", key="Host")
      st.text_input("Port", value="3306", key="Port")
      st.text_input("User", value="root", key="User")
      st.text_input("Password", type="password", value="Paulmon123", key="Password")
      st.text_input("Database",  value="Project", key="Database")
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
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
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