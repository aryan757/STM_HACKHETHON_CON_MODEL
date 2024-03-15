import os
#from constant import openai_key
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
#df = pd.read_csv("C:\\Users\\aryan\\Downloads\\STM_R_and_D\\updated_data.csv")
df = pd.read_csv("./dataset_gps.csv")
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

os.getenv("OPENAI_API_KEY")


def process_query(query):
    # Here, you can implement your logic to process the query
    # For simplicity, I'm just returning the query as it is
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    out_pass = agent(query)
    out_pass1 = out_pass['output']

    return out_pass1

def main():
    st.title("ST_Statistical_Query_Conversational_Model")

    # Input field for query
    query = st.text_input("Enter your query:")

    # Button to trigger search
    if st.button("Result"):
        result = process_query(query)

        # Display the result
        st.write("Search Results:")
        st.write(result)


if __name__ == "__main__":
    main()



