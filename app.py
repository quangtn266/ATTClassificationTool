import os

from dotenv import load_dotenv
import streamlit as st
from user_utils import *


# Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] = []
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] = []
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] = []


def main():
    load_dotenv('.env')
    Pinecone_API_Key = os.environ.get("Pinecone_API_Key")
    Pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
    Pinecone_index = os.environ.get("PINECONE_INDEX")

    st.header("Automatic Ticket Classification Tool")

    # Capture usewr input
    st.write("We are here to help you, please ask your question: ")
    user_input = st.text_input("⌨️")

    if user_input:
        # create embeddings instance.
        embeddings = create_embeddings()

        # Function to pull index data from Pinecone
        index = pull_from_pinecone(Pinecone_API_Key, Pinecone_environment, Pinecone_index, embeddings)

        # This function will help us in fetching the top relevant documents from vector store
        # - Pinecone index
        relevant_docs = get_similar_docs(index, user_input)

        # This will return the fine-tuned respoonse by LLM
        response = get_answer(relevant_docs, user_input)
        st.write(response)

        # Button to create a ticket with respective department
        button = st.button("Please, submit ticket")

        if button:
            # Get Response
            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            # Loading the trained machine learning model, so that we can use it to predict the class
            # to which this compliant belongs to...
            department_value = predict(query_result)
            st.write("Ur ticket has been submitted to : ", department_value)

            # Appending the tickets to below list, so that we can view/ use them later on...
            if department_value == "HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value == "IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transportation_tickets'].append(user_input)

if __name__ == '__main__':
    main()