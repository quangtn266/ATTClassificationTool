import os
import streamlit as st
from dotenv import load_dotenv
from pages.admin_utils import *

def main():
    load_dotenv()

    Pinecone_API_Key = os.environ.get("Pinecone_API_Key")
    Pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
    Pinecone_index = os.environ.get("PINECONE_INDEX")

    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your files...📇")

    # Upload the pdf file...
    pdf = st.file_uploader("Only PDF files allowed", type=['pdf'])

    # Extract the whole text from the uploaded pdf file
    if pdf is not None:
        with st.spinner('Wait for  it....'):
            text = read_pdf_data(pdf)
            st.write("🫵 Reading PDF done")

            # Create chunks
            docs_chunks = split_data(text)
            # st.write(docs_chunks)
            st.write("🫵 Splitting data into chunks done")

            # Create the embeddings
            embeddings = create_embeddings_load_data()
            st.write("🫵 Creating embeddings instance done")

            # Build the vector store (Push the PDF data embeddings)
            push_to_pinecone(Pinecone_API_Key, Pinecone_environment,
                Pinecone_index, "tickets", embeddings, docs_chunks)

        st.success("Succesfully pushed the embeddings to Pinecone")

if __name__ == "__main__":
    main()