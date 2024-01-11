import os

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import joblib
from dotenv import load_dotenv

load_dotenv(".env")
model_ChatGPT = os.environ.get("MODEL_CHATGPT")
trained_model = os.environ.get("TRAINED_MODEL")


# Function to pull index data from Pinecone...
def pull_from_pinecone(pinecone_apikey, pinecone_environment,
                       pinecone_index_name, embeddings):
    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_environment,
    )

    index_name = pinecone_index_name
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name=model_ChatGPT)
    return embeddings


# This function will help us in fetching the top relevant documents from our vector store
# - Pinecone Index
def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


def get_answer(docs, user_input):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response


def predict(query_result):
    Fitmodel = joblib.load(trained_model)
    result = Fitmodel.predict([query_result])
    return result[0]
