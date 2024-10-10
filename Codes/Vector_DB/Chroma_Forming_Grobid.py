# import
from langchain_community.vectorstores import Chroma
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import openai
from openai import OpenAI
import pickle


def load_documents(filename):
    with open(filename, 'rb') as file:
        documents = pickle.load(file)
    return documents

# Example usage
docs = load_documents('Document_list_grobid.pkl')
print(len(docs))
cleaned_docs = filter_complex_metadata(docs)

embedding_path = 'sentence-transformers/all-mpnet-base-v2'

encode_kwargs = {'normalize_embeddings': True}

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_path,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(cleaned_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_grobid")
print(vectorstore)