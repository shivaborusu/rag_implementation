from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import os
import yaml
from dotenv import load_dotenv
load_dotenv()


def load_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


def get_vector_store_client():
    qdrant_client = QdrantClient(url=os.environ["QDRANT_URI"])

    return qdrant_client


def collection_exists(qdrant_client, collection_name):
    return qdrant_client.collection_exists(collection_name=collection_name)
        

def create_collection(qdrant_client, collection_name):
    qdrant_client.create_collection(collection_name=collection_name,
                                    vectors_config=VectorParams(size=1024,
                                                                distance=Distance.COSINE))
    
    return "Success"


def get_embedder():
    model_kwargs = {'device': 'cpu'}  
    encode_kwargs = {'normalize_embeddings': True}  
    config = load_config()
    embedder = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"],
                                        model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs)
    
    return embedder


def get_llm():
    config = load_config()
    llm = ChatGroq(**config["llm"])
    
    return llm

def get_llm_judge():
    config = load_config()
    llm = ChatOllama(**config["llm_judge"])
    
    return llm


def get_prompt(user_question, context):
    prompt_template = """You are an useful assistant, 
    Answer the question based on the context provided.
    If no context provided, just say I don't have enough context to answer.
    Question:{user_question}
    Context: {context}
    Limit response to 5 sentences.
    """

    prompt = PromptTemplate.from_template(prompt_template)

    return prompt.invoke({"user_question":user_question,
                          "context":context})


