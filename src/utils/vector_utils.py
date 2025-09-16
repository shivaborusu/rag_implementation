from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
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



