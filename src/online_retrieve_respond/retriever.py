import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from langchain_qdrant import QdrantVectorStore
from utils.vector_utils import (get_vector_store_client,
                                load_config, get_embedder)
from utils.logger import get_logger
from dotenv import load_dotenv
load_dotenv()

class Retriever:
    def __init__(self) -> None:
        pass

    def search(self, query, num_res=4):

        config = load_config()
        embedder = get_embedder()
        qdrant_client = get_vector_store_client()
        collection_name = config["vector_store"]["collection_name"]
        results = self._get_search_results(qdrant_client,
                                           collection_name,
                                           embedder,
                                           query,
                                           num_res)

        return results
    


    def _get_search_results(self, qdrant_client,
                            collection_name, embedder,
                            query, num_res):
        
        qdrant_vector_store = QdrantVectorStore(
                        client=qdrant_client,
                        collection_name=collection_name,
                        embedding=embedder)
        
        results = qdrant_vector_store.similarity_search(query=query,
                                                        k=num_res)
        
        return results

