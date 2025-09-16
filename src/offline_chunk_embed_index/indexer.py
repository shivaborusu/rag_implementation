from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import mlflow.langchain as mlflow_langchain
import mlflow
from datetime import datetime
import time
from utils.vector_utils import (get_vector_store_client, collection_exists,
                                create_collection, load_config, get_embedder)
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_SERVER"])

class Indexer():
    def __init__(self):
        pass

    def index(self, pdf_file_path, collection_name):
        """
            Receives a single pdf to chunk, embed and index

        """
        
        config = load_config()
        mlflow.set_experiment(config["mlflow"]["index_exp_name"])

        with mlflow.start_run(run_name=f"indexing_{datetime.now().isoformat()}"):
            pdf_loader = self._get_pdf_loader(pdf_file_path)
            documents = pdf_loader.load()
            documents = self._chunk_documents(documents,
                                            config["offline"]["chunking"])
            vector_store_client = get_vector_store_client()
            embedder = get_embedder()
            mlflow.log_param("embedding_model", config["embedding"]["model_name"])

            status = self._add_documents_to_collection(documents,
                                            vector_store_client,
                                            collection_name,
                                            embedder)

    def _get_pdf_loader(self, pdf_file_path):
        logger.info("Getting PDF Loader")
        pdf_loader = PyPDFLoader(file_path=pdf_file_path,
                                extract_images=False,
                                mode='page')
        
        return pdf_loader

    def _chunk_documents(self, documents, chunk_config):
        logger.info("Chunking Documents, Count: %d", len(documents))
        text_splitter = RecursiveCharacterTextSplitter(**chunk_config)
        documents = text_splitter.split_documents(documents=documents)
        logger.info("Chunking Complete, Count: %d", len(documents))

        return documents
    

    def _add_documents_to_collection(self, documents,
                                     vector_store_client,
                                     collection_name,
                                     embedder):
        
        logger.info("Adding documents to vector store")
        if not collection_exists(vector_store_client, collection_name):
            create_collection(vector_store_client, collection_name)
        
        qdrant_vector_store = QdrantVectorStore(
                                client=vector_store_client,
                                collection_name=collection_name,
                                embedding=embedder)
        try:
            start = time.time()
            qdrant_vector_store.add_documents(documents)
            elapsed_time = (time.time() - start)
            mlflow.log_metric("count_of_embed_docs", len(documents))
            mlflow.log_metric("embedding_time", elapsed_time)
            logger.info("Adding documents to vector store: Successful")
        except Exception as e:
            logger.info("Exception while adding documents to vector store, {e}", exc_info=True)

        return True
        

test_file = "/Users/shivaborusu/Development/Repos/rag_implementation/.data/ds_interview.pdf"
Indexer().index(test_file, "pdf_documents")