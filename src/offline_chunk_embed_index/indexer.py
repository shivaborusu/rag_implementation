from langchain_community.document_loaders import PyPDFLoader
from utils.vector_utils import collection_exists
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
import yaml
from utils.vector_utils import collection_exists, create_collection
from utils.logger import get_logger
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Indexer():
    def __init__(self):
        pass

    def index(self, pdf_file_path, collection_name):
        """
            Receives a single pdf or a list of pdf files to chunk, embed and index

        """
        
        config = self._get_config()

        pdf_loader = self._get_pdf_loader(pdf_file_path)
        documents = pdf_loader.load()
        print("CHUNK_CONF: ", config["offline"]["chunking"])

        documents = self._chunk_documents(documents,
                                          config["offline"]["chunking"])
        
        sys.exit(0)
        vector_store_client = self._get_vector_store_client()
        embedder = self._get_embedder(config["offline"]["embedding"]["model_name"])

 

        status = self._add_documents_to_collection(documents,
                                          vector_store_client,
                                          collection_name,
                                          embedder)


    def _get_config(self):
        logger.info("Fetching the config")
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        return config


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


    def _get_vector_store_client(self):
        logger.info("Getting Vector Store Client")
        qdrant_client = QdrantClient(url=os.environ["QDRANT_URI"])

        return qdrant_client
    

    def _get_embedder(self, model_name):
        logger.info("Getting Embedding Config")
        model_kwargs = {'device': 'cpu'}  
        encode_kwargs = {'normalize_embeddings': True}  
        embedder = HuggingFaceEmbeddings(model_name=model_name,
                                         model_kwargs=model_kwargs,
                                         encode_kwargs=encode_kwargs)
        
        return embedder

    

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
            qdrant_vector_store.add_documents(documents)
            logger.info("Adding documents to vector store: Successful")
        except Exception as e:
            logger.info("Exception while adding documents to vector store, {e}", exc_info=True)

        return True
        

test_file = "/Users/shivaborusu/Development/Repos/rag_implementation/.data/ds_interview.pdf"
Indexer().index(test_file, "pdf_documents")