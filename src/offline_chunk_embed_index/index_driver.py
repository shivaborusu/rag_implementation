from indexer import Indexer
import os
import mlflow
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_SERVER"])

test_file = "/Users/shivaborusu/Development/Repos/rag_implementation/.data/ds_interview.pdf"
Indexer().index(test_file, "pdf_documents")