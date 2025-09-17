from retriever import Retriever
from dotenv import load_dotenv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
load_dotenv()

results = Retriever().search(query="what is a lakehouse")
print(results)
