from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import mlflow
import mlflow.langchain as mlflow_langchain
from retriever import Retriever
from utils.vector_utils import get_prompt, load_config
from utils.logger import get_logger
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_SERVER"])

class RetAugGen():
    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        mlflow_langchain.autolog()
        self.config = {}

    def get_response(self,query):
        self.config = load_config()
        mlflow.set_experiment(self.config["mlflow"]["retrieval_exp_name"])
        with mlflow.start_run(run_name=f"rag_{datetime.now().strftime("%m_%d_%Y")}"):
            self.logger.info("Starting RAG")
            context, _ = self._retrieve(query)

            aug_prompt = self._augment(query, context)

            response = self._generate(aug_prompt)
            self.logger.info("Successful response generation")

            return response


    def _retrieve(self, query):
        self.logger.info("Starting document retrieval")
        try:
            results = Retriever().search(query)

            context = [doc.page_content for doc in results]
            metadata = [doc.metadata for doc in results]

            return context, metadata
        except Exception as e:
            self.logger.error("An error occured while fetching the documents", exc_info=True)
            raise


    def _augment(self, query, context):
        self.logger.info("Augmenting...")
        prompt = get_prompt(query, context)

        return prompt
    
    def _generate(self, aug_prompt):
        self.logger.info("Generating the response")
        llm = ChatGroq(**self.config["online"]["llm"])

        ai_response = llm.invoke(aug_prompt)

        response = StrOutputParser().invoke(ai_response)

        return response
    