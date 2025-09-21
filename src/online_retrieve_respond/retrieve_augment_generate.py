from langchain_core.output_parsers import StrOutputParser
from retriever import Retriever
from utils.vector_utils import get_prompt
from utils.logger import get_logger
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class RetAugGen():
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def get_response(self,query):
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
        llm = ChatGroq(model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=200)

        ai_response = llm.invoke(aug_prompt)

        response = StrOutputParser().invoke(ai_response)

        return response
    