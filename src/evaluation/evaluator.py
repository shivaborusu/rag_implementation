import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.vector_utils import get_embedder, get_llm_judge
from utils.pg_utils import get_eval_data, add_eval_metrics
from datetime import datetime
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import EvaluationDataset
from utils.logger import get_logger


class Evaluator:
    def __init__(self):
        self.logger = get_logger(__name__)

    def evaluate(self, data_cutoff=datetime.now().date()):
        self.logger.info("Starting RAG Evaluations")
        
        self.logger.info("Fetching evaluation data")
        eval_data = get_eval_data(data_cutoff)

        self.logger.info("Getting LLM Judge and Embedding Model")
        evaluator_llm = LangchainLLMWrapper(get_llm_judge())
        embedder = get_embedder()

        self.logger.info("Evaluating...")
        evaluation_results = self._evaluate(eval_data, evaluator_llm, embedder)
        self.logger.info("Evaluation complete")


        add_eval_metrics(evaluation_results)
        self.logger.info("Stored eval results to the database")


        return "Completed"

    
    def _evaluate(self, eval_data, evaluator_llm, embedder):
        evaluation_results = []
        for row in eval_data:
            dataset= []
            dataset.append(
                {
                    "user_input": row[1],
                    "retrieved_contexts": row[2],
                    "response": row[3],
                    "reference": row[2][0]
                }
            )

            evaluation_dataset = EvaluationDataset.from_list(dataset)

            result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=[Faithfulness(), ResponseRelevancy()],
                        llm=evaluator_llm,
                        embeddings=embedder
                    )._repr_dict

            dataset.clear()

            evaluation_results.append((row[0],
                                       result.get('faithfulness'),
                                       result.get('answer_relevancy'))
                                    )
            
        return evaluation_results

