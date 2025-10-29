import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.vector_utils import get_embedder, get_llm_judge
from utils.pg_utils import get_eval_data
import time
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import EvaluationDataset

eval_data = get_eval_data()

evaluator_llm = LangchainLLMWrapper(get_llm_judge())
embedder = get_embedder()

dataset= []
for row in eval_data:
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
    )

    print(result)
    dataset.clear()