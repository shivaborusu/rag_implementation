import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.vector_utils import get_embedder, get_llm
from utils.pg_utils import get_eval_data

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness


from ragas import EvaluationDataset

eval_data = get_eval_data()

dataset = []

for row in eval_data:
    dataset.append(
        {
            "user_input": row[1],
            "retrieved_contexts": row[2],
            "response": row[3]
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)

evaluator_llm = LangchainLLMWrapper(get_llm())

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[Faithfulness()],
    llm=evaluator_llm,
)

print(result)