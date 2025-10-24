import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.vector_utils import get_embedder, get_llm
from utils.pg_utils import get_eval_data
from ragas.metrics import DiscreteMetric

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness


from ragas.llms.base import BaseRagasLLM
from langchain.schema import LLMResult
from typing import Dict, Any

class ChatGroqRagasLLM(BaseRagasLLM):
    def __init__(self, model: str = "mixtral-8x7b", temperature: float = 0.2):
        self.client = Groq()
        self.model = model
        self.temperature = temperature

    def generate_text(self, prompt: str, run_config: Dict[str, Any] = None) -> LLMResult:
        """Generate a single text completion using Groq."""
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content.strip()
        return LLMResult(generations=[[{"text": text}]])

from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision

# Initialize your evaluator LLM
evaluator_llm = ChatGroqRagasLLM(model="llama3-70b-8192")

# Suppose you have a dataset in Ragas format
dataset = {
    "question": ["Who wrote Hamlet?"],
    "contexts": [["Shakespeare was an English playwright."]],
    "answer": ["It was written by Shakespeare."],
    "ground_truth": ["William Shakespeare wrote Hamlet."]
}

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_correctness, context_precision],
    llm=evaluator_llm
)

print(results)


eval_data = get_eval_data()
print(eval_data)
