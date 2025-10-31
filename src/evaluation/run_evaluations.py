import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from evaluator import Evaluator
from utils.pg_utils import add_eval_metrics


if __name__ == "__main__":
    status = Evaluator().evaluate()



