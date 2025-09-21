import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import mlflow
from indexer import Indexer
from utils.logger import get_logger
from utils.sqlite_utils import add_indexed_file, get_indexed_files
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class IndexDriver():
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def start(self):
        data_dir = os.getenv('DATA_LOCATION', '')
        files_in_dir = os.listdir(data_dir)
        files_in_dir.remove('.DS_Store')
        indexed_documents = get_indexed_files()
        files_to_index = set(files_in_dir) - set(indexed_documents)

        if len(files_to_index) > 0:
            self.logger.info("Starting indexing process")
            for file_name in files_to_index:
                if ".pdf" in file_name:
                    self.logger.info("Now Indexing %s", file_name)
                    Indexer().index(os.path.join(data_dir, file_name))
                    add_indexed_file(file_name)
            self.logger.info("Indexing complete")
        else:
            self.logger.info("No new files to index")


if __name__ == "__main__":
    IndexDriver().start()