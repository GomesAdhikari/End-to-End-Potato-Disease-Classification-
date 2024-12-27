from src.components.model_training import ModelTraining
from src.logger import info_logger

PIPELINE = 'Model Evaluation Pipeline'
if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    evaluation = ModelTraining()
    evaluation.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")