from src.components.model_evaluation import ModelEvaluation
from src.logger import info_logger

PIPELINE = 'Model Evaluation Pipeline'
if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    evaluation = ModelEvaluation()
    evaluation.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")