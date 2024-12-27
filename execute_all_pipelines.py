from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.preprocessing import Preprocessing
from src.components.model_training import ModelTraining
from src.logger import info_logger


PIPELINE = "Data Ingestion Pipeline"

if __name__=='__main__':
    
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = DataIngestion()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = "Data Validation Pipeline"


if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    obj = DataValidation()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = "Preprocessing Pipeline"

if __name__ == '__main__':
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    obj = Preprocessing()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = 'Model Training Pipeline'

if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    evaluation = ModelTraining()
    evaluation.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

PIPELINE = 'Model Evaluation Pipeline'

if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    evaluation = ModelEvaluation()
    evaluation.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")