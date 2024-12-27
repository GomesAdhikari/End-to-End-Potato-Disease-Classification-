from src.components.data_validation import DataValidation
from src.logger import info_logger

PIPELINE = "Data Validation Training Pipeline"


if __name__ == "__main__":
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")
    obj = DataValidation()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")