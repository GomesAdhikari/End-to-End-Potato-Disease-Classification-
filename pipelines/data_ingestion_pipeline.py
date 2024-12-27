from src.components.data_ingestion import DataIngestion
from src.logger import info_logger

PIPELINE = "Data Ingestion Pipeline"

if __name__=='__main__':
    
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = DataIngestion()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")