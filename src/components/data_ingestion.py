from src.exception import DataIngestionError
from src.configuration.config_manager import DataIngestionCofig
from src.logger import info_logger,error_logger
import tensorflow as tf
import os
from dataclasses import dataclass
import zipfile

class DataIngestion:

    def __init__(self):

        self.root_dir = DataIngestionCofig.root_dir
        self.data_source = DataIngestionCofig.data_source
        self.data_dir = DataIngestionCofig.data_dir
        self.STATUS_FILE = DataIngestionCofig.STATUS_FILE

    
    def Start(self):

        info_logger.info('Data Ingestion has been started.')

        try :   
            os.makedirs(self.root_dir,exist_ok=True)

            with zipfile.ZipFile(self.data_source,'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                info_logger.info('Extracted all files')

            with open(self.STATUS_FILE, 'w') as f:
                 f.write("DataIngestion status: TRUE")
            info_logger.info('Data Ingested Successfully')

        except Exception as e:
            error_logger.error(e,DataIngestionError)






