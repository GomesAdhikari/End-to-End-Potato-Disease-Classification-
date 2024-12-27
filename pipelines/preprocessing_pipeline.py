from src.components.preprocessing import Preprocessing
from src.logger import info_logger

PIPELINE = "Preprocessing Pipeline"

if __name__ == '__main__':
    info_logger.info(f">>>>>>>> {PIPELINE} sttarted <<<<<<<<<")

    obj = Preprocessing()
    obj.Start()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")