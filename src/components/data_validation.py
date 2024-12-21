from src.logger import info_logger, error_logger
from src.exception import DataValidationError
import os
from src.configuration.config_manager import DataValidationConfig
from src.utils.utils import read_yaml
import sys


class DataValidation:

    def __init__(self):

        self.root_dir = DataValidationConfig.root_dir
        self.data_dir = DataValidationConfig.data_dir
        self.STATUS_FILE = DataValidationConfig.STATUS_FILE

    def Start(self):

        info_logger.info('Data validation has been started.')

        try:
            os.makedirs(self.root_dir,exist_ok=True)
            list = os.listdir(self.data_dir)
            print(list)
            diseases = read_yaml(r'C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\src\config\schema.yaml')
            diseases = diseases['DISEASES']
            print(diseases)
            if set(list) == set(diseases):
                info_logger.info("All matching features found.")
            else:
                error_logger.error('Matching Features not found',DataValidationError)
                sys.exit("Program terminated due to missing matching features.")  # Terminate the program if not matching
            data_dir = self.data_dir
            subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir / f)]
            
            # Ensure there are exactly 4 subfolders
            if len(subfolders) != 5:
                sys.exit("Error: The directory does not contain exactly 4 subfolders.")
            valid_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

            # Check each subfolder for JPG files
            for folder_name in subfolders:
                folder_path = data_dir / folder_name

                # List all files in the folder
                files = os.listdir(folder_path)
                
                # Filter out image files and non-image files
                image_files = [f for f in files if any(f.lower().endswith(ext) for ext in valid_image_extensions)]
                non_image_files = [f for f in files if not any(f.lower().endswith(ext) for ext in valid_image_extensions)]

                # If there are non-image files or no image files in the folder
                if non_image_files:
                    error_logger.error(f"Non-image files found in folder {folder_name}: {non_image_files}", DataValidationError)
                    sys.exit("Program terminated due to non-image files.")
                elif not image_files:
                    error_logger.error(f"No image files found in folder {folder_name}", DataValidationError)
                    sys.exit("Program terminated due to missing image files.")

                # Log if folder only contains image files
                print(f"Folder {folder_name} contains only image files.")

            with open(self.STATUS_FILE, 'w') as f:
                 f.write("DataIngestion status: TRUE")
                 info_logger.info('Data Ingested Successfully')

        except Exception as e:
            error_logger.error(e,DataValidationError)
