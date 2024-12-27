from src.logger import info_logger, error_logger
from src.exception import DataValidationError
import os
from src.configuration.config_manager import PreprocessingConfig
import tensorflow as tf
import pickle
import numpy as np
from pathlib import Path
import shutil
from PIL import Image

IMAGE_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 32

class Preprocessing:
    def __init__(self):
        self.root_dir = PreprocessingConfig.root_dir
        self.data_path = PreprocessingConfig.data_path
        self.train_data_dir = PreprocessingConfig.train_data_dir
        self.val_data_dir = PreprocessingConfig.val_data_dir
        self.test_data_dir = PreprocessingConfig.test_data_dir
        self.STATUS_FILE = PreprocessingConfig.STATUS_FILE
        self.ds = False

    def process_and_save_image(self, source_path, target_path):
        """Process a single image and save it to the target path"""
        try:
            # Read and resize image using PIL
            with Image.open(source_path) as img:
                img = img.convert('RGB')  # Ensure RGB format
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                # Create target directory if it doesn't exist
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # Save the processed image
                img.save(target_path, quality=95)
            return True
        except Exception as e:
            error_logger.error(f"Error processing image {source_path}: {str(e)}")
            return False

    def load_data(self):
        """Load and prepare the dataset paths"""
        try:
            info_logger.info('Started loading image paths')
            
            # Get all class directories
            data_path = Path(self.data_path)
            class_dirs = sorted([d for d in data_path.glob('*') if d.is_dir()])
            
            if not class_dirs:
                raise ValueError(f"No class directories found in {self.data_path}")
            
            # Create class mappings
            self.class_names = [d.name for d in class_dirs]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            
            # Collect image paths
            image_paths = []
            labels = []
            valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            
            for class_dir in class_dirs:
                class_idx = self.class_to_idx[class_dir.name]
                info_logger.info(f'Processing class: {class_dir.name}')
                
                for img_path in class_dir.glob('*'):
                    if img_path.suffix in valid_extensions:
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
            
            if not image_paths:
                raise ValueError("No valid images found")
            
            # Convert to numpy arrays for easier handling
            self.image_paths = np.array(image_paths)
            self.labels = np.array(labels)
            
            info_logger.info(f'Found {len(image_paths)} images from {len(class_dirs)} classes')
            return True
            
        except Exception as e:
            error_logger.error(f"Error in load_data: {str(e)}")
            return False

    def split_data(self, train_split=0.8, val_split=0.1, test_split=0.1):
        """Split the dataset into train, validation, and test sets"""
        try:
            assert (train_split + test_split + val_split) == 1
            
            # Create indices and shuffle
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            
            # Calculate split sizes
            train_size = int(train_split * len(indices))
            val_size = int(val_split * len(indices))
            
            # Split indices
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create split datasets
            splits = {
                'train': (self.image_paths[train_indices], self.labels[train_indices]),
                'val': (self.image_paths[val_indices], self.labels[val_indices]),
                'test': (self.image_paths[test_indices], self.labels[test_indices])
            }
            
            return splits
            
        except Exception as e:
            error_logger.error(f"Error in split_data: {str(e)}")
            raise

    def save_datasets(self, splits):
        """Save the processed datasets as files"""
        try:
            info_logger.info('Started saving datasets')
            
            # Create base directories
            save_dirs = {
                'train': self.train_data_dir,
                'val': self.val_data_dir,
                'test': self.test_data_dir
            }
            
            # Process and save each split
            for split_name, (paths, labels) in splits.items():
                base_dir = save_dirs[split_name]
                
                # Create split directory
                os.makedirs(base_dir, exist_ok=True)
                
                # Create class directories
                for class_name in self.class_names:
                    os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)
                
                # Process and save images
                for img_path, label in zip(paths, labels):
                    # Get source image filename and class name
                    filename = os.path.basename(img_path)
                    class_name = self.class_names[label]
                    
                    # Create target path
                    target_path = os.path.join(base_dir, class_name, filename)
                    
                    # Process and save image
                    self.process_and_save_image(img_path, target_path)
            
            # Save class mapping
            class_mapping_file = os.path.join(self.root_dir, 'class_mapping.pkl')
            with open(class_mapping_file, 'wb') as f:
                pickle.dump(self.class_to_idx, f)
            
            info_logger.info('Successfully saved all datasets and class mapping')
            return True
            
        except Exception as e:
            error_logger.error(f"Error saving datasets: {str(e)}")
            return False

    def Start(self):
        try:
            os.makedirs(self.root_dir, exist_ok=True)
            info_logger.info('Started Data Preprocessing')
            
            # Load data paths
            if not self.load_data():
                raise DataValidationError("Failed to load data")
            
            # Split datasets
            splits = self.split_data()
            
            # Save datasets
            if self.save_datasets(splits):
                # Write success status
                with open(self.STATUS_FILE, 'w') as f:
                    f.write("DataPreprocessing status: TRUE")
                info_logger.info('Data preprocessing completed successfully')
            else:
                raise DataValidationError("Failed to save datasets")
                
        except Exception as e:
            error_logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == '__main__':
    obj = Preprocessing()
    obj.Start()