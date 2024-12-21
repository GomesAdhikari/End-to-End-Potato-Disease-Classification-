import yaml
import sys
from dataclasses import dataclass
from src.utils.utils import read_yaml
from pathlib import Path
import os

config_path = r'C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\src\config\config.yaml'
config_obj = read_yaml(config_path)

@dataclass
class DataIngestionCofig:

    configuration = config_obj['DataIngestion']

    root_dir :Path = Path(configuration['root_dir'])
    data_source :Path =  Path(configuration['data_source'])
    data_dir :Path = Path(configuration['data_dir'])
    STATUS_FILE :Path = Path(configuration['STATUS_FILE'])
    

@dataclass
class DataValidationConfig:

    configuration = config_obj['DataValidation']

    root_dir :Path = Path(configuration['root_dir'])
    data_dir :Path = Path(configuration['data_dir'])
    STATUS_FILE :Path = Path(configuration['STATUS_FILE'])

@dataclass
class PreprocessingConfig:

    configuration = config_obj['Preprocessing']
    
    root_dir: Path = Path(configuration['root_dir'])
    data_path: Path = Path(configuration['data_path'])
    train_data_dir: Path = Path(configuration['train_data_dir'])
    val_data_dir: Path =  Path(configuration['val_data_dir'])
    test_data_dir: Path = Path(configuration['test_data_dir'])
    STATUS_FILE: Path = Path(configuration['STATUS_FILE'])

