from src.logger import error_logger, info_logger
from src.exception import ModelTrainingError
import tensorflow as tf
import os
from src.configuration.config_manager import ModelTrainingConfig, DataValidationConfig
from tensorflow.keras import layers, models
import pickle

class ModelTraining:
    def __init__(self):
        self.train_data_path = ModelTrainingConfig.train_data_path
        self.val_data_path = ModelTrainingConfig.val_data_path
        self.test_data_path = ModelTrainingConfig.test_data_path
        self.model_dir = ModelTrainingConfig.model_dir
        self.STATUS_FILE = ModelTrainingConfig.STATUS_FILE
        self.classes = len(os.listdir(DataValidationConfig.data_dir))
        
        # Model parameters
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 128
        self.CHANNELS = 3
        self.EPOCHS = 50

    def process_path(self, file_path, label):
        """Process a single image file path"""
        # Read the image file
        img = tf.io.read_file(file_path)
        # Decode the image
        img = tf.io.decode_jpeg(img, channels=self.CHANNELS)
        # Resize the image
        img = tf.image.resize(img, [self.IMAGE_SIZE, self.IMAGE_SIZE])
        # Convert to float32
        img = tf.cast(img, tf.float32)
        return img, label

    def load_dataset(self, data_path, is_training=False):
        """Load dataset from directory"""
        try:
            info_logger.info(f'Loading dataset from: {data_path}')
            
            # Get class names (directory names)
            class_names = sorted(os.listdir(data_path))
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            
            # Collect all file paths and labels
            file_paths = []
            labels = []
            
            for class_name in class_names:
                class_dir = os.path.join(data_path, class_name)
                class_idx = class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    file_paths.append(img_path)
                    labels.append(class_idx)
            
            # Create dataset from paths and labels
            dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            
            # Map the file paths to actual images
            dataset = dataset.map(
                self.process_path, 
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            if is_training:
                # Shuffle and repeat for training
                dataset = dataset.shuffle(buffer_size=1000)
                dataset = dataset.repeat()
            
            # Batch and prefetch
            dataset = dataset.batch(self.BATCH_SIZE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset, len(file_paths)
            
        except Exception as e:
            error_logger.error(f"Error loading dataset: {str(e)}")
            raise ModelTrainingError(f"Failed to load dataset: {str(e)}")

    def BuildModel(self):
        try:
            info_logger.info('Model Building Started!')

            # Data preprocessing layers
            preprocessing = tf.keras.Sequential([
                layers.Rescaling(1./255),
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
            ])

            # Model architecture
            model = models.Sequential([
                layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS)),
                preprocessing,
                
                # First convolution block
                layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                
                # Third convolution block
                layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.classes, activation='softmax'),
            ])

            info_logger.info('Model Building Completed.')
            return model
        
        except Exception as e:
            error_logger.error(f"Error building model: {str(e)}")
            raise ModelTrainingError(f"Failed to build model: {str(e)}")

    def Start(self):
        info_logger.info('Started model training reading data from preprocessing.')
        try:
            # Load datasets and get sizes
            train_ds, train_size = self.load_dataset(str(self.train_data_path), is_training=True)
            val_ds, val_size = self.load_dataset(str(self.val_data_path), is_training=False)
            test_ds, test_size = self.load_dataset(str(self.test_data_path), is_training=False)
            
            # Calculate steps
            steps_per_epoch = train_size // self.BATCH_SIZE
            validation_steps = val_size // self.BATCH_SIZE
            test_steps = test_size // self.BATCH_SIZE
            
            info_logger.info(f"Training steps per epoch: {steps_per_epoch}")
            info_logger.info(f"Validation steps: {validation_steps}")
            info_logger.info("Loaded datasets successfully")

            # Build and compile model
            model = self.BuildModel()
            model.summary()
            
            # Configure optimizer with learning rate
            
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )

            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # Train model with explicit steps
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )

            # Evaluate model
            test_loss, test_accuracy = model.evaluate(test_ds, steps=test_steps)
            info_logger.info(f'Test accuracy: {test_accuracy:.4f}')

            # Save training history
            history_file = os.path.join(self.model_dir, 'training_history.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(history.history, f)

            # Save final model
            model.save(os.path.join(self.model_dir, 'final_model.keras'))

            # Write success status
            with open(self.STATUS_FILE, 'w') as f:
                f.write("ModelTraining status: TRUE")

            info_logger.info('Model training completed successfully')

        except Exception as e:
            error_logger.error(f"Error in model training: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
if __name__ == '__main__':
    obj = ModelTraining()
    obj.Start()