from src.configuration.config_manager import ModelEvaluationConfig
from src.logger import info_logger, error_logger
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import os

class ModelEvaluation:

    def __init__(self):
        # Initializing configuration parameters
        self.model_path = ModelEvaluationConfig.model_path
        self.test_data = ModelEvaluationConfig.test_data  # Assuming test data path is configured
        self.root_dir = ModelEvaluationConfig.root_dir
        self.RESULTS = ModelEvaluationConfig.RESULTS  # Assuming where you want to store results

    def load_model(self):
        """Load the trained model."""
        try:
            model = load_model(self.model_path)
            info_logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            error_logger.error(f"Error loading model from {self.model_path}: {str(e)}")
            raise e

    def load_class_mapping(self):
        """Load the class mapping from pickle."""
        try:
            with open(ModelEvaluationConfig.class_mapping, "rb") as file:
                class_to_idx = pickle.load(file)
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
            info_logger.info(f"Class mapping loaded from {ModelEvaluationConfig.class_mapping}")
            return idx_to_class
        except Exception as e:
            error_logger.error(f"Error loading class mapping: {str(e)}")
            raise e

    def load_dataset(self):
        """Load the test dataset."""
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.test_data,
                seed=123,
                shuffle=False,  # Keep order to match predictions with labels
                image_size=(128, 128),
                batch_size=32
            )
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            info_logger.info(f"Test dataset loaded from {self.test_data}")
            return dataset
        except Exception as e:
            error_logger.error(f"Error loading dataset from {self.test_data}: {str(e)}")
            raise e

    def predict_on_dataset(self, model, dataset):
        """Make predictions on the dataset."""
        try:
            predictions = model.predict(dataset)
            info_logger.info("Predictions made on the dataset")
            return predictions
        except Exception as e:
            error_logger.error(f"Error making predictions: {str(e)}")
            raise e

    def map_predictions_to_labels(self, predictions, idx_to_class, dataset):
        """Map predictions to actual labels."""
        try:
            predicted_indices = tf.argmax(predictions, axis=1).numpy()  # Convert Tensor to NumPy array

            actual_labels = []
            predicted_labels = []

            for batch_index, (images, labels) in enumerate(dataset):
                actual_indices = labels.numpy()  # Convert Tensor to NumPy array
                for sample_index in range(len(actual_indices)):
                    actual_class = idx_to_class[int(actual_indices[sample_index])]
                    predicted_class = idx_to_class[int(predicted_indices[batch_index * 32 + sample_index])]

                    # Append the actual and predicted labels
                    actual_labels.append(actual_class)
                    predicted_labels.append(predicted_class)

            return actual_labels, predicted_labels
        except Exception as e:
            error_logger.error(f"Error mapping predictions to labels: {str(e)}")
            raise e

    def evaluate_model(self, actual_labels, predicted_labels):
        """Evaluate model performance (Accuracy and classification report)."""
        try:
            accuracy = accuracy_score(actual_labels, predicted_labels)
            info_logger.info(f"Accuracy: {accuracy:.4f}")

            report = classification_report(actual_labels, predicted_labels)
            info_logger.info(f"Classification Report: \n{report}")
            return accuracy, report
        except Exception as e:
            error_logger.error(f"Error evaluating model: {str(e)}")
            raise e
        
    def save_results(self, accuracy, report):
        """Save evaluation results to the specified file."""
        try:
            # Ensure the directory exists, create it if not
            results_dir = os.path.dirname(self.RESULTS)
            if not os.path.exists(results_dir) and results_dir:
                os.makedirs(results_dir)
                info_logger.info(f"Created directory: {results_dir}")

            # Open the results file in write mode and save the evaluation results
            with open(self.RESULTS, 'w') as result_file:
                result_file.write(f"Accuracy: {accuracy:.4f}\n")
                result_file.write(f"\nClassification Report:\n{report}")
            info_logger.info(f"Evaluation results saved to {self.RESULTS}")

        except Exception as e:
            error_logger.error(f"Error saving results to {self.RESULTS}: {str(e)}")
            raise e
        
    def Start(self):
        info_logger.info('Started Model evaluation')
        try:
            # Load model
            model = self.load_model()

            # Load class mapping
            idx_to_class = self.load_class_mapping()

            # Load dataset
            dataset = self.load_dataset()

            # Make predictions
            predictions = self.predict_on_dataset(model, dataset)

            # Map predictions to actual labels
            actual_labels, predicted_labels = self.map_predictions_to_labels(predictions, idx_to_class, dataset)

            # Evaluate model
            accuracy, report = self.evaluate_model(actual_labels, predicted_labels)

            # Save results to file
            self.save_results(accuracy, report)

        except Exception as e:
            error_logger.error(f"Model evaluation failed: {str(e)}")
            raise e


