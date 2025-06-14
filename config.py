import os

do_data_parallel = True
data_dir = '../imagenette2'
report_path = '../classification_report.txt'

per_device_batch_size = 8
learning_rate = 1e-3
epochs = 2

device = 'cuda'

# Number of classes in the image dataset
classes_count = 10
image_size = 640

# Per process memory fraction
memory_limit = 1.0

visible_devices = [0,1,2,3]

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow_experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "0")
artifact_path = 'model_artifacts'
registered_model_name = 'imagenette2_classifier'