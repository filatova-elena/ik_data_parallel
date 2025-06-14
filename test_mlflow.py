import mlflow
import time
import config as cfg
from tqdm import tqdm
from torchvision import models

# This is just a placeholder to test the MLflow logging functionality
# Logs a few metrics and a model without actual training

if __name__ == '__main__':
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)  # Replace with your EC2 public IP
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    # Load the pretrained ResNet-152 model
    model = models.resnet152(pretrained=True)

    with mlflow.start_run(): 
        mlflow.log_params({
            "batch_size": cfg.per_device_batch_size * len(cfg.visible_devices),
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "visible_devices": cfg.visible_devices,
            "memory_limit": cfg.memory_limit,
        })

        training_start_time = time.time()  # Start timer

        for epoch in tqdm(range(cfg.epochs)):
            start_time = time.time()  # Start timer

            train_loss = 0.1
            accuracy = 9.5

            end_time = time.time()  # End timer
            epoch_duration = end_time - start_time

            # Log metrics
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_accuracy", float(accuracy), step=epoch)
            mlflow.log_metric("epoch_duration_seconds", epoch_duration, step=epoch)

        # Log classification report
        report = "classification report placeholder"

        print(f"Final Accuracy: {accuracy:.4f}")  # Print the accuracy

        with open(cfg.report_path, "w") as f:
            f.write(report)
            mlflow.log_artifact(cfg.report_path)

        # Log model
        mlflow.pytorch.log_model(model, cfg.artifact_path, registered_model_name=cfg.registered_model_name)

        # Log total training time and format it
        print(f"Total training time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start_time))}")