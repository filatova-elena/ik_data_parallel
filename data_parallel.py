import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as cfg
from sklearn.metrics import classification_report
import mlflow
import time

# Define train function
def train(model, train_loader, optimizer, criterion, rank, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, leave=False, desc=f"Training Epoch {epoch+1}/{cfg.epochs}"):
        inputs, labels = batch
        inputs = inputs.to(rank)
        labels = labels.to(rank)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if cfg.do_data_parallel and torch.cuda.device_count() > 1:
          loss = loss.mean()

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Track peak memory after each batch
        batch_mem = torch.cuda.max_memory_allocated(rank) / 1e9  # Convert to GB
        print(f"\nBatch memory usage {batch_mem:.2f} GB used")

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}, Train Loss: {avg_train_loss}, Memory Usage: {torch.cuda.max_memory_allocated(rank) / 1e9:.2f} GB")
    return avg_train_loss

# Define test function
def test(model, test_loader, rank):
    model.eval()
    total_accuracy = 0.0

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False, desc="Testing"):
            inputs, labels = batch

            inputs = inputs.to(rank)
            labels = labels.to(rank)

            outputs = model(inputs)
            
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions = (predicted_labels == labels).sum().item()
            total_accuracy += correct_predictions  # Accumulate correct predictions

            all_predicted_labels.extend(predicted_labels.tolist())  # Append predicted labels to the list
            all_true_labels.extend(labels.tolist())  # Append true labels to the list

    accuracy = total_accuracy / len(test_loader.dataset)
    print(f'Accuracy on Test Set: {accuracy:.4f}')  # Print the accuracy
    return all_true_labels, all_predicted_labels, accuracy

def create_model(rank):
    # Load the pretrained ResNet-152 model
    model = models.resnet152(pretrained=True)

    # Replace the fully connected layer to match the number of classes (10 for tiny Imagenette)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, cfg.classes_count)

    # Move the model to the appropriate device
    model.to(rank)

    print(f"Devices available: {torch.cuda.device_count()}")

    if torch.cuda.device_count() > 1 and cfg.do_data_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids = cfg.visible_devices)
    
    return model

def create_data_loaders():
    # Define transformations for training and validation
    train_transforms = transforms.Compose([
        transforms.Resize((cfg.image_size,cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((cfg.image_size,cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(cfg.data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size = cfg.per_device_batch_size * len(cfg.visible_devices), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = cfg.per_device_batch_size * len(cfg.visible_devices), shuffle=False)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader

if __name__ == '__main__':
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)  # Replace with your EC2 public IP
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    # Define model and training parameters
    rank = torch.device(cfg.device)
    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)

    train_loader, val_loader = create_data_loaders()
    
    model = create_model(rank)
    optimizer = optim.Adam(model.parameters(), lr = cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    training_start_time = time.time()   # Start timer for training

    with mlflow.start_run():
        
        mlflow.log_params({
            "batch_size": cfg.per_device_batch_size * len(cfg.visible_devices),
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "visible_devices": cfg.visible_devices,
            "memory_limit": cfg.memory_limit,
        })
    

        for epoch in tqdm(range(cfg.epochs)):
            start_time = time.time()  # Start timer

            train_loss = train(model, train_loader, optimizer, criterion, rank, epoch)
            all_true_labels, all_predicted_labels, accuracy = test(model, val_loader, rank)

            end_time = time.time()  # End timer
            epoch_duration = end_time - start_time

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)
            mlflow.log_metric("epoch_duration_seconds", epoch_duration, step=epoch)

    # Log classification report
    report = classification_report(all_true_labels, all_predicted_labels, target_names=train_loader.dataset.classes)
    print("\n\nClassification Report:")
    print(report)
    print("==========================================\n\n")

    print(f"Final Accuracy: {accuracy:.4f}")  # Print the accuracy
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - training_start_time))}")

    with open(cfg.report_path, "w") as f:
        f.write(report)
        mlflow.log_artifact(cfg.report_path)

    # Log model
    mlflow.pytorch.log_model(
        pytorch_model=model.module if hasattr(model, 'module') else model,
        artifact_path=cfg.artifact_path,
        registered_model_name=cfg.registered_model_name
    )