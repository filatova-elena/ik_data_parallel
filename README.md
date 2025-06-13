# Project description

Fine tune **RsNet-152** model using Data parallel on Tiny Imagenette dataset.

- Tiny imagenette dataset has been previously downloaded to `../imagenette` directory. To download dataset, do the following:
  ```
  wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
  tar -xvzf imagenette2.tgz
  ```
- This project runs in a venv environment, to set up the environment, do the following:
  ```
  # First, check if installed
  python3 -m venv --help

  # If not, install it
  sudo apt install sudo apt install python3-venv
  python3 -m venv ~/assignment-env

  # Activate it
  source ~/assignment-env/bin/activate

  # Install dependencies
  pip install torch torchvision tqdm ray[tune] mlflow scikit-learn virtualenv
  ```
- Creates **RsNet-152** model and trains it for a few epochs.
- Logs final classification report to MLFlow. To start MLFlow, do the following:
  ```
  mlflow ui --backend-store-uri sqlite:////home/ubuntu/mlruns.db --host 0.0.0.0 --port 5000
  ```

  
