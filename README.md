# Simple Machine Learning Project

This project demonstrates a simple machine learning workflow using scikit-learn.

## Setup

1. Install UV (if not already installed):
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv add -r requirements.txt
```

## Running the Model

To train and evaluate the model, run:
```bash
python train_model.py
```

The script will:
- Generate synthetic data
- Train a logistic regression model
- Print the model's accuracy and classification report
- Generate a feature importance plot (saved as 'feature_importance.png')

## Project Structure

- `train_model.py`: Main script for training and evaluating the model
- `requirements.txt`: Project dependencies
- `feature_importance.png`: Generated plot showing feature importance (created after running the script) 