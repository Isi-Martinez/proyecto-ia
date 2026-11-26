# AI Project - Online Purchase Intention Prediction

## Project Structure

- `proyecto.py`: Main script for model training, hyperparameter tuning, and evaluation.
- `online_shoppers_intention.csv`: Input dataset.
- `requirements.txt`: Python dependencies required to run the project.
- `results.txt`: File automatically generated when running the script, containing model metrics and results.

## Prerequisites

- Python 3.9 or higher.
- `pip` installed.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Run the main script from the project's root directory:

```bash
python proyecto.py
```

The evaluation results (best hyperparameters, overfitting check, test set metrics, confusion matrix, and classification report) **are not printed to the terminal**; instead, they are saved directly to the `results.txt` file.
